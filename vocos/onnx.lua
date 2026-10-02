-- vocos/onnx.lua -- reads .onnx model files in pure luajit.
--
-- an .onnx file is a serialised protobuf message. protobuf's wire format
-- is small and stable: every field is a varint tag holding a field number
-- and a wire type, followed by a payload whose shape the wire type alone
-- determines. that means a reader can walk the whole file knowing only
-- the handful of field numbers it actually cares about, and skip every
-- other field without understanding it. no schema compiler, no generated
-- code, no external library.
--
-- what this module extracts:
--   model.init     array of initializer tensors (the trained weights)
--   model.by_name  the same tensors indexed by name
--   model.nodes    graph nodes with op_type, inputs, outputs, attributes
--                  (only when opts.nodes is set, since it costs time)
--   model.inputs / model.outputs   graph input and output names
--
-- raw tensor payloads are copied into freshly allocated, correctly
-- aligned ffi arrays rather than aliased into the file buffer: the file
-- offset of raw_data has no alignment guarantee, and casting a misaligned
-- byte pointer to float* is undefined on architectures that care.
-- protobuf stores raw_data little endian, which matches every platform
-- this project targets.
--
-- field numbers below come from the onnx protobuf definition and are part
-- of its stable public format.

local ffi = require("ffi")

local M = {}

-- tensor element types, from the onnx TensorProto.DataType enum
local FLOAT, UINT8, INT8, UINT16, INT16, INT32, INT64 = 1, 2, 3, 4, 5, 6, 7
local BOOL, FLOAT16, DOUBLE, UINT32, UINT64 = 9, 10, 11, 12, 13

local DTYPE_NAME = {
  [FLOAT] = "float32", [UINT8] = "uint8", [INT8] = "int8",
  [UINT16] = "uint16", [INT16] = "int16", [INT32] = "int32",
  [INT64] = "int64", [BOOL] = "bool", [FLOAT16] = "float16",
  [DOUBLE] = "float64", [UINT32] = "uint32", [UINT64] = "uint64",
}

local DTYPE_SIZE = {
  [FLOAT] = 4, [UINT8] = 1, [INT8] = 1, [UINT16] = 2, [INT16] = 2,
  [INT32] = 4, [INT64] = 8, [BOOL] = 1, [FLOAT16] = 2, [DOUBLE] = 8,
  [UINT32] = 4, [UINT64] = 8,
}

function M.dtype_name(dt)
  return DTYPE_NAME[dt] or ("unknown(" .. tostring(dt) .. ")")
end

function M.dtype_size(dt)
  return DTYPE_SIZE[dt]
end

-- ---------------------------------------------------------- wire format

-- a cursor is {buf = uint8_t*, pos = number, len = number}. positions are
-- 0 based offsets into buf.

local function read_varint(cur)
  local buf = cur.buf
  local pos = cur.pos
  local value = 0.0
  local scale = 1.0
  while true do
    if pos >= cur.len then
      error("onnx: varint runs past end of buffer")
    end
    local b = buf[pos]
    pos = pos + 1
    value = value + (b % 128) * scale
    if b < 128 then
      break
    end
    scale = scale * 128.0
    if scale > 2 ^ 63 then
      error("onnx: varint too long")
    end
  end
  cur.pos = pos
  return value
end

local function read_fixed32(cur)
  if cur.pos + 4 > cur.len then
    error("onnx: fixed32 runs past end of buffer")
  end
  local v = ffi.new("uint32_t[1]")
  ffi.copy(v, cur.buf + cur.pos, 4)
  cur.pos = cur.pos + 4
  return v[0]
end

local function read_fixed64(cur)
  if cur.pos + 8 > cur.len then
    error("onnx: fixed64 runs past end of buffer")
  end
  local v = ffi.new("uint64_t[1]")
  ffi.copy(v, cur.buf + cur.pos, 8)
  cur.pos = cur.pos + 8
  return v[0]
end

-- returns field number and wire type
local function read_tag(cur)
  local key = read_varint(cur)
  local wire = key % 8
  local field = (key - wire) / 8
  return field, wire
end

-- returns start offset and length of a length delimited payload, leaving
-- the cursor positioned after it
local function read_bytes_span(cur)
  local n = read_varint(cur)
  local start = cur.pos
  if start + n > cur.len then
    error("onnx: length delimited field runs past end of buffer")
  end
  cur.pos = start + n
  return start, n
end

local function read_string(cur)
  local start, n = read_bytes_span(cur)
  return ffi.string(cur.buf + start, n)
end

local function skip_field(cur, wire)
  if wire == 0 then
    read_varint(cur)
  elseif wire == 1 then
    cur.pos = cur.pos + 8
  elseif wire == 2 then
    read_bytes_span(cur)
  elseif wire == 5 then
    cur.pos = cur.pos + 4
  else
    error("onnx: unsupported wire type " .. tostring(wire))
  end
  if cur.pos > cur.len then
    error("onnx: skip ran past end of buffer")
  end
end

-- reads a repeated numeric field that may arrive either packed (one
-- length delimited blob) or unpacked (one tag per value). appends to out.
local function read_repeated_varint(cur, wire, out)
  if wire == 2 then
    local start, n = read_bytes_span(cur)
    local sub = {buf = cur.buf, pos = start, len = start + n}
    while sub.pos < sub.len do
      out[#out + 1] = read_varint(sub)
    end
  else
    out[#out + 1] = read_varint(cur)
  end
end

-- ------------------------------------------------------------- tensors

-- converts an ieee half precision bit pattern to a lua number
local function half_to_float(h)
  local sign = 1.0
  if h >= 32768 then
    sign = -1.0
    h = h - 32768
  end
  local expo = math.floor(h / 1024)
  local mant = h % 1024
  if expo == 0 then
    if mant == 0 then
      return sign * 0.0
    end
    return sign * mant * 2 ^ -24
  elseif expo == 31 then
    if mant == 0 then
      return sign * math.huge
    end
    return 0.0 / 0.0
  end
  return sign * (1.0 + mant / 1024.0) * 2 ^ (expo - 15)
end

local function count_dims(dims)
  local n = 1
  for i = 1, #dims do
    n = n * dims[i]
  end
  return n
end

-- turns a parsed TensorProto into its final numeric array. float and
-- float16 both become float arrays, integer types keep their width.
local function materialise(t)
  local n = count_dims(t.dims)
  t.n = n
  local dt = t.dtype

  if t.raw_start then
    local size = DTYPE_SIZE[dt]
    if not size then
      error("onnx: tensor " .. tostring(t.name) .. " has unsupported dtype " .. tostring(dt))
    end
    if t.raw_len ~= n * size then
      error(string.format(
        "onnx: tensor %s raw_data is %d bytes but dims imply %d elements of %d bytes",
        tostring(t.name), t.raw_len, n, size))
    end
    if dt == FLOAT then
      t.data = ffi.new("float[?]", n)
      ffi.copy(t.data, t.src + t.raw_start, t.raw_len)
      t.kind = "float"
    elseif dt == FLOAT16 then
      local tmp = ffi.new("uint16_t[?]", n)
      ffi.copy(tmp, t.src + t.raw_start, t.raw_len)
      t.data = ffi.new("float[?]", n)
      for i = 0, n - 1 do
        t.data[i] = half_to_float(tonumber(tmp[i]))
      end
      t.kind = "float"
    elseif dt == INT64 then
      t.data = ffi.new("int64_t[?]", n)
      ffi.copy(t.data, t.src + t.raw_start, t.raw_len)
      t.kind = "int64"
    elseif dt == INT32 then
      t.data = ffi.new("int32_t[?]", n)
      ffi.copy(t.data, t.src + t.raw_start, t.raw_len)
      t.kind = "int32"
    else
      t.data = ffi.new("uint8_t[?]", t.raw_len)
      ffi.copy(t.data, t.src + t.raw_start, t.raw_len)
      t.kind = "raw"
    end
    t.raw_start, t.raw_len, t.src = nil, nil, nil
    return t
  end

  -- typed repeated fields instead of raw_data
  if t.float_data and #t.float_data > 0 then
    t.data = ffi.new("float[?]", n)
    for i = 1, n do
      t.data[i - 1] = t.float_data[i] or 0.0
    end
    t.kind = "float"
  elseif t.int64_data and #t.int64_data > 0 then
    t.data = ffi.new("int64_t[?]", n)
    for i = 1, n do
      t.data[i - 1] = t.int64_data[i] or 0
    end
    t.kind = "int64"
  elseif t.int32_data and #t.int32_data > 0 then
    t.data = ffi.new("int32_t[?]", n)
    for i = 1, n do
      t.data[i - 1] = t.int32_data[i] or 0
    end
    t.kind = "int32"
  else
    t.data = nil
    t.kind = "empty"
  end
  t.float_data, t.int64_data, t.int32_data = nil, nil, nil
  return t
end

-- TensorProto: dims=1, data_type=2, float_data=4, int32_data=5,
-- int64_data=7, name=8, raw_data=9
local function parse_tensor(cur, stop, src)
  local t = {dims = {}, dtype = 0, name = ""}
  while cur.pos < stop do
    local field, wire = read_tag(cur)
    if field == 1 then
      read_repeated_varint(cur, wire, t.dims)
    elseif field == 2 then
      t.dtype = read_varint(cur)
    elseif field == 4 then
      t.float_data = t.float_data or {}
      if wire == 2 then
        local start, n = read_bytes_span(cur)
        local sub = {buf = cur.buf, pos = start, len = start + n}
        while sub.pos < sub.len do
          local v = ffi.new("float[1]")
          ffi.copy(v, sub.buf + sub.pos, 4)
          sub.pos = sub.pos + 4
          t.float_data[#t.float_data + 1] = v[0]
        end
      else
        local v = ffi.new("uint32_t[1]")
        v[0] = read_fixed32(cur)
        local f = ffi.new("float[1]")
        ffi.copy(f, v, 4)
        t.float_data[#t.float_data + 1] = f[0]
      end
    elseif field == 5 then
      t.int32_data = t.int32_data or {}
      read_repeated_varint(cur, wire, t.int32_data)
    elseif field == 7 then
      t.int64_data = t.int64_data or {}
      read_repeated_varint(cur, wire, t.int64_data)
    elseif field == 8 then
      t.name = read_string(cur)
    elseif field == 9 then
      local start, n = read_bytes_span(cur)
      t.raw_start, t.raw_len, t.src = start, n, src
    else
      skip_field(cur, wire)
    end
  end
  return materialise(t)
end

-- --------------------------------------------------------- graph nodes

-- AttributeProto: name=1, f=2, i=3, s=4, t=5, type=20, floats=7, ints=8
local function parse_attribute(cur, stop, src)
  local a = {}
  while cur.pos < stop do
    local field, wire = read_tag(cur)
    if field == 1 then
      a.name = read_string(cur)
    elseif field == 2 then
      local v = ffi.new("uint32_t[1]")
      v[0] = read_fixed32(cur)
      local f = ffi.new("float[1]")
      ffi.copy(f, v, 4)
      a.f = f[0]
    elseif field == 3 then
      a.i = read_varint(cur)
    elseif field == 4 then
      a.s = read_string(cur)
    elseif field == 5 then
      local start, n = read_bytes_span(cur)
      local sub = {buf = cur.buf, pos = start, len = cur.len}
      a.t = parse_tensor(sub, start + n, src)
    elseif field == 7 then
      a.floats = a.floats or {}
      if wire == 2 then
        local start, n = read_bytes_span(cur)
        local sub = {buf = cur.buf, pos = start, len = start + n}
        while sub.pos < sub.len do
          local v = ffi.new("float[1]")
          ffi.copy(v, sub.buf + sub.pos, 4)
          sub.pos = sub.pos + 4
          a.floats[#a.floats + 1] = v[0]
        end
      else
        local v = ffi.new("uint32_t[1]")
        v[0] = read_fixed32(cur)
        local f = ffi.new("float[1]")
        ffi.copy(f, v, 4)
        a.floats[#a.floats + 1] = f[0]
      end
    elseif field == 8 then
      a.ints = a.ints or {}
      read_repeated_varint(cur, wire, a.ints)
    else
      skip_field(cur, wire)
    end
  end
  return a
end

-- NodeProto: input=1, output=2, name=3, op_type=4, attribute=5, domain=7
local function parse_node(cur, stop, src)
  local node = {input = {}, output = {}, attr = {}}
  while cur.pos < stop do
    local field, wire = read_tag(cur)
    if field == 1 then
      node.input[#node.input + 1] = read_string(cur)
    elseif field == 2 then
      node.output[#node.output + 1] = read_string(cur)
    elseif field == 3 then
      node.name = read_string(cur)
    elseif field == 4 then
      node.op_type = read_string(cur)
    elseif field == 5 then
      local start, n = read_bytes_span(cur)
      local sub = {buf = cur.buf, pos = start, len = cur.len}
      local a = parse_attribute(sub, start + n, src)
      if a.name then
        node.attr[a.name] = a
      end
    else
      skip_field(cur, wire)
    end
  end
  return node
end

-- ValueInfoProto: name=1. the type is skipped, names are what callers
-- actually need to line inputs up with a session.
local function parse_value_info(cur, stop)
  local name = nil
  while cur.pos < stop do
    local field, wire = read_tag(cur)
    if field == 1 then
      name = read_string(cur)
    else
      skip_field(cur, wire)
    end
  end
  return name
end

-- GraphProto: node=1, name=2, initializer=5, input=11, output=12
local function parse_graph(cur, stop, src, opts, model)
  while cur.pos < stop do
    local field, wire = read_tag(cur)
    if field == 1 then
      local start, n = read_bytes_span(cur)
      if opts.nodes then
        local sub = {buf = cur.buf, pos = start, len = cur.len}
        model.nodes[#model.nodes + 1] = parse_node(sub, start + n, src)
      end
    elseif field == 2 then
      model.graph_name = read_string(cur)
    elseif field == 5 then
      local start, n = read_bytes_span(cur)
      local sub = {buf = cur.buf, pos = start, len = cur.len}
      local t = parse_tensor(sub, start + n, src)
      model.init[#model.init + 1] = t
      model.by_name[t.name] = t
    elseif field == 11 then
      local start, n = read_bytes_span(cur)
      local sub = {buf = cur.buf, pos = start, len = cur.len}
      local name = parse_value_info(sub, start + n)
      if name then
        model.inputs[#model.inputs + 1] = name
      end
    elseif field == 12 then
      local start, n = read_bytes_span(cur)
      local sub = {buf = cur.buf, pos = start, len = cur.len}
      local name = parse_value_info(sub, start + n)
      if name then
        model.outputs[#model.outputs + 1] = name
      end
    else
      skip_field(cur, wire)
    end
  end
end

-- ------------------------------------------------------------- loading

-- ModelProto: ir_version=1, producer_name=2, producer_version=3, graph=7
local function parse_model(buf, len, opts)
  local model = {
    init = {}, by_name = {}, nodes = {},
    inputs = {}, outputs = {},
    graph_name = "",
  }
  local cur = {buf = buf, pos = 0, len = len}
  while cur.pos < len do
    local field, wire = read_tag(cur)
    if field == 1 then
      model.ir_version = read_varint(cur)
    elseif field == 2 then
      model.producer = read_string(cur)
    elseif field == 3 then
      model.producer_version = read_string(cur)
    elseif field == 7 then
      local start, n = read_bytes_span(cur)
      local sub = {buf = buf, pos = start, len = len}
      parse_graph(sub, start + n, buf, opts, model)
    else
      skip_field(cur, wire)
    end
  end
  return model
end

-- loads an .onnx file. opts.nodes requests graph structure as well as
-- weights. returns model, or nil plus an error message.
function M.load(path, opts)
  opts = opts or {}
  local f, ferr = io.open(path, "rb")
  if not f then
    return nil, "cannot open " .. tostring(path) .. ": " .. tostring(ferr)
  end
  local size = f:seek("end")
  f:seek("set", 0)
  local buf = ffi.new("uint8_t[?]", size)
  local chunk_size = 8 * 1024 * 1024
  local offset = 0
  while offset < size do
    local want = size - offset
    if want > chunk_size then
      want = chunk_size
    end
    local chunk = f:read(want)
    if not chunk or #chunk == 0 then
      f:close()
      return nil, "short read at offset " .. tostring(offset)
    end
    ffi.copy(buf + offset, chunk, #chunk)
    offset = offset + #chunk
  end
  f:close()

  local ok, model = pcall(parse_model, buf, size, opts)
  if not ok then
    return nil, tostring(model)
  end
  -- the file buffer itself is not referenced by the tensors (their data
  -- was copied), but keeping it costs nothing and makes reparsing cheap
  -- if a caller ever wants it.
  model.file_size = size
  return model
end

-- total bytes held by the materialised tensors
function M.weight_bytes(model)
  local total = 0
  for i = 1, #model.init do
    local t = model.init[i]
    local size = DTYPE_SIZE[t.dtype]
    if size and t.n then
      total = total + t.n * size
    end
  end
  return total
end

function M.shape_string(dims)
  local parts = {}
  for i = 1, #dims do
    parts[i] = tostring(dims[i])
  end
  return "{" .. table.concat(parts, ", ") .. "}"
end

-- mean and standard deviation of a float tensor, used to sanity check
-- that what was parsed looks like trained weights rather than noise
function M.stats(t)
  if t.kind ~= "float" or not t.data or t.n == 0 then
    return nil
  end
  local sum = 0.0
  for i = 0, t.n - 1 do
    sum = sum + t.data[i]
  end
  local mean = sum / t.n
  local acc = 0.0
  local maxabs = 0.0
  for i = 0, t.n - 1 do
    local d = t.data[i] - mean
    acc = acc + d * d
    local a = t.data[i]
    if a < 0 then a = -a end
    if a > maxabs then maxabs = a end
  end
  return mean, math.sqrt(acc / t.n), maxabs
end

return M
