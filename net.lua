-- vocos/net.lua -- the Vocos network itself, inference only, in plain luajit.
--
-- vocos-mel-22khz (Siuzdak 2023; this checkpoint by BSC-LT): mel (80) ->
--   convolution 80 -> 512 (kernel 7) -> layer norm
--   -> 8 ConvNeXt blocks, each:
--        depthwise convolution (kernel 7, every channel on its own)
--        -> layer norm -> 512 -> 1536 -> GELU -> 1536 -> 512
--        -> times gamma (512 numbers) -> added to what came in
--   -> layer norm -> linear 512 -> 1026
--   = 513 log magnitudes (capped at log 100) and 513 phase angles per frame
-- the 1536 hidden numbers of a block are its "neurons"; the blocks may have
-- different numbers of them (see prune below).
--
-- weights come from the ONNX file as published by k2-fsa/sherpa-onnx
-- (vocos-22khz-univ.onnx, read by vocos/onnx.lua, no ONNX Runtime), or from
-- the compact file this module writes (save / load: float32 or float16).
--
-- the pointwise layers skip zero weights, so a network with weights set to
-- zero gets faster in proportion.

local ffi = require("ffi")

local M = {}
local Net = {}
Net.__index = Net
local NB, BINS, D = 80, 513, 512
M.MAGIC = "VOCOSLUA1"

local function floats(n) return ffi.new("float[?]", n) end

-- erf, Abramowitz and Stegun 7.1.26 (error below 1.5e-7)
local function erf(x)
  local s = x < 0 and -1 or 1
  x = math.abs(x)
  local t = 1 / (1 + 0.3275911 * x)
  local y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * math.exp(-x * x)
  return s * y
end

local function new_net(H)
  return setmetatable({L = #H, H = H, w = {}}, Net)
end
-- an empty network with H[b] neurons in block b (fill net.w, see layout)
M.new = new_net

-- the network from vocos-22khz-univ.onnx
function M.from_onnx(path)
  local R = require("vocos.onnx")
  local m = assert(R.load(path, {}))
  local W = m.by_name
  local function get(n) return assert(W[n], "vocos: no tensor " .. n .. " in " .. path).data end
  local function copy(src, n) local t = floats(n); for i = 0, n - 1 do t[i] = src[i] end; return t end
  -- src rows x cols (row-major) -> cols x rows
  local function transpose(src, rows, cols)
    local t = floats(rows * cols)
    for r = 0, rows - 1 do for c = 0, cols - 1 do t[c * rows + r] = src[r * cols + c] end end
    return t
  end
  local net = new_net({1536, 1536, 1536, 1536, 1536, 1536, 1536, 1536})
  local w = net.w
  w.embed_w = copy(get("vocos.backbone.embed.weight"), D * NB * 7)
  w.embed_b = copy(get("vocos.backbone.embed.bias"), D)
  w.norm_g = copy(get("vocos.backbone.norm.weight"), D)
  w.norm_b = copy(get("vocos.backbone.norm.bias"), D)
  for b = 0, 7 do
    local p = "vocos.backbone.convnext." .. b .. "."
    w["dw_w" .. b] = copy(get(p .. "dwconv.weight"), D * 7)
    w["dw_b" .. b] = copy(get(p .. "dwconv.bias"), D)
    w["ln_g" .. b] = copy(get(p .. "norm.weight"), D)
    w["ln_b" .. b] = copy(get(p .. "norm.bias"), D)
    w["pw1_w" .. b] = transpose(get("onnx::MatMul_" .. (705 + 2 * b)), D, 1536)    -- {1536, 512}
    w["pw1_b" .. b] = copy(get(p .. "pwconv1.bias"), 1536)
    w["pw2_w" .. b] = transpose(get("onnx::MatMul_" .. (706 + 2 * b)), 1536, D)    -- {512, 1536}
    w["pw2_b" .. b] = copy(get(p .. "pwconv2.bias"), D)
    w["gamma" .. b] = copy(get(p .. "gamma"), D)
  end
  w.fin_g = copy(get("vocos.backbone.final_layer_norm.weight"), D)
  w.fin_b = copy(get("vocos.backbone.final_layer_norm.bias"), D)
  w.head_w = transpose(get("onnx::MatMul_721"), D, 2 * BINS)                       -- {1026, 512}
  w.head_b = copy(get("vocos.head.out.bias"), 2 * BINS)
  return net
end

-- the tensors in file order, with their sizes
function Net:layout()
  local list = {{"embed_w", D * NB * 7}, {"embed_b", D}, {"norm_g", D}, {"norm_b", D}}
  for b = 0, self.L - 1 do
    local H = self.H[b + 1]
    for _, e in ipairs({{"dw_w", D * 7}, {"dw_b", D}, {"ln_g", D}, {"ln_b", D}, {"pw1_w", H * D},
        {"pw1_b", H}, {"pw2_w", D * H}, {"pw2_b", D}, {"gamma", D}}) do
      list[#list + 1] = {e[1] .. b, e[2]}
    end
  end
  for _, e in ipairs({{"fin_g", D}, {"fin_b", D}, {"head_w", 2 * BINS * D}, {"head_b", 2 * BINS}}) do
    list[#list + 1] = e
  end
  return list
end

function Net:count()
  local n = 0
  for _, e in ipairs(self:layout()) do n = n + e[2] end
  return n
end

-- ---------------------------------------------------------------- forward

-- out {C_out, F} = w {C_out, C_in} x {C_in, F} + b; zero weights skipped
local function pointwise(x, Cin, F, w, b, Cout)
  local out = floats(Cout * F)
  local acc = ffi.new("double[?]", F)
  for o = 0, Cout - 1 do
    local bo = b[o]
    for f = 0, F - 1 do acc[f] = bo end
    local wr = o * Cin
    for i = 0, Cin - 1 do
      local wv = w[wr + i]
      if wv ~= 0 then
        local xr = i * F
        for f = 0, F - 1 do acc[f] = acc[f] + wv * x[xr + f] end
      end
    end
    local orow = o * F
    for f = 0, F - 1 do out[orow + f] = acc[f] end
  end
  return out
end

-- layer norm over the channels of every frame
local function layernorm(x, C, F, g, b)
  local out = floats(C * F)
  for f = 0, F - 1 do
    local s1, s2 = 0, 0
    for c = 0, C - 1 do local v = x[c * F + f]; s1 = s1 + v; s2 = s2 + v * v end
    local mu = s1 / C
    local var = s2 / C - mu * mu
    local r = 1 / math.sqrt((var > 0 and var or 0) + 1e-6)
    for c = 0, C - 1 do out[c * F + f] = (x[c * F + f] - mu) * r * g[c] + b[c] end
  end
  return out
end

-- x: log mel float[80 * F] (band-major). returns the magnitude, cos and sin
-- of the phase, each double[513 * F] (bin-major); with capture, also the
-- GELU outputs of every block (float[H * F] each)
function Net:forward(mel, F, capture)
  local w = self.w
  -- embed: convolution 80 -> 512, kernel 7, padding 3
  local h = floats(D * F)
  do
    local acc = ffi.new("double[?]", F)
    local ew, eb = w.embed_w, w.embed_b
    for o = 0, D - 1 do
      for f = 0, F - 1 do acc[f] = eb[o] end
      for i = 0, NB - 1 do
        local xr, wr = i * F, (o * NB + i) * 7
        for k = 0, 6 do
          local wv = ew[wr + k]
          local off = k - 3
          local lo, hi = math.max(0, -off), math.min(F - 1, F - 1 - off)
          for f = lo, hi do acc[f] = acc[f] + wv * mel[xr + f + off] end
        end
      end
      for f = 0, F - 1 do h[o * F + f] = acc[f] end
    end
  end
  h = layernorm(h, D, F, w.norm_g, w.norm_b)
  local acts = capture and {} or nil
  local y = floats(D * F)
  for b = 0, self.L - 1 do
    local H = self.H[b + 1]
    -- depthwise convolution, kernel 7, padding 3
    local dw, db = w["dw_w" .. b], w["dw_b" .. b]
    for c = 0, D - 1 do
      local row, wr = c * F, c * 7
      for f = 0, F - 1 do
        local a = db[c]
        for k = 0, 6 do
          local g = f + k - 3
          if g >= 0 and g < F then a = a + dw[wr + k] * h[row + g] end
        end
        y[row + f] = a
      end
    end
    local z = layernorm(y, D, F, w["ln_g" .. b], w["ln_b" .. b])
    local u = pointwise(z, D, F, w["pw1_w" .. b], w["pw1_b" .. b], H)
    for i = 0, H * F - 1 do local x = u[i]; u[i] = 0.5 * x * (1 + erf(x * 0.70710678118654752)) end
    if acts then acts[b + 1] = u end
    local v = pointwise(u, H, F, w["pw2_w" .. b], w["pw2_b" .. b], D)
    local gm = w["gamma" .. b]
    for c = 0, D - 1 do
      local g, row = gm[c], c * F
      for f = 0, F - 1 do h[row + f] = h[row + f] + g * v[row + f] end
    end
  end
  h = layernorm(h, D, F, w.fin_g, w.fin_b)
  local o = pointwise(h, D, F, w.head_w, w.head_b, 2 * BINS)
  local mag = ffi.new("double[?]", BINS * F)
  local cx, sy = ffi.new("double[?]", BINS * F), ffi.new("double[?]", BINS * F)
  local cap = math.log(100)
  for i = 0, BINS * F - 1 do
    mag[i] = math.exp(math.min(o[i], cap))
    local p = o[BINS * F + i]
    cx[i], sy[i] = math.cos(p), math.sin(p)
  end
  return mag, cx, sy, acts
end

-- speech from a log mel: float[256 * (F - 1)], count. taper: the fade above
-- 7.8 kHz (vocos/dsp.lua taper; false: none)
function Net:synth(mel, F, taper)
  local dsp = require("vocos.dsp")
  if taper == nil then self.taper = self.taper or dsp.taper(7800); taper = self.taper
  elseif taper == false then taper = nil end
  local mag, cx, sy = self:forward(mel, F)
  return dsp.istft(mag, cx, sy, F, taper)
end

-- ---------------------------------------------------- smaller for one voice

-- how much every neuron of every block gives on some mels ({mel, F} pairs):
-- its mean |GELU output| times the length of what it writes back (its
-- column of the second matrix times gamma). returns imp[block][neuron]
function Net:activity(mels)
  local sum, frames = {}, 0
  for b = 1, self.L do sum[b] = ffi.new("double[?]", self.H[b]) end
  for _, m in ipairs(mels) do
    local mel, F = m[1], m[2]
    local _, _, _, acts = self:forward(mel, F, true)
    for b = 1, self.L do
      local a, s = acts[b], sum[b]
      for j = 0, self.H[b] - 1 do
        local t = 0
        for f = 0, F - 1 do local v = a[j * F + f]; t = t + (v < 0 and -v or v) end
        s[j] = s[j] + t
      end
    end
    frames = frames + F
  end
  local imp = {}
  for b = 1, self.L do
    local H, g, w2 = self.H[b], self.w["gamma" .. (b - 1)], self.w["pw2_w" .. (b - 1)]
    imp[b] = {}
    for j = 0, H - 1 do
      local n2 = 0
      for c = 0, D - 1 do local v = w2[c * H + j] * g[c]; n2 = n2 + v * v end
      imp[b][j + 1] = sum[b][j] / frames * math.sqrt(n2)
    end
  end
  return imp
end

-- a copy keeping the `keep` most important neurons of every block (imp from
-- activity), or keep[b] of block b when keep is a table
function Net:prune(imp, keep)
  local out = new_net({})
  for name, t in pairs(self.w) do out.w[name] = t end
  for b = 1, self.L do
    local H = self.H[b]
    local k = type(keep) == "table" and keep[b] or keep
    k = math.min(k, H)
    local idx = {}
    for j = 1, H do idx[j] = j end
    table.sort(idx, function(p, q) return imp[b][p] > imp[b][q] end)
    local K = {}
    for i = 1, k do K[i] = idx[i] - 1 end
    table.sort(K)
    local w1, b1, w2 = self.w["pw1_w" .. (b - 1)], self.w["pw1_b" .. (b - 1)], self.w["pw2_w" .. (b - 1)]
    local n1, nb1, n2 = floats(k * D), floats(k), floats(D * k)
    for i, j in ipairs(K) do
      ffi.copy(n1 + (i - 1) * D, w1 + j * D, D * 4)
      nb1[i - 1] = b1[j]
      for c = 0, D - 1 do n2[c * k + i - 1] = w2[c * H + j] end
    end
    out.w["pw1_w" .. (b - 1)], out.w["pw1_b" .. (b - 1)], out.w["pw2_w" .. (b - 1)] = n1, nb1, n2
    out.H[b] = k
  end
  out.L = #out.H
  return out
end

-- ------------------------------------------------------------------ files

local conv = ffi.new("union { float f; uint32_t u; }")
local function f32_to_f16(x)
  local u = conv
  u.f = x
  local b = u.u
  local sign = bit.band(bit.rshift(b, 16), 0x8000)
  local e = bit.band(bit.rshift(b, 23), 0xff) - 127 + 15
  local m = bit.band(b, 0x7fffff)
  if e <= 0 then
    if e < -10 then return sign end
    m = bit.bor(m, 0x800000)
    local shift = 14 - e
    local h = bit.rshift(m, shift)
    if bit.band(bit.rshift(m, shift - 1), 1) == 1 then h = h + 1 end
    return bit.bor(sign, h)
  elseif e >= 31 then
    return bit.bor(sign, 0x7c00)
  end
  local h = bit.bor(sign, bit.lshift(e, 10), bit.rshift(m, 13))
  if bit.band(m, 0x1000) ~= 0 then h = h + 1 end   -- round half up
  return h
end

local function f16_to_f32(h)
  local sign = bit.band(h, 0x8000) ~= 0 and -1 or 1
  local e = bit.band(bit.rshift(h, 10), 0x1f)
  local m = bit.band(h, 0x3ff)
  if e == 0 then return sign * m * 2 ^ -24 end
  if e == 31 then return m == 0 and sign * math.huge or 0 / 0 end
  return sign * (1 + m / 1024) * 2 ^ (e - 15)
end

-- writes the network; half = true: 2 bytes a number instead of 4
function Net:save(path, half)
  local f = assert(io.open(path, "wb"))
  f:write(M.MAGIC, "\n", half and "f16" or "f32", "\n", table.concat(self.H, " "), "\n")
  for _, e in ipairs(self:layout()) do
    local t, n = self.w[e[1]], e[2]
    if half then
      local q = ffi.new("uint16_t[?]", n)
      for i = 0, n - 1 do q[i] = f32_to_f16(t[i]) end
      f:write(ffi.string(q, n * 2))
    else
      f:write(ffi.string(t, n * 4))
    end
  end
  f:close()
end

function M.load_file(path)
  local f = assert(io.open(path, "rb"))
  assert(f:read("*l") == M.MAGIC, path .. ": not a vocos.lua file")
  local kind = f:read("*l")
  local H = {}
  for x in f:read("*l"):gmatch("%d+") do H[#H + 1] = tonumber(x) end
  local net = new_net(H)
  for _, e in ipairs(net:layout()) do
    local n = e[2]
    local t = floats(n)
    if kind == "f16" then
      local raw = f:read(n * 2)
      local q = ffi.cast("const uint16_t*", raw)
      for i = 0, n - 1 do t[i] = f16_to_f32(q[i]) end
    else
      ffi.copy(t, f:read(n * 4), n * 4)
    end
    net.w[e[1]] = t
  end
  f:close()
  return net
end

return M
