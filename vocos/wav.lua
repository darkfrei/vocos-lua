-- vocos/wav.lua -- 16-bit PCM WAV files, read and written.

local ffi = require("ffi")
local M = {}

local function u32(s, i) return s:byte(i) + 256 * s:byte(i + 1) + 65536 * s:byte(i + 2) + 16777216 * s:byte(i + 3) end
local function u16(s, i) return s:byte(i) + 256 * s:byte(i + 1) end

-- path -> int16 pcm (interleaved if stereo), frames per channel, rate, channels
function M.read(path)
  local f = assert(io.open(path, "rb"))
  local s = f:read("*a")
  f:close()
  assert(s:sub(1, 4) == "RIFF" and s:sub(9, 12) == "WAVE", path .. ": not a WAV file")
  local pos, rate, channels, bits = 13, nil, nil, nil
  while pos + 8 <= #s do
    local id, size = s:sub(pos, pos + 3), u32(s, pos + 4)
    if id == "fmt " then
      assert(u16(s, pos + 8) == 1, path .. ": only uncompressed PCM is read")
      channels, rate, bits = u16(s, pos + 10), u32(s, pos + 12), u16(s, pos + 22)
      assert(bits == 16, path .. ": only 16-bit samples are read")
    elseif id == "data" then
      assert(rate, path .. ": data before fmt")
      size = math.min(size, #s - pos - 7)
      local n = math.floor(size / 2)
      local pcm = ffi.new("int16_t[?]", n)
      ffi.copy(pcm, s:sub(pos + 8, pos + 7 + n * 2), n * 2)
      return pcm, math.floor(n / channels), rate, channels
    end
    pos = pos + 8 + size + size % 2
  end
  error(path .. ": no data")
end

-- path -> mono float samples at `rate` (default 22050), count
function M.read_mono(path, rate)
  local R = require("vocos.resample")
  local pcm, n, sr, ch = M.read(path)
  local mono, m = R.to_mono_rate(pcm, n, sr, ch, rate or 22050)
  local x = ffi.new("float[?]", m)
  for i = 0, m - 1 do x[i] = mono[i] / 32768 end
  return x, m
end

-- float samples (-1..1) -> 16-bit mono WAV. peak: if given, scaled so that
-- the loudest sample is at that level (0.9: a little below full scale)
function M.write(path, x, n, rate, peak)
  rate = rate or 22050
  local g = 1
  if peak then
    local mx = 1e-9
    for i = 0, n - 1 do mx = math.max(mx, math.abs(x[i])) end
    g = peak / mx
  end
  local q = ffi.new("int16_t[?]", n)
  for i = 0, n - 1 do
    local v = math.floor(x[i] * g * 32767 + 0.5)
    q[i] = v > 32767 and 32767 or (v < -32768 and -32768 or v)
  end
  local function le32(v) return string.char(v % 256, math.floor(v / 256) % 256, math.floor(v / 65536) % 256, math.floor(v / 16777216) % 256) end
  local function le16(v) return string.char(v % 256, math.floor(v / 256) % 256) end
  local f = assert(io.open(path, "wb"))
  f:write("RIFF", le32(36 + n * 2), "WAVEfmt ", le32(16), le16(1), le16(1), le32(rate), le32(rate * 2), le16(2),
    le16(16), "data", le32(n * 2), ffi.string(q, n * 2))
  f:close()
end

return M
