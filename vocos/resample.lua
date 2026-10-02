-- vocos/resample.lua
-- downmix (stereo -> mono) and resample (native rate -> a target rate) for
-- decoded pcm. band-limited windowed-sinc interpolation, measured in
-- tests/test.lua rather than trusted.

local ffi = require("ffi")

local M = {}

-- downmixes interleaved stereo int16 pcm to mono via simple averaging.
-- interleaved: int16_t* (2*n entries, L,R,L,R,...). returns (int16_t[?], n).
function M.downmix_stereo(interleaved, n)
  local out = ffi.new("int16_t[?]", n)
  for i = 0, n - 1 do
    local avg = (interleaved[2 * i] + interleaved[2 * i + 1]) / 2
    out[i] = math.floor(avg + 0.5)
  end
  return out, n
end

-- resamples mono pcm from src_rate to dst_rate. samples: int16_t*
-- (n entries starting at offset). returns (int16_t[?], out_n).
--
-- band-limited interpolation: every output sample is a windowed-sinc sum
-- over the input around its exact position, with the cutoff just below
-- the lower of the two nyquist frequencies. this replaced a version that
-- low-passed and then interpolated linearly, which measured only 31 dB
-- clean at 3 kHz and 22 dB at 5 kHz (22050 -> 16000): enough for speech
-- recognition, audibly metallic and hissy for anything meant to be heard.
-- tests/test.lua holds it to better than 70 dB.
local KAISER_BETA = 8.6    -- about 85 dB of stopband
local ZEROS = 20           -- sinc zero crossings on each side
local PHASES = 512         -- kernel table resolution per input sample

local function bessel_i0(x)
  local sum, term, k = 1, 1, 1
  repeat
    term = term * (x / (2 * k)) ^ 2
    sum = sum + term
    k = k + 1
  until term < 1e-12 * sum
  return sum
end

local kernel_cache = {}

-- table of the kernel h(u) for u = 0, 1/PHASES, ... half (input samples)
local function kernel(c)
  local key = string.format("%.9f", c)
  if kernel_cache[key] then return kernel_cache[key] end
  local half = math.ceil(ZEROS / (2 * c))
  local len = half * PHASES + 2
  local t = ffi.new("double[?]", len)
  local i0b = bessel_i0(KAISER_BETA)
  for i = 0, len - 1 do
    local u = i / PHASES
    local r = u / half
    if r > 1 then
      t[i] = 0
    else
      local x = 2 * c * u
      local sinc = (x == 0) and 1 or math.sin(math.pi * x) / (math.pi * x)
      local w = bessel_i0(KAISER_BETA * math.sqrt(1 - r * r)) / i0b
      t[i] = 2 * c * sinc * w
    end
  end
  local k = {t = t, half = half}
  kernel_cache[key] = k
  return k
end

function M.resample_mono(samples, offset, n, src_rate, dst_rate)
  if src_rate == dst_rate then
    local out = ffi.new("int16_t[?]", n)
    ffi.copy(out, samples + offset, n * 2)
    return out, n
  end
  -- cutoff in cycles per input sample: 95% of the lower nyquist
  local c = 0.5 * math.min(src_rate, dst_rate) * 0.95 / src_rate
  local K = kernel(c)
  local t, half = K.t, K.half
  local ratio = src_rate / dst_rate
  local out_n = math.floor((n - 1) / ratio) + 1
  local out = ffi.new("int16_t[?]", out_n)
  for j = 0, out_n - 1 do
    local pos = j * ratio
    local k0 = math.floor(pos)
    local lo = math.max(0, k0 - half + 1)
    local hi = math.min(n - 1, k0 + half)
    local acc = 0.0
    for k = lo, hi do
      local u = math.abs(pos - k) * PHASES
      local i = math.floor(u)
      local fr = u - i
      acc = acc + samples[offset + k] * (t[i] + (t[i + 1] - t[i]) * fr)
    end
    if acc > 32767 then acc = 32767 elseif acc < -32768 then acc = -32768 end
    out[j] = (acc >= 0) and math.floor(acc + 0.5) or math.ceil(acc - 0.5)
  end
  return out, out_n
end

-- convenience: int16 pcm (possibly stereo-interleaved, at its native rate)
-- -> mono pcm at dst_rate (default 22050, what Vocos takes)
function M.to_mono_rate(samples, n_per_channel, rate, channels, dst_rate)
  dst_rate = dst_rate or 22050
  local mono, mono_n
  if channels == 2 then
    mono, mono_n = M.downmix_stereo(samples, n_per_channel)
  else
    mono, mono_n = samples, n_per_channel
  end
  return M.resample_mono(mono, 0, mono_n, rate, dst_rate)
end

return M
