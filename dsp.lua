-- vocos/dsp.lua -- the signal side of Vocos: mel analysis, STFT, inverse STFT.
--
-- settings fixed by the model (vocos-mel-22khz): 22050 Hz, n_fft 1024,
-- hop 256, periodic hann window, centred frames with reflect padding,
-- 80 slaney mel bands from 0 to 8000 Hz, natural log of the magnitude mel
-- spectrum floored at 1e-5. these match what torchaudio / librosa produce
-- for the same settings, so a mel made here can be fed to the original
-- model and back.
--
-- layouts: a mel is float[80 * F], band-major (band b, frame f at b * F + f);
-- a spectrum is double[513 * F], bin-major (bin k, frame f at k * F + f).

local ffi = require("ffi")

local M = {}
local SR, NFFT, HOP, NMELS, BINS = 22050, 1024, 256, 80, 513
M.SAMPLE_RATE, M.NFFT, M.HOP, M.N_MELS, M.BINS = SR, NFFT, HOP, NMELS, BINS

-- radix-2 complex fft, in place, on double arrays (the inverse is unscaled)
local rev, tw = ffi.new("int32_t[?]", NFFT), ffi.new("double[?]", NFFT)
do
  for i = 0, NFFT - 1 do
    local r, x = 0, i
    for _ = 1, 10 do r = r * 2 + x % 2; x = math.floor(x / 2) end
    rev[i] = r
  end
  for i = 0, NFFT / 2 - 1 do
    tw[2 * i] = math.cos(2 * math.pi * i / NFFT)
    tw[2 * i + 1] = -math.sin(2 * math.pi * i / NFFT)
  end
end
local function fft(re, im, inverse)
  for i = 0, NFFT - 1 do
    local j = rev[i]
    if j > i then re[i], re[j] = re[j], re[i]; im[i], im[j] = im[j], im[i] end
  end
  local sign = inverse and -1 or 1
  local size = 2
  while size <= NFFT do
    local half, step = size / 2, NFFT / size
    for start = 0, NFFT - 1, size do
      for k = 0, half - 1 do
        local wr, wi = tw[2 * k * step], tw[2 * k * step + 1] * sign
        local a, b = start + k, start + k + half
        local xr = re[b] * wr - im[b] * wi
        local xi = re[b] * wi + im[b] * wr
        re[b] = re[a] - xr; im[b] = im[a] - xi
        re[a] = re[a] + xr; im[a] = im[a] + xi
      end
    end
    size = size * 2
  end
end
M.fft = fft

-- periodic hann, as torch and librosa use for the stft
local win = ffi.new("double[?]", NFFT)
for i = 0, NFFT - 1 do win[i] = 0.5 - 0.5 * math.cos(2 * math.pi * i / NFFT) end

-- slaney mel filterbank (librosa's default): float[80 * 513], band-major
local function hz_to_mel(f)
  local f_sp, min_log_hz = 200 / 3, 1000
  if f < min_log_hz then return f / f_sp end
  return min_log_hz / f_sp + math.log(f / min_log_hz) / (math.log(6.4) / 27)
end
local function mel_to_hz(m)
  local f_sp, min_log_hz = 200 / 3, 1000
  local min_log_mel = min_log_hz / f_sp
  if m < min_log_mel then return m * f_sp end
  return min_log_hz * math.exp((math.log(6.4) / 27) * (m - min_log_mel))
end
local fb = ffi.new("float[?]", NMELS * BINS)
do
  local lo, hi = hz_to_mel(0), hz_to_mel(8000)
  local pts = {}
  for i = 0, NMELS + 1 do pts[i] = mel_to_hz(lo + (hi - lo) * i / (NMELS + 1)) end
  for m = 0, NMELS - 1 do
    local f0, f1, f2 = pts[m], pts[m + 1], pts[m + 2]
    local norm = 2 / (f2 - f0)
    for k = 0, BINS - 1 do
      local f = k * SR / NFFT
      fb[m * BINS + k] = math.max(0, math.min((f - f0) / (f1 - f0), (f2 - f) / (f2 - f1))) * norm
    end
  end
end
function M.filterbank() return fb end

-- the fade above `cutoff` Hz applied to the magnitude before the inverse
-- transform (the mel stops at 8 kHz, so the network invents the top
-- octave; heard as a faint rattle). half at the cutoff, 480 Hz wide.
-- false: no fade.
function M.taper(cutoff)
  local t = ffi.new("double[?]", BINS)
  for k = 0, BINS - 1 do
    local f = k * SR / NFFT
    if not cutoff then t[k] = 1
    else
      local x = math.max(0, math.min(1, (cutoff + 240 - f) / 480))
      t[k] = 0.5 - 0.5 * math.cos(math.pi * x)
    end
  end
  return t
end

local function frame(x, n, f, re, im)
  local s0 = f * HOP - NFFT / 2
  for i = 0, NFFT - 1 do
    local j = s0 + i
    if j < 0 then j = -j end                       -- reflect padding
    if j >= n then j = 2 * (n - 1) - j end
    re[i] = x[j] * win[i]; im[i] = 0
  end
  fft(re, im, false)
end

-- samples (float or double array, n of them, 22050 Hz, range -1..1) ->
-- log mel float[80 * F], F
function M.mel(x, n)
  local F = math.floor(n / HOP) + 1
  local out = ffi.new("float[?]", NMELS * F)
  local re, im = ffi.new("double[?]", NFFT), ffi.new("double[?]", NFFT)
  local mag = ffi.new("double[?]", BINS)
  for f = 0, F - 1 do
    frame(x, n, f, re, im)
    for k = 0, BINS - 1 do mag[k] = math.sqrt(re[k] * re[k] + im[k] * im[k]) end
    for m = 0, NMELS - 1 do
      local s, row = 0, m * BINS
      for k = 0, BINS - 1 do s = s + fb[row + k] * mag[k] end
      out[m * F + f] = math.log(math.max(s, 1e-5))
    end
  end
  return out, F
end

-- the short-time spectrum: magnitude, cos and sin of the phase, each
-- double[513 * F], and F
function M.stft(x, n)
  local F = math.floor(n / HOP) + 1
  local re, im = ffi.new("double[?]", NFFT), ffi.new("double[?]", NFFT)
  local mag = ffi.new("double[?]", BINS * F)
  local cx, sy = ffi.new("double[?]", BINS * F), ffi.new("double[?]", BINS * F)
  for f = 0, F - 1 do
    frame(x, n, f, re, im)
    for k = 0, BINS - 1 do
      local a = math.sqrt(re[k] * re[k] + im[k] * im[k])
      mag[k * F + f] = a
      if a > 0 then cx[k * F + f], sy[k * F + f] = re[k] / a, im[k] / a
      else cx[k * F + f], sy[k * F + f] = 1, 0 end
    end
  end
  return mag, cx, sy, F
end

-- the inverse: magnitude and phase (cos, sin) [513 * F] -> float[count],
-- count = 256 * (F - 1). taper: M.taper(...) or nil (none)
function M.istft(mag, cx, sy, F, taper)
  local re, im = ffi.new("double[?]", NFFT), ffi.new("double[?]", NFFT)
  taper = taper or M.taper(false)
  local total = NFFT + HOP * (F - 1)
  local acc = ffi.new("double[?]", total)
  local wsum = ffi.new("double[?]", total)
  for f = 0, F - 1 do
    for k = 0, BINS - 1 do
      local a = mag[k * F + f] * taper[k]
      re[k] = a * cx[k * F + f]; im[k] = a * sy[k * F + f]
    end
    for k = 1, BINS - 2 do re[NFFT - k] = re[k]; im[NFFT - k] = -im[k] end
    im[0], im[BINS - 1] = 0, 0
    fft(re, im, true)
    local s0 = f * HOP
    for i = 0, NFFT - 1 do
      acc[s0 + i] = acc[s0 + i] + re[i] / NFFT * win[i]
      wsum[s0 + i] = wsum[s0 + i] + win[i] * win[i]
    end
  end
  local count = HOP * (F - 1)
  local out = ffi.new("float[?]", count)
  local pad = NFFT / 2
  for i = 0, count - 1 do
    local w = wsum[i + pad]
    out[i] = w > 1e-11 and acc[i + pad] / w or acc[i + pad]
  end
  return out, count
end

return M
