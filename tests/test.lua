-- tests/test.lua -- run from the repository root:  luajit tests/test.lua
-- with VOCOS=path/to/vocos-22khz-univ.onnx also checks the real network
-- against numbers produced by ONNX Runtime from the same mel.
package.path = "./?.lua;./?/init.lua;" .. package.path
local ffi = require("ffi")
local vocos = require("vocos")
local dsp, net, wav = vocos.dsp, vocos.net, vocos.wav

local fails = 0
local function check(ok, what)
  print((ok and "ok    " or "FAIL  ") .. what)
  if not ok then fails = fails + 1 end
end

-- a gliding harmonic tone, 1 s at 22050 Hz
local function tone(n, rate)
  local x = ffi.new("float[?]", n)
  local ph = 0
  for i = 0, n - 1 do
    local f0 = 120 + 80 * i / n
    ph = ph + 2 * math.pi * f0 / rate
    local s = 0
    for h = 1, 20 do if h * f0 < 7000 then s = s + math.sin(h * ph) / h end end
    x[i] = 0.3 * s
  end
  return x
end

-- 1. stft and istft give the signal back
do
  local n = 22050
  local x = tone(n, 22050)
  local mag, cx, sy, F = dsp.stft(x, n)
  local y, count = dsp.istft(mag, cx, sy, F)
  local e = 0
  for i = 0, math.min(n, count) - 1 do e = math.max(e, math.abs(x[i] - y[i])) end
  check(e < 1e-4, string.format("stft -> istft gives the signal back (max error %.1e)", e))
end

-- 2. the mel of a 1 kHz sine peaks in the band around 1 kHz
do
  local n = 22050
  local x = ffi.new("float[?]", n)
  for i = 0, n - 1 do x[i] = 0.5 * math.sin(2 * math.pi * 1000 * i / 22050) end
  local mel, F = dsp.mel(x, n)
  local best, bb = -math.huge, -1
  for b = 0, 79 do if mel[b * F + 10] > best then best, bb = mel[b * F + 10], b end end
  check(F == 87 and bb >= 25 and bb <= 27, string.format("mel: 87 frames, a 1 kHz sine peaks in band %d", bb))
end

-- 3. the resampler: a 3 kHz tone 44100 -> 22050 stays clean
do
  local R = require("vocos.resample")
  local n = 44100
  local pcm = ffi.new("int16_t[?]", n)
  for i = 0, n - 1 do pcm[i] = math.floor(16000 * math.sin(2 * math.pi * 3000 * i / 44100) + 0.5) end
  local out, m = R.resample_mono(pcm, 0, n, 44100, 22050)
  -- fit the ideal tone (least squares on sin and cos) and measure what is left
  local ss, sc, cc, ys, yc = 0, 0, 0, 0, 0
  for i = 200, m - 201 do
    local a = 2 * math.pi * 3000 * i / 22050
    local s, c = math.sin(a), math.cos(a)
    ss, sc, cc, ys, yc = ss + s * s, sc + s * c, cc + c * c, ys + out[i] * s, yc + out[i] * c
  end
  local det = ss * cc - sc * sc
  local A, B = (ys * cc - yc * sc) / det, (yc * ss - ys * sc) / det
  local sig, err = 0, 0
  for i = 200, m - 201 do
    local a = 2 * math.pi * 3000 * i / 22050
    local fit = A * math.sin(a) + B * math.cos(a)
    sig, err = sig + fit * fit, err + (out[i] - fit) ^ 2
  end
  local snr = 10 * math.log10(sig / err)
  check(m == 22050 and snr > 60, string.format("resampler 44100 -> 22050: %.0f dB clean", snr))
end

-- 4. wav: written and read back
do
  local path = os.tmpname()
  local x = tone(1000, 22050)
  wav.write(path, x, 1000, 22050)
  local pcm, n, rate, ch = wav.read(path)
  local e = 0
  for i = 0, n - 1 do e = math.max(e, math.abs(pcm[i] / 32767 - x[i])) end
  os.remove(path)
  check(n == 1000 and rate == 22050 and ch == 1 and e < 1e-4, "wav: written and read back")
end

-- 5. a small random network of Vocos' shape: files, pruning, zero weights
do
  local seed = 7
  local function rnd() seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 * 2 - 1 end
  local H = {8, 8, 8}
  local v = net.new(H)
  for _, e in ipairs(v:layout()) do
    local t = ffi.new("float[?]", e[2])
    local one = e[1]:match("_g%d*$") or e[1]:match("^norm_g$") or e[1]:match("^fin_g$")
    for i = 0, e[2] - 1 do t[i] = one and 1 or rnd() * 0.05 end
    v.w[e[1]] = t
  end
  local n = 4096
  local mel, F = dsp.mel(tone(n, 22050), n)
  local m1 = v:forward(mel, F)
  local p = os.tmpname()
  v:save(p)
  local m2 = vocos.load(p):forward(mel, F)
  v:save(p, true)
  local m3 = vocos.load(p):forward(mel, F)
  os.remove(p)
  local d2, d3 = 0, 0
  for i = 0, 513 * F - 1 do
    d2 = math.max(d2, math.abs(m1[i] - m2[i]))
    d3 = math.max(d3, math.abs(math.log(m1[i]) - math.log(m3[i])))
  end
  check(d2 == 0, "file (float32): the same network back")
  check(d3 < 0.05, string.format("file (float16): nearly the same (log magnitude within %.3f)", d3))
  local imp = v:activity({{mel, F}})
  local same = v:prune(imp, 8)
  local m4 = same:forward(mel, F)
  local d4 = 0
  for i = 0, 513 * F - 1 do d4 = math.max(d4, math.abs(m1[i] - m4[i])) end
  check(d4 < 1e-9, "prune keeping every neuron changes nothing")
  -- a neuron with zero weights out goes without a trace
  local w2 = v.w.pw2_w1
  for c = 0, 511 do w2[c * 8 + 3] = 0 end
  local m5 = v:forward(mel, F)
  imp = v:activity({{mel, F}})
  check(imp[2][4] == 0, "a neuron writing nothing back has importance 0")
  local small = v:prune(imp, {8, 7, 8})
  local m6 = small:forward(mel, F)
  local d6 = 0
  for i = 0, 513 * F - 1 do d6 = math.max(d6, math.abs(m5[i] - m6[i]) / m5[i]) end
  check(small.H[2] == 7 and d6 < 1e-5, "and taking it out changes nothing")
end

-- 6. the real network against ONNX Runtime (needs the model file)
local model = os.getenv("VOCOS")
if model then
  local ref = dofile("tests/onnxruntime_reference.lua")
  local n = 22050
  local mel, F = dsp.mel(tone(n, 22050), n)
  local v = vocos.load(model)
  local mag, cx, sy = v:forward(mel, F)
  local dm, dp = 0, 0
  for _, p in ipairs(ref.points) do
    local i = p[1] * F + p[2]
    dm = math.max(dm, math.abs(math.log(mag[i]) - p[3]))
    if p[3] > -3 then dp = math.max(dp, math.abs(math.atan2(sy[i] * p[4] - cx[i] * p[5], cx[i] * p[4] + sy[i] * p[5]))) end
  end
  check(F == ref.F and dm < 1e-3 and dp < 1e-2,
    string.format("the network matches ONNX Runtime (log magnitude within %.1e, phase within %.1e rad)", dm, dp))
else
  print("skip  the network against ONNX Runtime (set VOCOS=path/to/vocos-22khz-univ.onnx)")
end

print(fails == 0 and "all tests passed" or (fails .. " failed"))
os.exit(fails == 0 and 0 or 1)
