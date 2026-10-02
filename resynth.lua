-- examples/resynth.lua -- a recording through Vocos and back: wav -> mel -> wav.
--   luajit examples/resynth.lua MODEL in.wav out.wav
-- MODEL: vocos-22khz-univ.onnx or a .vocos file (examples/convert.lua).
-- prints how far the mel of the result is from the mel it was made from
-- (mean |log| difference, 80 bands; Vocos on clean speech: about 0.19).
package.path = "./?.lua;./?/init.lua;" .. package.path
local vocos = require("vocos")
local model, inp, outp = arg[1], arg[2], arg[3]
assert(model and inp and outp, "usage: luajit examples/resynth.lua MODEL in.wav out.wav")
local v = vocos.load(model)
local x, n = vocos.wav.read_mono(inp)
local mel, F = vocos.mel(x, n)
local t0 = os.clock()
local y, count = v:synth(mel, F)
local secs = os.clock() - t0
vocos.wav.write(outp, y, count, 22050)
local m2, F2 = vocos.mel(y, count)
local acc, c = 0, 0
for b = 0, 79 do
  for f = 4, math.min(F, F2) - 5 do acc = acc + math.abs(m2[b * F2 + f] - mel[b * F + f]); c = c + 1 end
end
print(string.format("%s: %.1f s of speech in %.1f s; mel difference %.3f", outp, count / 22050, secs, acc / math.max(c, 1)))
