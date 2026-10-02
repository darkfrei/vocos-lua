-- examples/convert.lua -- the ONNX model into a compact .vocos file, and
-- optionally smaller for one voice.
--   luajit examples/convert.lua vocos-22khz-univ.onnx out.vocos [f16] [KEEP voice1.wav voice2.wav ...]
-- f16: 2 bytes a number (27 MB instead of 54).
-- KEEP: keep only the KEEP neurons of every block (of 1536) that give the
-- most on the given recordings of the voice (no retraining: measured on one
-- male voice, the mel difference of resynthesis grows from 0.189 to 0.244
-- keeping 1024, 0.311 keeping 768, 0.520 keeping 512).
package.path = "./?.lua;./?/init.lua;" .. package.path
local vocos = require("vocos")
local src, dst = arg[1], arg[2]
assert(src and dst, "usage: luajit examples/convert.lua MODEL out.vocos [f16] [KEEP voice.wav ...]")
local i, half, keep = 3, false, nil
if arg[i] == "f16" then half = true; i = i + 1 end
if arg[i] then keep = assert(tonumber(arg[i]), "KEEP must be a number"); i = i + 1 end
local v = vocos.load(src)
if keep then
  local mels = {}
  for j = i, #arg do
    local x, n = vocos.wav.read_mono(arg[j])
    local mel, F = vocos.mel(x, n)
    mels[#mels + 1] = {mel, F}
  end
  assert(#mels > 0, "KEEP needs recordings of the voice")
  v = v:prune(v:activity(mels), keep)
end
v:save(dst, half)
local f = io.open(dst, "rb"); local size = f:seek("end"); f:close()
print(string.format("%s: %d blocks, %.2f M numbers, %.1f MB", dst, v.L, v:count() / 1e6, size / 1e6))
