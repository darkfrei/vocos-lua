-- vocos.lua -- the Vocos vocoder (mel spectrogram -> speech) in pure LuaJIT.
--
--   local vocos = require("vocos")
--   local v = vocos.load("vocos-22khz-univ.onnx")    -- or a .vocos file
--   local x, n = vocos.wav.read_mono("speech.wav")   -- 22050 Hz floats
--   local mel, F = vocos.mel(x, n)                   -- float[80 * F]
--   local y, count = v:synth(mel, F)                 -- float[count]
--   vocos.wav.write("out.wav", y, count, 22050)
--
-- no ONNX Runtime, no C library, no GPU: the weights are read straight out
-- of the .onnx file and the network runs in LuaJIT (about 1 second per
-- second of speech on one core of a desktop CPU, full size).

local M = {}
M.dsp = require("vocos.dsp")
M.net = require("vocos.net")
M.wav = require("vocos.wav")
M.VERSION = "1.0"

-- a network from an .onnx file (vocos-22khz-univ.onnx) or a .vocos file
-- written by Net:save
function M.load(path)
  local f = assert(io.open(path, "rb"))
  local head = f:read(9)
  f:close()
  if head == M.net.MAGIC then return M.net.load_file(path) end
  return M.net.from_onnx(path)
end

-- samples (22050 Hz) -> log mel float[80 * F], F
function M.mel(x, n) return M.dsp.mel(x, n) end

return M
