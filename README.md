# vocos.lua

[Русская версия для начинающих: README_ru.md](README_ru.md)

The [Vocos](https://github.com/gemelo-ai/vocos) neural vocoder (mel spectrogram → speech) in pure LuaJIT.

- **No ONNX Runtime, no C libraries, no GPU.** The weights are read straight out of the published `.onnx` file by a small protobuf reader written in Lua, and the network runs in LuaJIT with FFI arrays.
- **Exact.** Its output matches ONNX Runtime on the same model to about 1e-5. The test suite checks this against stored reference numbers.
- **Complete.** It includes the mel analysis that matches the model (librosa / torchaudio settings), the STFT and its inverse, a band-limited resampler, and WAV reading and writing.
- **Smaller files.** It can save the network in its own format, either 4 or 2 bytes per number (54 MB → 27 MB, same sound). It can also keep only the neurons a given voice actually uses.

It was written for a game made with [LÖVE](https://love2d.org), which ships LuaJIT. The same code runs anywhere `luajit` does.

## Model

Download `vocos-22khz-univ.onnx` (54 MB) from the sherpa-onnx releases:

    https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

This is [BSC-LT/vocos-mel-22khz](https://huggingface.co/BSC-LT/vocos-mel-22khz) (Apache-2.0), exported to ONNX by k2-fsa. It is universal (trained on many speakers): whose voice you hear depends only on the mel you give it.

Fixed settings: 22050 Hz, n_fft 1024, hop 256, 80 slaney mel bands (0–8000 Hz), natural log of the magnitude mel spectrum floored at 1e-5.

## Use

```lua
package.path = "path/to/vocos-lua/?.lua;path/to/vocos-lua/?/init.lua;" .. package.path
local vocos = require("vocos")

local v = vocos.load("vocos-22khz-univ.onnx")      -- or a .vocos file (below)

-- a recording through the vocoder and back
local x, n = vocos.wav.read_mono("speech.wav")     -- any rate, mono or stereo -> 22050 Hz floats
local mel, F = vocos.mel(x, n)                     -- float[80 * F], band-major
local y, count = v:synth(mel, F)                   -- float[count], 22050 Hz
vocos.wav.write("out.wav", y, count, 22050)
```

A mel from your own text-to-speech model works the same way: pass `float[80 * F]`, band `b` and frame `f` at `b * F + f`, natural-log magnitude.

Lower level:
- `v:forward(mel, F)` → magnitude, cos and sin of the phase, each `double[513 * F]`;
- `vocos.dsp.istft(mag, cos, sin, F, taper)` makes the samples;
- `vocos.dsp.stft(x, n)` is the analysis.

`synth` fades out the top above 7.8 kHz, where the mel (which stops at 8 kHz) gives the network nothing to go on. `v:synth(mel, F, false)` turns this off.

### Command line

    luajit examples/resynth.lua vocos-22khz-univ.onnx in.wav out.wav
    luajit examples/convert.lua vocos-22khz-univ.onnx vocos.vocos f16

`resynth` prints how far the mel of the result is from the mel it was made from: the mean |log| difference over 80 bands. For clean speech this is about 0.18–0.19 for Vocos.

## Speed and size

Measured on one core of a desktop CPU:

| | file | per second of speech | mel difference |
|---|---|---|---|
| `.onnx` (float32) | 54 MB | ~0.95 s | 0.187 / 0.180 / 0.183 |
| `.vocos` float16 | 27 MB | ~0.95 s | 0.187 / 0.180 / 0.183 |

The mel differences are for three recordings of one male reader. The network takes about 0.2 s to read from the `.onnx` file.

In a game, run it in a `love.thread` and synthesise a phrase ahead of when it is needed.

## Smaller for one voice

Vocos is universal, so for any single voice many of its neurons barely work. `examples/convert.lua` can keep, in each of the 8 blocks, only the neurons that contribute most on recordings of that voice (their mean output times the size of what they write back):

    luajit examples/convert.lua vocos-22khz-univ.onnx voice.vocos f16 1024 voice1.wav voice2.wav voice3.wav

Without retraining, on one male voice, mel difference by neurons kept per block (of 1536):

| kept per block | numbers | mel difference |
|---|---|---|
| 1536 (all) | 13.5 M | 0.189 |
| 1024 | 9.3 M | 0.24 |
| 768 | 7.2 M | 0.31 |
| 512 | 5.1 M | 0.52 |

Cutting more needs retraining against the full model. That is done elsewhere (the voice project this was written for); this library runs the result. The pointwise layers skip zero weights, so sparse weights are also faster.

## Tests

    luajit tests/test.lua
    VOCOS=vocos-22khz-univ.onnx luajit tests/test.lua    # also against ONNX Runtime's numbers

## Files

    vocos/init.lua      load, mel
    vocos/net.lua       the network: from ONNX, forward, synth, activity, prune, save/load
    vocos/dsp.lua       FFT, mel filterbank, mel analysis, STFT, inverse STFT
    vocos/onnx.lua      .onnx reader (protobuf, pure Lua)
    vocos/resample.lua  windowed-sinc resampler, stereo -> mono
    vocos/wav.lua       16-bit PCM WAV
    examples/           resynth.lua, convert.lua
    tests/              test.lua, onnxruntime_reference.lua

## License

MIT (see `LICENSE`). The model weights are not part of this repository; they are Apache-2.0 from BSC-LT.

Vocos: Hubert Siuzdak, *Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis*, 2023.
