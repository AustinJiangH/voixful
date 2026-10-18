# Granite Speech 5.0 TurboCTC → Core AI

Converts [`ibm-granite/granite-speech-5.0-470m-turboctc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc)
(470M, English, Apache-2.0) into a self-contained `.aimodel` for the
`VoixfulGraniteTurboCTC` backend.

The model is a 16-layer block-attention conformer with self-conditioned CTC and
a 16,384-entry byte-level BPE head, decoded greedily — so the asset is **one
graph per audio window and nothing else**:

| Function | Inputs | Output |
|---|---|---|
| `ctc_<F>` | `audio [1, F*320] f32`, `length [1] i32` (real samples) | `ids [1, F/4] i32` (per-frame argmax, blank 0 past `length`) |

Everything model-side is in-graph: the exact `GraniteSpeech5FeatureExtractor`
front-end (STFT/mel via DFT matmuls, log10 floor at max−8, deltas, 2-frame
stacking), the encoder's padding mask (driven by `length`, so a padded window
decodes exactly like the unpadded clip), the CTC head and the argmax. The
Swift side only collapses repeats/blanks and detokenizes with the bundled
`tokens.json`. `turboctc.json` carries the window table.

Default windows: 256 / 512 / 1024 / 1536 stacked 50 Hz frames (≈5 / 10 / 20 / 30 s).

Output is **lowercase without punctuation** (plain CTC); numbers come out
normalized ("12%", "1st").

## Setup

Own venv — the checkpoint uses native `GraniteSpeech5ForCTC` weight names, so
it needs `transformers>=5.16` (do not share with `convert/granite-speech`):

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --pre -e . coreai-opt
```

## Convert

```sh
# coreai-opt JIT-compiles its k-means kernel with ninja: keep the venv on PATH
PATH=$PWD/.venv/bin:$PATH python -m turboctc_convert.convert \
    --quantize palette4 --check-wav ../../fixtures/sample.wav \
    --output granite-speech-5.0-470m-turboctc-palette4.aimodel
```

`--check-wav` runs the palettized torch pipeline and prints it next to the
fp32 upstream reference before the (slower) export.

## License

Upstream weights: Apache-2.0 (IBM). Converter code: Apache-2.0.
