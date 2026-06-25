# convert/parakeet — Parakeet TDT 0.6B → Core AI `.aimodel`

Converts NVIDIA **Parakeet TDT 0.6B** (`nvidia/parakeet-tdt-0.6b-v3` / `-v2`) into a
Core AI `.aimodel` — a full-context, TDT-transducer backend for Voixful.

## How it differs from the Nemotron pipeline

| | Nemotron streaming | Parakeet TDT |
|---|---|---|
| Encoder | cache-aware FastConformer (streaming, 3 cache tensors) | **full-context** Conformer (offline/batch, **no caches**) |
| Decoder/joint | standard RNN-T | **TDT** (Token-and-Duration Transducer) |
| Joint output | `[B,T,U, vocab+1]` | `[B,T,U, vocab+1 + num_durations]` (8193 + 5) |
| Vocab | 1024 | 8192 |
| License | NVIDIA Open Model License | **CC-BY-4.0** |

Two consequences for the runtime (see below):
1. The encoder takes the **whole clip's** mel at once (fixed at 1000 frames here —
   `coreai_torch 0.4.0` can't carry a dynamic time axis), not streamed chunks.
2. Greedy decode must **split the joint output** into token + duration logits and
   **advance the encoder time index by the predicted duration** — a TDT decoder,
   not the RNN-T `GreedyRNNTDecoder`.

## Setup

Standalone, like the other convert dirs — `uv` + Python 3.12 + the pre-release
Apple `coreai` wheels:

```sh
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e .                      # pulls the pre-release coreai wheels
uv pip install "pydantic>=2.6,<2.10"     # required for the --quantize path
```

(Plain pip works too with `--pre`: `python3.12 -m venv .venv && pip install --pre -e .`.)

## Run

```sh
python -m parakeet_convert.convert --quantize palette4 --output parakeet-tdt-0.6b-v3-palette4.aimodel
# --quantize: none | int8 | palette4 | palette6
```

## Status / results

**Conversion + Swift runtime both work.** The converter produces a 3-function
`.aimodel` (`encode`/`predict_step`/`join`), and `VoixfulEngine`'s
`ParakeetTDTEngine` runs it end-to-end (full-context encode + TDT greedy decode).
Drive it with:

```sh
voixful transcribe clip.wav --model parakeet-tdt-0.6b-v3-palette4.aimodel
```

Both the FP16 and `palette4` variants produce NeMo-identical transcripts; palettizing
the encoder cuts the asset from 2.3 GB to ~364 MB with no quality loss (same lever as
the Nemotron pipeline).

### Runtime gotchas (what the TDT path needs that the RNN-T path doesn't)

The `ParakeetTDTEngine` is a separate backend from the streaming Nemotron engine
because four things differ from the streaming path — each was a real bug found
bringing it up:

1. **`normalize: per_feature`.** Parakeet's preprocessor standardizes each mel bin
   to zero-mean/unit-std over the utterance (Nemotron uses `normalize: NA`). Feed
   un-normalized mel and the encoder goes out-of-distribution → the TDT decoder
   collapses to the prediction network's language-model prior (fluent nonsense).
2. **`torch.stft(center=True)` window centering.** With `win_length(400) < n_fft(512)`
   the Hann window sits **centered** in the FFT frame. Left-aligning it shifts every
   frame `(512-400)/2 = 56` samples early — enough to wreck the acoustics. The Swift
   `MelExtractor` now takes a `frameStartOffset` (Parakeet passes 56; Nemotron 0).
3. **SOS seed.** NeMo seeds the predictor with `g₀ = LSTM(zero_embed, zero_state)`,
   *not* a literal zero vector. The prediction embedding uses `padding_idx = blank`,
   so feeding the blank token once before decoding reproduces `g₀` exactly. Without
   it the joint emits blank through the start of the clip (lost first sentence).
4. **TDT greedy decode.** Split the joint output into token logits `[0, 8193)` and
   duration logits `[8193, 8198)`; argmax both; advance the encoder time index by the
   predicted duration (0–4), not always +1, with a per-frame symbol cap of 10.

## License

- Conversion code → Apache-2.0 (repo root `LICENSE`/`NOTICE`).
- `.aimodel` weights → derived from `nvidia/parakeet-tdt-0.6b-v3`, **CC-BY-4.0**
  (attribution only). Credit NVIDIA / the source model when redistributing.
