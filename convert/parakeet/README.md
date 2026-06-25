# convert/parakeet — Parakeet TDT 0.6B → Core AI `.aimodel`

Converts NVIDIA **Parakeet TDT 0.6B** (`nvidia/parakeet-tdt-0.6b-v3` / `-v2`) — the
model FluidAudio ships — into a Core AI `.aimodel`, for a same-model comparison.

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

## Run

```sh
uv pip install -e . --no-deps                 # in the shared coreai venv
uv pip install "pydantic>=2.6,<2.10" && uv pip install --pre coreai-opt   # for --quantize
python -m parakeet_convert.convert --quantize palette4 --output parakeet-palette4.aimodel
# --quantize: none | int8 | palette4 | palette6
```

## Status / results

**Conversion works** — produces a 3-function `.aimodel` (`encode`/`predict_step`/`join`).
Measured sizes (M5 Pro):

| Variant | `.aimodel` size |
|---|---|
| FP16 | 2.3 GB |
| `palette4` | 364 MB (~6.3×) |

**Not yet runnable in the Swift runtime.** Our `NemotronASR.StreamingASR` is built
for the *streaming, cache-aware, RNN-T* Nemotron path. Parakeet needs:
- a **full-context encode** driver (feed the whole padded clip once), and
- a **TDT greedy decoder** (split token/duration logits, advance time by duration).

That runtime work (a `VoixfulParakeet` backend) is the follow-up that unlocks
transcripts + the RAM/RTFx/WER comparison vs FluidAudio.

## License

- Conversion code → Apache-2.0 (repo root `LICENSE`/`NOTICE`).
- `.aimodel` weights → derived from `nvidia/parakeet-tdt-0.6b-v3`, **CC-BY-4.0**
  (attribution only). Credit NVIDIA / the source model when redistributing.
