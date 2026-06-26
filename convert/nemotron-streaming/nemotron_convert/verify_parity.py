"""Compare the NeMo reference forward against the converted .aimodel on a clip.

Drives both paths chunk-by-chunk through the streaming encoder + RNN-T greedy
decode and checks:

  * encoder SNR  — FP32 reference vs FP16 .aimodel encoder outputs (>= threshold)
  * exact token-sequence match — the decoded RNN-T token ids must be identical

This targets the English-only checkpoint `nvidia/nemotron-speech-streaming-en-0.6b`
(vocab 1024, blank id 1024). There is no language conditioning on this model, so
the encoder takes `(mel_chunk, mel_length)` only.

The encoder is exported fixed-shape at T=128 mel frames (see export.py), so both
paths feed 128-frame chunks (zero-padded on the final chunk) with `mel_length`
set to the real frame count of that chunk. The three streaming caches are carried
across chunks: the NeMo wrapper threads them via `register_buffer`; the Core AI
runtime threads them via in-place-mutated `state` NDArrays allocated once.
"""

from __future__ import annotations

import argparse
import asyncio
import math

import numpy as np
import soundfile as sf
import torch

from .convert import DEFAULT_MODEL_ID, ENCODE_FN, JOIN_FN, PREDICT_STEP_FN
from .extract import extract_modules_from_nemo
from .models import (
    NemotronEncoderForCoreAI,
    NemotronJoinerForCoreAI,
    NemotronPredictionForCoreAI,
)

# Tolerate FP32 -> FP16 drift on encoder outputs but require exact token match.
ENCODER_SNR_DB_THRESHOLD = 40.0

# Encoder is exported fixed-shape at T=128 mel frames (export.py:export_encoder).
ENCODER_CHUNK_FRAMES = 128

# Greedy RNN-T: cap symbols emitted per encoder frame to avoid runaway loops.
MAX_SYMBOLS_PER_STEP = 10

# Cache shapes for the English-only checkpoint (models.py header / verified against
# the converted asset's state descriptors). cache_last_channel_len is int32 in the
# converted asset (the FP16 converter coerced the int64 buffer).
N_LAYERS = 24
D_MODEL = 1024
CACHE_CHANNEL_SIZE = 70
CACHE_TIME_SIZE = 8


def _snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Signal-to-noise ratio in dB between two same-shape float arrays."""

    diff = reference.astype(np.float64) - candidate.astype(np.float64)
    signal_power = float(np.mean(reference.astype(np.float64) ** 2))
    noise_power = float(np.mean(diff**2))
    if noise_power <= 0.0:
        return math.inf
    if signal_power <= 0.0:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)


def _load_wav(path: str) -> np.ndarray:
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32)
    if sr != 16_000:
        # NeMo expects 16 kHz mono. Resample with scipy (GCD-reduced poly).
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), 16_000)
        samples = resample_poly(samples, 16_000 // g, int(sr) // g).astype(np.float32)
    return samples


def _compute_mel(preprocessor, samples: np.ndarray) -> np.ndarray:
    """Run NeMo's mel preprocessor. Returns mel as numpy [1, 128, T]."""

    with torch.no_grad():
        signal = torch.from_numpy(samples).unsqueeze(0)
        length = torch.tensor([samples.shape[0]], dtype=torch.int64)
        mel, _mel_len = preprocessor(input_signal=signal, length=length)
    return mel.cpu().numpy()


def _iter_chunks(mel: np.ndarray):
    """Yield (padded_mel_chunk [1, 128, ENCODER_CHUNK_FRAMES], valid_frame_count).

    The final chunk is zero-padded out to ENCODER_CHUNK_FRAMES so it matches the
    encoder's fixed-shape export; `valid_frame_count` carries the real length.
    """

    total = mel.shape[2]
    for start in range(0, total, ENCODER_CHUNK_FRAMES):
        valid = min(ENCODER_CHUNK_FRAMES, total - start)
        if valid <= 0:
            break
        chunk = mel[:, :, start : start + valid]
        if valid < ENCODER_CHUNK_FRAMES:
            pad = np.zeros(
                (chunk.shape[0], chunk.shape[1], ENCODER_CHUNK_FRAMES - valid),
                dtype=chunk.dtype,
            )
            chunk = np.concatenate([chunk, pad], axis=2)
        yield chunk, valid


# ---------------------------------------------------------------------------
# NeMo reference path
# ---------------------------------------------------------------------------


def _run_nemo_reference(
    modules: dict, mel: np.ndarray, blank_id: int
) -> tuple[np.ndarray, list[int]]:
    """Run the NeMo-side reference: encoder (chunked, cache threaded via buffers)
    -> RNN-T greedy decode. Returns (concatenated encoder output, token ids).
    """

    encoder = NemotronEncoderForCoreAI(modules["encoder"]).eval()
    prediction = NemotronPredictionForCoreAI(modules["prediction"]).eval()
    joiner = NemotronJoinerForCoreAI(modules["joiner"]).eval()

    enc_outputs: list[np.ndarray] = []
    tokens: list[int] = []

    with torch.no_grad():
        for chunk, valid in _iter_chunks(mel):
            mel_chunk = torch.from_numpy(chunk)
            mel_length = torch.tensor([valid], dtype=torch.int32)
            # The wrapper threads the three streaming caches via register_buffer.
            enc_out = encoder(mel_chunk, mel_length)  # [1, T_out, d_model]
            enc_np = enc_out.cpu().numpy()
            enc_outputs.append(enc_np)

            for t in range(enc_out.shape[1]):
                enc_step = enc_out[:, t : t + 1, :]  # [1, 1, d_model]
                prev = tokens[-1] if tokens else blank_id
                emitted = 0
                while emitted < MAX_SYMBOLS_PER_STEP:
                    prev_tok = torch.tensor([[prev]], dtype=torch.int32)
                    pred_emb = prediction(prev_tok)  # [1, 1, pred_hidden]
                    logits = joiner(enc_step, pred_emb)  # [1, 1, 1, vocab+1]
                    token = int(logits.reshape(-1).argmax().item())
                    if token == blank_id:
                        break
                    tokens.append(token)
                    prev = token
                    emitted += 1

    enc_all = np.concatenate(enc_outputs, axis=1)
    return enc_all, tokens


# ---------------------------------------------------------------------------
# Core AI runtime path
# ---------------------------------------------------------------------------


async def _run_coreai(
    aimodel_path: str, mel: np.ndarray, blank_id: int
) -> tuple[np.ndarray, list[int]]:
    """Run the converted .aimodel through the `coreai.runtime` Python bindings.

    State NDArrays are allocated once and threaded across chunks; the runtime
    mutates them in place (the streaming cache).
    """

    from coreai.runtime import (
        AIModel,
        ComputeUnitKind,
        NDArray,
        SpecializationOptions,
    )

    # The default (auto-partitioned) compute path intermittently hits an
    # MPSGraph compile error ("Unresolved symbol: MPSGraph") on this asset.
    # Pinning a single compute unit avoids the problematic CPU+GPU split.
    # Try GPU first, then Neural Engine.
    model = None
    last_err: Exception | None = None
    for factory in (ComputeUnitKind.gpu, ComputeUnitKind.neural_engine):
        try:
            opts = SpecializationOptions.from_preferred_compute_unit_kind(factory())
            model = await AIModel.load(aimodel_path, opts)
            # Force a function load now so a backend compile failure surfaces here.
            model.load_function(ENCODE_FN)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            model = None
    if model is None:
        # Last resort: the unconstrained default path.
        model = await AIModel.load(aimodel_path)
    _ = last_err

    encode_fn = model.load_function(ENCODE_FN)
    predict_fn = model.load_function(PREDICT_STEP_FN)
    join_fn = model.load_function(JOIN_FN)

    # Encoder streaming caches — allocated once, mutated in place across chunks.
    enc_state = {
        "cache_last_channel": NDArray(
            np.zeros((1, N_LAYERS, CACHE_CHANNEL_SIZE, D_MODEL), dtype=np.float32)
        ),
        "cache_last_time": NDArray(
            np.zeros((1, N_LAYERS, D_MODEL, CACHE_TIME_SIZE), dtype=np.float32)
        ),
        "cache_last_channel_len": NDArray(np.zeros((1,), dtype=np.int32)),
    }
    # Prediction LSTM state — same threading.
    pred_state = {
        "lstm_h": NDArray(np.zeros((2, 1, 640), dtype=np.float32)),
        "lstm_c": NDArray(np.zeros((2, 1, 640), dtype=np.float32)),
    }

    enc_outputs: list[np.ndarray] = []
    tokens: list[int] = []

    for chunk, valid in _iter_chunks(mel):
        mel_chunk = NDArray(chunk.astype(np.float32))
        mel_length = NDArray(np.array([valid], dtype=np.int32))
        out = await encode_fn(
            inputs={"mel_chunk": mel_chunk, "mel_length": mel_length},
            state=enc_state,
        )
        enc_out = out["enc_out"].numpy()  # [1, T_out, d_model]
        enc_outputs.append(enc_out)

        for t in range(enc_out.shape[1]):
            enc_step = np.ascontiguousarray(enc_out[:, t : t + 1, :])  # [1,1,d_model]
            enc_step_nd = NDArray(enc_step)
            prev = tokens[-1] if tokens else blank_id
            emitted = 0
            while emitted < MAX_SYMBOLS_PER_STEP:
                prev_tok = NDArray(np.array([[prev]], dtype=np.int32))
                pred_out = await predict_fn(
                    inputs={"prev_token": prev_tok}, state=pred_state
                )
                pred_emb = pred_out["pred_emb"]
                join_out = await join_fn(
                    inputs={"enc_step": enc_step_nd, "pred_emb": pred_emb}
                )
                logits = join_out["logits"].numpy()
                token = int(np.argmax(logits.reshape(-1)))
                if token == blank_id:
                    break
                tokens.append(token)
                prev = token
                emitted += 1

    enc_all = np.concatenate(enc_outputs, axis=1)
    return enc_all, tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nemotron-verify-parity",
        description=(
            "Compare the NeMo reference forward vs the converted .aimodel: "
            "encoder SNR + exact RNN-T token-sequence match."
        ),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--aimodel", required=True)
    parser.add_argument("--wav", required=True)
    args = parser.parse_args(argv)

    samples = _load_wav(args.wav)

    modules = extract_modules_from_nemo(args.model_id)
    # Blank id = vocab size (logits are [.., vocab + 1]; blank is the last index).
    vocab_size = int(modules["tokenizer"].vocab_size)
    blank_id = vocab_size
    tokenizer = modules["tokenizer"]
    print(f"[setup]    vocab_size={vocab_size}  blank_id={blank_id}")

    mel = _compute_mel(modules["preprocessor"], samples)
    print(f"[setup]    mel shape={tuple(mel.shape)}  audio={len(samples) / 16000:.2f}s")

    nemo_enc, nemo_tokens = _run_nemo_reference(modules, mel, blank_id)

    # Decode the reference transcript so a human can eyeball it ("quick brown fox").
    try:
        ref_text = tokenizer.ids_to_text(nemo_tokens)
    except Exception:  # noqa: BLE001
        ref_text = "<unavailable>"
    print(f"[nemo]     tokens={len(nemo_tokens)}  text={ref_text!r}")

    coreai_available = True
    coreai_enc: np.ndarray | None = None
    coreai_tokens: list[int] = []
    try:
        coreai_enc, coreai_tokens = asyncio.run(
            _run_coreai(args.aimodel, mel, blank_id)
        )
        try:
            coreai_text = tokenizer.ids_to_text(coreai_tokens)
        except Exception:  # noqa: BLE001
            coreai_text = "<unavailable>"
        print(f"[coreai]   tokens={len(coreai_tokens)}  text={coreai_text!r}")
    except Exception as exc:  # noqa: BLE001
        coreai_available = False
        print(
            "[coreai]   FAILED to run the converted .aimodel via coreai.runtime: "
            f"{type(exc).__name__}: {exc}"
        )
        print(
            "[coreai]   The NeMo reference path above ran to completion; the Core "
            "AI comparison is being skipped (see error)."
        )

    if not coreai_available or coreai_enc is None:
        print("[result]   INCONCLUSIVE — Core AI path unavailable.")
        return 2

    # Encoder SNR check (compare overlapping frame count, in case of any drift).
    n = min(nemo_enc.shape[1], coreai_enc.shape[1])
    snr = _snr_db(nemo_enc[:, :n, :], coreai_enc[:, :n, :])
    enc_pass = snr >= ENCODER_SNR_DB_THRESHOLD
    print(
        f"[encoder]  SNR = {snr:.2f} dB  "
        f"(threshold >= {ENCODER_SNR_DB_THRESHOLD:.1f} dB) -> "
        f"{'PASS' if enc_pass else 'FAIL'}"
    )

    tokens_pass = nemo_tokens == coreai_tokens
    print(
        f"[tokens]   nemo={len(nemo_tokens)} coreai={len(coreai_tokens)} "
        f"exact_match={tokens_pass} -> {'PASS' if tokens_pass else 'FAIL'}"
    )
    if not tokens_pass:
        for i, (a, b) in enumerate(zip(nemo_tokens, coreai_tokens)):
            if a != b:
                print(f"           first diff at index {i}: nemo={a} coreai={b}")
                break
        if len(nemo_tokens) != len(coreai_tokens):
            print(
                f"           length mismatch: nemo={len(nemo_tokens)} "
                f"coreai={len(coreai_tokens)}"
            )

    overall = enc_pass and tokens_pass
    print(f"[result]   {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
