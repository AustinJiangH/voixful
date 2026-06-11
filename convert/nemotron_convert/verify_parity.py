"""Compare NeMo reference vs the converted .aimodel on a 30-second clip. See PLAN.md §3 Step 6."""

from __future__ import annotations

import argparse
import math

import numpy as np
import soundfile as sf
import torch
from coreai.runtime import AIModel

from .convert import ENCODE_FN, JOIN_FN, PREDICT_STEP_FN
from .extract import extract_modules_from_nemo
from .models import (
    NemotronEncoderForCoreAI,
    NemotronJoinerForCoreAI,
    NemotronPredictionForCoreAI,
)

# PLAN.md §3 Step 6: tolerate up to ~40 dB SNR on encoder outputs (FP32 -> FP16 drift)
# but require exact match on token sequences.
ENCODER_SNR_DB_THRESHOLD = 40.0

# PLAN.md §2: blank id is the last token (vocab 13,087).
BLANK_ID = 13087


def _snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Signal-to-noise ratio in dB between two same-shape float arrays."""

    diff = reference - candidate
    signal_power = float(np.mean(reference.astype(np.float64) ** 2))
    noise_power = float(np.mean(diff.astype(np.float64) ** 2))
    if noise_power <= 0.0:
        return math.inf
    if signal_power <= 0.0:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)


def _load_wav(path: str) -> np.ndarray:
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    # VERIFY: sample-rate handling — PLAN.md §2 mandates 16 kHz mono. If the test
    # fixture is not 16 kHz mono, resample (e.g., via librosa/soxr) before this returns.
    if sr != 16_000:
        raise ValueError(f"Expected 16 kHz audio, got {sr} Hz from {path}")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples.astype(np.float32)


def _lang_id_one_hot(target_lang: str) -> np.ndarray:
    """Build the 128-dim one-hot language ID vector per PLAN.md §2 / §6.

    VERIFY: language-id table — PLAN.md §6 mentions a 128-dim one-hot but does not pin
    down the exact lang -> index mapping. Inspect NeMo's tokenizer or `modules["config"]`
    in Phase 1 to obtain the canonical mapping.
    """

    _ = target_lang  # placeholder until the lang table is resolved
    vec = np.zeros((1, 128), dtype=np.float32)
    vec[0, 0] = 1.0
    return vec


def _run_nemo_reference(
    modules: dict, samples: np.ndarray, lang_one_hot: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """Run a NeMo-side reference forward pass: preprocessor -> encoder -> RNN-T greedy
    decode. Returns (final encoder output, decoded token ids).

    VERIFY: streaming vs. full-clip — for a 30-second clip we can choose to drive the
    streaming pipeline chunk-by-chunk (matching the .aimodel's runtime usage) or to
    forward the whole clip in one shot. PLAN.md §3 Step 6 says "feed both the same 30-
    second clip"; the streaming path is the more faithful test. The skeleton below
    iterates chunks; confirm the exact chunk size and cache reset policy in Phase 2/3.
    """

    encoder_wrapper = NemotronEncoderForCoreAI(modules["encoder"])
    prediction_wrapper = NemotronPredictionForCoreAI(modules["prediction"])
    joiner_wrapper = NemotronJoinerForCoreAI(modules["joiner"])
    encoder_wrapper.eval()
    prediction_wrapper.eval()
    joiner_wrapper.eval()

    with torch.no_grad():
        # VERIFY: preprocessor signature — NeMo's preprocessor expects (input_signal,
        # length) and returns (mel, mel_length). Confirm in Phase 1 by inspecting
        # `modules["preprocessor"].forward`.
        signal = torch.from_numpy(samples).unsqueeze(0)
        length = torch.tensor([samples.shape[0]], dtype=torch.int64)
        mel, _mel_len = modules["preprocessor"](input_signal=signal, length=length)

        # 320 ms chunk @ 10 ms hop = 32 frames; PLAN.md §2 default.
        # VERIFY: chunk frame count — depends on the resolved `att_context_size` mapping.
        chunk_frames = 32
        lang_t = torch.from_numpy(lang_one_hot)

        last_enc_out: torch.Tensor | None = None
        tokens: list[int] = []
        # VERIFY: utterance-boundary semantics — PLAN.md §4 mandates a single reset of
        # all caches per utterance; the wrappers ship initialized to zero so the first
        # chunk is fine, but a multi-utterance test would call .reset() between clips.

        for start in range(0, mel.shape[2], chunk_frames):
            mel_chunk = mel[:, :, start : start + chunk_frames]
            if mel_chunk.shape[2] == 0:
                break
            enc_out = encoder_wrapper(mel_chunk, lang_t)
            last_enc_out = enc_out

            # Greedy RNN-T loop per PLAN.md §6 — pure Python reference version.
            for t in range(enc_out.shape[1]):
                enc_step = enc_out[:, t : t + 1, :]
                prev = tokens[-1] if tokens else BLANK_ID
                symbols_this_step = 0
                while symbols_this_step < 10:
                    prev_tok = torch.tensor([prev], dtype=torch.int32)
                    pred_emb = prediction_wrapper(prev_tok)
                    logits = joiner_wrapper(enc_step, pred_emb)
                    token = int(logits.reshape(-1).argmax().item())
                    if token == BLANK_ID:
                        break
                    tokens.append(token)
                    prev = token
                    symbols_this_step += 1

    assert last_enc_out is not None, "encoder produced no output on the input clip"
    return last_enc_out.cpu().numpy(), tokens


def _run_coreai(
    aimodel_path: str,
    samples: np.ndarray,
    lang_one_hot: np.ndarray,
    preprocessor,
) -> tuple[np.ndarray, list[int]]:
    """Run the same clip through the converted .aimodel via the Python `coreai.runtime`
    bindings.

    VERIFY: coreai.runtime API — PLAN.md §3 Step 6 names `coreai.runtime.AIModel` as
    the Python-side loader; the exact method names (e.g., `load_function`, `run`,
    state-binding shape) are not pinned in the plan. Confirm against the WWDC 2026
    session sample code once `coreai` is installed.
    """

    model = AIModel.load(aimodel_path)
    encode_fn = model.load_function(ENCODE_FN)
    predict_fn = model.load_function(PREDICT_STEP_FN)
    join_fn = model.load_function(JOIN_FN)

    with torch.no_grad():
        signal = torch.from_numpy(samples).unsqueeze(0)
        length = torch.tensor([samples.shape[0]], dtype=torch.int64)
        mel, _mel_len = preprocessor(input_signal=signal, length=length)
    mel_np = mel.cpu().numpy()

    chunk_frames = 32
    last_enc_out: np.ndarray | None = None
    tokens: list[int] = []

    for start in range(0, mel_np.shape[2], chunk_frames):
        mel_chunk = mel_np[:, :, start : start + chunk_frames]
        if mel_chunk.shape[2] == 0:
            break
        # VERIFY: state-binding API shape — could be `states=` dict, in-place mutation,
        # or returned-then-rebound. PLAN.md §4 Swift snippet uses `MutableViews`; the
        # Python equivalent is what needs confirming here.
        enc_out = encode_fn.run(inputs={"mel_chunk": mel_chunk, "lang_id": lang_one_hot})[
            "enc_out"
        ]
        last_enc_out = enc_out

        for t in range(enc_out.shape[1]):
            enc_step = enc_out[:, t : t + 1, :]
            prev = tokens[-1] if tokens else BLANK_ID
            symbols_this_step = 0
            while symbols_this_step < 10:
                prev_tok = np.array([prev], dtype=np.int32)
                pred_emb = predict_fn.run(inputs={"prev_token": prev_tok})["pred_emb"]
                logits = join_fn.run(
                    inputs={"enc_step": enc_step, "pred_emb": pred_emb}
                )["logits"]
                token = int(np.argmax(logits.reshape(-1)))
                if token == BLANK_ID:
                    break
                tokens.append(token)
                prev = token
                symbols_this_step += 1

    assert last_enc_out is not None, "Core AI encoder produced no output on the input clip"
    return last_enc_out, tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nemotron-verify-parity",
        description=(
            "Compare NeMo reference vs the converted .aimodel: encoder SNR + exact "
            "token-sequence match. See PLAN.md §3 Step 6."
        ),
    )
    parser.add_argument("--model-id", default="nvidia/nemotron-3.5-asr-streaming-0.6b")
    parser.add_argument("--aimodel", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--target-lang", default="en-US")
    args = parser.parse_args(argv)

    samples = _load_wav(args.wav)
    lang_one_hot = _lang_id_one_hot(args.target_lang)

    modules = extract_modules_from_nemo(args.model_id)
    nemo_enc, nemo_tokens = _run_nemo_reference(modules, samples, lang_one_hot)
    coreai_enc, coreai_tokens = _run_coreai(
        args.aimodel, samples, lang_one_hot, modules["preprocessor"]
    )

    # Encoder SNR check.
    snr = _snr_db(nemo_enc, coreai_enc)
    enc_pass = snr >= ENCODER_SNR_DB_THRESHOLD
    print(
        f"[encoder]  SNR = {snr:.2f} dB  "
        f"(threshold >= {ENCODER_SNR_DB_THRESHOLD:.1f} dB) -> "
        f"{'PASS' if enc_pass else 'FAIL'}"
    )

    # Token-sequence exact-match check.
    tokens_pass = nemo_tokens == coreai_tokens
    print(
        f"[tokens]   nemo={len(nemo_tokens)} coreai={len(coreai_tokens)} "
        f"exact_match={tokens_pass} -> {'PASS' if tokens_pass else 'FAIL'}"
    )
    if not tokens_pass:
        # Show the first divergence for fast triage.
        for i, (a, b) in enumerate(zip(nemo_tokens, coreai_tokens)):
            if a != b:
                print(f"           first diff at index {i}: nemo={a} coreai={b}")
                break

    return 0 if (enc_pass and tokens_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
