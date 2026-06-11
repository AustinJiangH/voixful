"""Default: FP16. No coreai_opt invocation. See PLAN.md §3 Step 5b for the conditional quantization path."""

from __future__ import annotations

import argparse
from pathlib import Path

import coreai_torch

from .export import export_encoder, export_joiner, export_prediction
from .extract import extract_modules_from_nemo
from .models import (
    NemotronEncoderForCoreAI,
    NemotronJoinerForCoreAI,
    NemotronPredictionForCoreAI,
)

# The 40-language `nvidia/nemotron-3.5-asr-streaming-0.6b` checkpoint requires
# NeMo's unreleased `EncDecRNNTBPEModelWithPrompt` class (not in NeMo 2.7.3 on
# PyPI as of 2026-06). The English-only sibling uses the standard
# `EncDecRNNTBPEModel` and converts cleanly. Switch back to the 40-locale
# checkpoint once the upstream class lands in a NeMo release.
DEFAULT_MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
DEFAULT_OUTPUT = "./nemotron-streaming.aimodel"
DEFAULT_MAX_CHUNK_FRAMES = 256  # Headroom — the real model wants min 121 frames per its streaming cfg.

# The three named functions packed into a single multi-function .aimodel per
# PLAN.md §3 Step 4 and Decision Log #3.
ENCODE_FN = "encode"
PREDICT_STEP_FN = "predict_step"
JOIN_FN = "join"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nemotron-convert",
        description=(
            "Convert nvidia/nemotron-3.5-asr-streaming-0.6b to a single Core AI "
            ".aimodel with three functions (encode, predict_step, join). "
            "FP16 default; see PLAN.md §3 Step 5 for rationale."
        ),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-chunk-frames",
        type=int,
        default=DEFAULT_MAX_CHUNK_FRAMES,
        help="Upper bound on dynamic mel time axis (10 ms hop). 112 == 1120 ms.",
    )
    args = parser.parse_args(argv)

    output_path = Path(args.output)

    # Step (a): pull the raw nn.Modules out of the NeMo ASRModel.
    modules = extract_modules_from_nemo(args.model_id)

    # Step (b): build the three Core-AI-shaped wrappers.
    # VERIFY: d_model — PLAN.md §10 open question 2. The wrapper defaults to 1024
    # (matching soniqo/speech-swift's cache shape). Once Phase 1 inspects
    # modules["config"], pass the resolved value through here rather than relying on
    # the default.
    encoder_wrapper = NemotronEncoderForCoreAI(modules["encoder"])
    prediction_wrapper = NemotronPredictionForCoreAI(modules["prediction"])
    joiner_wrapper = NemotronJoinerForCoreAI(modules["joiner"])

    encoder_wrapper.eval()
    prediction_wrapper.eval()
    joiner_wrapper.eval()

    # Step (c): torch.export each.
    exported_encoder = export_encoder(encoder_wrapper, max_chunk_frames=args.max_chunk_frames)
    exported_prediction = export_prediction(prediction_wrapper)
    exported_joiner = export_joiner(joiner_wrapper)

    # Step (d): pack into one multi-function .aimodel per PLAN.md §3 Step 4.
    converter = coreai_torch.TorchConverter()

    converter.add_exported_program(
        exported_encoder,
        entrypoint_name=ENCODE_FN,
        # English-only checkpoint has no language conditioning; input is
        # (mel_chunk, mel_length). The multilingual variant adds a lang_id
        # input — re-add when switching back to the 40-language checkpoint.
        input_names=["mel_chunk", "mel_length"],
        # Three streaming caches per the verified shapes; no `pre_cache` in
        # this checkpoint (see models.py header).
        state_names=[
            "cache_last_channel",
            "cache_last_time",
            "cache_last_channel_len",
        ],
        output_names=["enc_out"],
    )
    converter.add_exported_program(
        exported_prediction,
        entrypoint_name=PREDICT_STEP_FN,
        input_names=["prev_token"],
        state_names=["lstm_h", "lstm_c"],
        output_names=["pred_emb"],
    )
    converter.add_exported_program(
        exported_joiner,
        entrypoint_name=JOIN_FN,
        input_names=["enc_step", "pred_emb"],
        output_names=["logits"],
    )

    # VERIFY: add_exported_program kwargs — PLAN.md §3 Step 4 snippet does not show a
    # `entrypoint_name=` keyword explicitly. The WWDC 2026 "Dive into Core AI" session
    # demonstrated multi-function .aimodel packing; confirm the exact keyword name
    # (could be `name=` or positional) against the coreai_torch API once installed.

    # Step (e): convert, framework-level optimize (no weight quantization), save.
    # PLAN.md §3 Step 5: `coreai_opt` is NOT invoked here. FP16 default.
    ai_program = converter.to_coreai()
    ai_program.optimize()
    ai_program.save_asset(output_path)

    print(
        f"Wrote {output_path} — a single Core AI multi-function .aimodel containing "
        f"the three named inference functions '{ENCODE_FN}', '{PREDICT_STEP_FN}', and "
        f"'{JOIN_FN}' at FP16 precision (no coreai_opt quantization applied; see "
        f"PLAN.md §3 Step 5b for the conditional quantization path)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
