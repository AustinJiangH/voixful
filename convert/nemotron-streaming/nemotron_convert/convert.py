"""Default: FP16. No coreai_opt invocation. See PLAN.md §3 Step 5b for the conditional quantization path."""

from __future__ import annotations

import argparse
from pathlib import Path

import coreai_torch
import torch

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
    parser.add_argument(
        "--quantize",
        choices=["none", "int8", "int8-mmap", "palette4", "palette6"],
        default="none",
        help=(
            "Encoder weight compression (PLAN.md §3 Step 5b). 'int8'/'int8-mmap' = "
            "INT8 linear weight-only (disk/bandwidth). 'palette4'/'palette6' = "
            "4/6-bit k-means palettization — the format Apple documents as the "
            "Neural Engine's runtime-memory + latency win (kept compressed, "
            "decompressed just-in-time). Prediction LSTM + joiner stay FP16."
        ),
    )
    args = parser.parse_args(argv)

    output_path = Path(args.output)

    # Build the INT8 weight-only quantizer BEFORE importing NeMo (coreai_opt's
    # pydantic models resolve cleanly only in a NeMo-free import state). Use the
    # official `QuantizerConfig.presets.w8()` recipe — data-free, weight-only,
    # op key 'weight'. See apple/coreai-optimization + WWDC26 session 325.
    quantize_encoder = None
    if args.quantize in ("int8", "int8-mmap"):
        from coreai_opt.quantization import ExecutionMode, Quantizer, QuantizerConfig

        mmap_dir = None
        if args.quantize == "int8-mmap":
            # mmap_dir requires eager mode, which also quantizes conv ops — they
            # need an explicit per-channel axis (axis=0 = output channels) that
            # the graph-mode preset resolves automatically but eager does not.
            from coreai_opt.quantization import (
                ModuleQuantizerConfig,
                QuantizationSpec,
            )
            from coreai_opt.quantization.spec.granularity import PerChannelGranularity

            wspec = QuantizationSpec(
                dtype=torch.int8, granularity=PerChannelGranularity(axis=0)
            )
            mqc = ModuleQuantizerConfig(
                op_input_spec=None,
                op_output_spec=None,
                op_state_spec={"weight": wspec},
                op_type_config=None,
                op_name_config=None,
                module_input_spec=None,
                module_output_spec=None,
                module_state_spec=None,
                qat_schedule=None,
            )
            qcfg = QuantizerConfig(global_config=mqc, execution_mode=ExecutionMode.EAGER)
            mmap_dir = output_path.parent / f"{output_path.name}.weights"
            mmap_dir.mkdir(parents=True, exist_ok=True)
        else:
            qcfg = QuantizerConfig.presets.w8()

        def quantize_encoder(enc):
            # Match export_encoder's fixed-shape example inputs (T=128).
            example_mel = torch.zeros(1, 128, 128, dtype=torch.float32)
            example_length = torch.tensor([128], dtype=torch.int32)
            q = Quantizer(enc, qcfg)
            q.prepare(example_inputs=(example_mel, example_length))
            return q.finalize(mmap_dir=str(mmap_dir)) if mmap_dir else q.finalize()

    elif args.quantize in ("palette4", "palette6"):
        # K-means weight palettization (4/6-bit LUT). Apple's docs flag this as
        # the Neural Engine's runtime-memory format (weights stay low-bit,
        # converted on the fly layer-to-layer).
        from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig

        _preset = {
            "palette4": KMeansPalettizerConfig.presets.w4,
            "palette6": KMeansPalettizerConfig.presets.w6,
        }[args.quantize]
        pcfg = _preset()

        def quantize_encoder(enc):
            example_mel = torch.zeros(1, 128, 128, dtype=torch.float32)
            example_length = torch.tensor([128], dtype=torch.int32)
            p = KMeansPalettizer(enc, pcfg)
            p.prepare(example_inputs=(example_mel, example_length))
            return p.finalize()

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

    # Step (b2): apply the INT8 weight-only quantizer to the encoder.
    if quantize_encoder is not None:
        encoder_wrapper = quantize_encoder(encoder_wrapper)
        print("Applied INT8 weight-only quantization to the encoder (presets.w8).")

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
    if output_path.exists():
        # save_asset refuses to overwrite (FileExistsError on main.mlirb);
        # treat the output like any build artifact and replace it.
        import shutil
        shutil.rmtree(output_path)
        print(f"Replaced existing {output_path}")
    ai_program.save_asset(output_path)

    # Bundle the SentencePiece vocab inside the .aimodel so the runtime is
    # self-contained (StreamingASR reads <model>/pieces.txt first).
    pieces = list(modules["tokenizer"].vocab)
    # Fail-fast: the English-only Nemotron checkpoint has a 1024-token vocab. If a
    # model/tooling swap changes this, abort at convert time rather than shipping an
    # asset the Swift runtime later rejects (vocab mismatch vs the joiner's 1025-wide
    # logits = 1024 tokens + blank).
    EXPECTED_VOCAB = 1024
    assert len(pieces) == EXPECTED_VOCAB, (
        f"pieces.txt vocab mismatch: got {len(pieces)} pieces, expected "
        f"{EXPECTED_VOCAB} for nvidia/nemotron-speech-streaming-en-0.6b. "
        "A model or tokenizer swap changed the vocab — fix the architecture "
        "constants before shipping this asset."
    )
    (output_path / "pieces.txt").write_text("\n".join(pieces) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}/pieces.txt ({len(pieces)} pieces) — self-contained.")

    print(
        f"Wrote {output_path} — a single Core AI multi-function .aimodel containing "
        f"the three named inference functions '{ENCODE_FN}', '{PREDICT_STEP_FN}', and "
        f"'{JOIN_FN}' at FP16 precision (no coreai_opt quantization applied; see "
        f"PLAN.md §3 Step 5b for the conditional quantization path)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
