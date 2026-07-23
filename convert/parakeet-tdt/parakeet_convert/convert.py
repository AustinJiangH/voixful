"""Parakeet TDT 0.6B -> single Core AI .aimodel (encode / predict_step / join).

Full-context (batch) model — no streaming caches. TDT joint emits token + duration
logits. `--quantize` mirrors the Nemotron pipeline (int8 linear or palettization).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coreai_torch
import torch

from .export import (
    export_enc_projector,
    export_encoder,
    export_join_fused,
    export_joiner,
    export_prediction,
)
from .extract import extract_modules_from_nemo
from .models import (
    ParakeetEncoderForCoreAI,
    ParakeetJoinerForCoreAI,
    ParakeetPredictionForCoreAI,
)

DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_OUTPUT = "./parakeet-tdt.aimodel"

ENCODE_FN = "encode"
PREDICT_STEP_FN = "predict_step"
JOIN_FN = "join"


def _build_quantizer(kind: str):
    """Return a callable encoder->encoder, built BEFORE NeMo import (coreai_opt's
    pydantic models resolve cleanly only in a NeMo-free import state)."""
    if kind == "none":
        return None

    example_mel = torch.zeros(1, 128, 1000, dtype=torch.float32)
    example_length = torch.tensor([1000], dtype=torch.int32)

    if kind in ("int8",):
        from coreai_opt.quantization import Quantizer, QuantizerConfig

        cfg = QuantizerConfig.presets.w8()

        def q(enc):
            qq = Quantizer(enc, cfg)
            qq.prepare(example_inputs=(example_mel, example_length))
            return qq.finalize()

        return q

    if kind in ("palette4", "palette6"):
        from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig

        preset = {
            "palette4": KMeansPalettizerConfig.presets.w4,
            "palette6": KMeansPalettizerConfig.presets.w6,
        }[kind]
        cfg = preset()

        def p(enc):
            pp = KMeansPalettizer(enc, cfg)
            pp.prepare(example_inputs=(example_mel, example_length))
            return pp.finalize()

        return p

    raise ValueError(kind)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parakeet-convert")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--quantize",
        choices=["none", "int8", "palette4", "palette6"],
        default="none",
        help="Encoder compression: palette4/palette6 = the Neural-Engine RAM+speed win.",
    )
    parser.add_argument(
        "--windows",
        default="125,250,375,500,750,1000",
        help=(
            "Comma-separated encoder mel-frame windows. The traced encoder always "
            "runs its full window, so a short clip in a long window burns compute "
            "on zero padding; exporting several windows lets the runtime pick the "
            "smallest >= the clip's frames. The LARGEST keeps the name 'encode' "
            "(runtime/back-compat contract); smaller ones are 'encode_<frames>'."
        ),
    )
    args = parser.parse_args(argv)
    output_path = Path(args.output)
    windows = sorted({int(w) for w in args.windows.split(",")})

    quantize_encoder = _build_quantizer(args.quantize)  # before NeMo import

    modules = extract_modules_from_nemo(args.model_id)

    encoder_wrapper = ParakeetEncoderForCoreAI(modules["encoder"]).eval()
    prediction_wrapper = ParakeetPredictionForCoreAI(modules["prediction"]).eval()
    joiner_wrapper = ParakeetJoinerForCoreAI(modules["joiner"]).eval()

    if quantize_encoder is not None:
        encoder_wrapper = quantize_encoder(encoder_wrapper)
        print(f"Applied {args.quantize} compression to the encoder.")

    exported_prediction = export_prediction(prediction_wrapper)
    exported_joiner = export_joiner(joiner_wrapper)

    converter = coreai_torch.TorchConverter()
    # One encoder program per window; the largest is the canonical `encode`
    # (existing runtime contract), smaller windows are `encode_<frames>`.
    for w in windows:
        name = ENCODE_FN if w == windows[-1] else f"{ENCODE_FN}_{w}"
        converter.add_exported_program(
            export_encoder(encoder_wrapper, mel_frames=w),
            entrypoint_name=name,
            input_names=["mel", "mel_length"],
            output_names=["enc_out"],
        )
        print(f"Exported encoder window {w} frames -> function '{name}'")
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
        output_names=["logits"],  # [B, T, U, vocab+1+num_durations]
    )

    # Fused decode pair: project the whole clip's enc_out once, then a join
    # whose two argmaxes run in-graph — per-step host I/O drops from a
    # [1,1,1,8198] logits readback to two int32 scalars.
    from .models import ParakeetEncProjectorForCoreAI, ParakeetJoinFusedForCoreAI

    projector_wrapper = ParakeetEncProjectorForCoreAI(modules["joiner"]).eval()
    fused_wrapper = ParakeetJoinFusedForCoreAI(modules["joiner"]).eval()
    print("Exporting project_enc (T=125) + join_fused ...")
    converter.add_exported_program(
        export_enc_projector(projector_wrapper),
        entrypoint_name="project_enc",
        input_names=["enc_out"],
        output_names=["enc_proj"],
    )
    converter.add_exported_program(
        export_join_fused(fused_wrapper),
        entrypoint_name="join_fused",
        input_names=["enc_proj_t", "pred_emb"],
        output_names=["token", "duration"],
    )

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
    # self-contained (the CLI reads <model>/pieces.txt; a wrong vocab errors).
    pieces = list(modules["tokenizer"].vocab)
    # Fail-fast: Parakeet-TDT uses an 8192-token SentencePiece vocab. If a model or
    # tokenizer swap changes this, abort at convert time rather than shipping an asset
    # the Swift runtime later rejects on a vocab-size mismatch.
    EXPECTED_VOCAB = 8192
    assert len(pieces) == EXPECTED_VOCAB, (
        f"pieces.txt vocab mismatch: got {len(pieces)} pieces, expected "
        f"{EXPECTED_VOCAB} for Parakeet-TDT. A model or tokenizer swap changed the "
        "vocab — fix the architecture constants before shipping this asset."
    )
    (output_path / "pieces.txt").write_text("\n".join(pieces) + "\n", encoding="utf-8")

    print(f"Wrote {output_path} (quantize={args.quantize}) + pieces.txt ({len(pieces)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
