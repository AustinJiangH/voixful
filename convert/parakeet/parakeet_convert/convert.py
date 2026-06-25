"""Parakeet TDT 0.6B -> single Core AI .aimodel (encode / predict_step / join).

Full-context (batch) model — no streaming caches. TDT joint emits token + duration
logits. `--quantize` mirrors the Nemotron pipeline (int8 linear or palettization).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coreai_torch
import torch

from .export import export_encoder, export_joiner, export_prediction
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
    args = parser.parse_args(argv)
    output_path = Path(args.output)

    quantize_encoder = _build_quantizer(args.quantize)  # before NeMo import

    modules = extract_modules_from_nemo(args.model_id)

    encoder_wrapper = ParakeetEncoderForCoreAI(modules["encoder"]).eval()
    prediction_wrapper = ParakeetPredictionForCoreAI(modules["prediction"]).eval()
    joiner_wrapper = ParakeetJoinerForCoreAI(modules["joiner"]).eval()

    if quantize_encoder is not None:
        encoder_wrapper = quantize_encoder(encoder_wrapper)
        print(f"Applied {args.quantize} compression to the encoder.")

    exported_encoder = export_encoder(encoder_wrapper)
    exported_prediction = export_prediction(prediction_wrapper)
    exported_joiner = export_joiner(joiner_wrapper)

    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported_encoder,
        entrypoint_name=ENCODE_FN,
        input_names=["mel", "mel_length"],
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
        output_names=["logits"],  # [B, T, U, vocab+1+num_durations]
    )

    ai_program = converter.to_coreai()
    ai_program.optimize()
    ai_program.save_asset(output_path)

    # Bundle the SentencePiece vocab inside the .aimodel so the runtime is
    # self-contained (the CLI reads <model>/pieces.txt; a wrong vocab errors).
    pieces = list(modules["tokenizer"].vocab)
    (output_path / "pieces.txt").write_text("\n".join(pieces) + "\n", encoding="utf-8")

    print(f"Wrote {output_path} (quantize={args.quantize}) + pieces.txt ({len(pieces)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
