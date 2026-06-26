"""Cohere Transcribe 2B -> single Core AI .aimodel (encode / decode_step).

Attention encoder-decoder. `--quantize` palettizes the encoder (where >90% of the
2B params live), mirroring the Nemotron/Parakeet pipelines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coreai_torch
import torch

from .export import (
    CACHE_LEN,
    MEL_FRAMES,
    N_MELS,
    encoder_example,
    export_decoder,
    export_encoder,
    trace_t_enc,
)
from .extract import DEFAULT_MODEL_ID, load_cohere
from .models import CohereDecoderStepForCoreAI, CohereEncoderForCoreAI

DEFAULT_OUTPUT = "./cohere-transcribe.aimodel"
ENCODE_FN = "encode"
DECODE_FN = "decode_step"


def _build_quantizer(kind: str):
    """Encoder compressor, built BEFORE the model import path matters."""
    if kind == "none":
        return None
    mel, length = encoder_example()
    if kind in ("palette4", "palette6"):
        from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig

        preset = {
            "palette4": KMeansPalettizerConfig.presets.w4,
            "palette6": KMeansPalettizerConfig.presets.w6,
        }[kind]
        cfg = preset()

        def p(enc):
            pp = KMeansPalettizer(enc, cfg)
            pp.prepare(example_inputs=(mel, length))
            return pp.finalize()

        return p
    if kind == "int8":
        from coreai_opt.quantization import Quantizer, QuantizerConfig

        cfg = QuantizerConfig.presets.w8()

        def q(enc):
            qq = Quantizer(enc, cfg)
            qq.prepare(example_inputs=(mel, length))
            return qq.finalize()

        return q
    raise ValueError(kind)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cohere-convert")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--quantize",
        choices=["none", "int8", "palette4", "palette6"],
        default="none",
        help="Encoder compression (>90%% of params): palette4 = the RAM win.",
    )
    args = parser.parse_args(argv)
    output_path = Path(args.output)

    quantize_encoder = _build_quantizer(args.quantize)

    print(f"Loading {args.model_id} ...")
    model = load_cohere(args.model_id)

    encoder_wrapper = CohereEncoderForCoreAI(model.encoder, model.encoder_decoder_proj).eval()
    decoder_wrapper = CohereDecoderStepForCoreAI(model, max_cache_len=CACHE_LEN).eval()

    t_enc = trace_t_enc(encoder_wrapper)
    print(f"Encoder output time length T_enc = {t_enc} (from mel {N_MELS}x{MEL_FRAMES})")
    # Fail-fast: the decoder's cross-attention is exported against a fixed encoder
    # output length of 438 frames. If the traced length drifts (mel config / encoder
    # swap), abort here rather than shipping a decoder whose cross-attention mask
    # shape no longer matches the encoder output.
    EXPECTED_T_ENC = 438
    assert t_enc == EXPECTED_T_ENC, (
        f"encoder T_enc mismatch: traced {t_enc}, expected {EXPECTED_T_ENC}. "
        "The mel config or encoder changed — fix the architecture constants "
        "before shipping this asset."
    )

    if quantize_encoder is not None:
        encoder_wrapper = quantize_encoder(encoder_wrapper)
        print(f"Applied {args.quantize} compression to the encoder.")

    print("Exporting encoder ...")
    exported_encoder = export_encoder(encoder_wrapper)
    print("Exporting decoder_step ...")
    exported_decoder = export_decoder(decoder_wrapper, t_enc)

    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported_encoder,
        entrypoint_name=ENCODE_FN,
        input_names=["input_features", "length"],
        output_names=["encoder_hidden_states"],
    )
    converter.add_exported_program(
        exported_decoder,
        entrypoint_name=DECODE_FN,
        input_names=[
            "input_id", "position_id", "encoder_hidden_states",
            "self_attention_mask", "cross_attention_mask", "k_cache", "v_cache",
        ],
        output_names=["logits", "k_cache_out", "v_cache_out"],
    )

    ai_program = converter.to_coreai()
    ai_program.optimize()
    ai_program.save_asset(output_path)

    # Bundle the SentencePiece vocab inside the .aimodel so the runtime is
    # self-contained (the CLI reads <model>/pieces.txt; a wrong vocab errors).
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    pieces = [tok.convert_ids_to_tokens(i) for i in range(16384)]
    # Fail-fast: the Cohere transcribe vocab is 16384 tokens. If a model or tokenizer
    # swap changes this, abort at convert time rather than shipping an asset the Swift
    # runtime later rejects on a vocab-size mismatch.
    EXPECTED_VOCAB = 16384
    assert len(pieces) == EXPECTED_VOCAB, (
        f"pieces.txt vocab mismatch: got {len(pieces)} pieces, expected "
        f"{EXPECTED_VOCAB} for Cohere transcribe. A model or tokenizer swap changed "
        "the vocab — fix the architecture constants before shipping this asset."
    )
    (output_path / "pieces.txt").write_text("\n".join(pieces) + "\n", encoding="utf-8")

    print(f"Wrote {output_path} (quantize={args.quantize}, T_enc={t_enc}) + pieces.txt ({len(pieces)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
