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
    export_cached_decoder,
    export_cross_kv_projector,
    export_decoder,
    export_encoder,
    trace_t_enc,
)
from .extract import DEFAULT_MODEL_ID, load_cohere
from .models import (
    CohereCrossKVProjectorForCoreAI,
    CohereDecoderStepCachedForCoreAI,
    CohereDecoderStepForCoreAI,
    CohereEncoderForCoreAI,
)

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
    parser.add_argument(
        "--windows",
        default="875,1750,3500",
        help=(
            "Comma-separated encoder mel-frame windows. The traced encoder always "
            "runs its full window, so a short clip in the 35 s window burns 48-layer "
            "compute on zero padding; exporting several windows lets the runtime pick "
            "the smallest >= the clip's frames. The LARGEST must stay 3500 (the "
            "decoder is traced against its T_enc=438) and keeps the name 'encode'; "
            "smaller windows are 'encode_<frames>' and the runtime zero-pads their "
            "output up to 438 (the cross-attention mask hides the padding)."
        ),
    )
    args = parser.parse_args(argv)
    output_path = Path(args.output)
    windows = sorted({int(w) for w in args.windows.split(",")})
    assert windows[-1] == MEL_FRAMES, (
        f"largest --windows entry must be {MEL_FRAMES} (the decoder's cross-attention "
        f"is traced against its T_enc); got {windows[-1]}"
    )

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

    print("Exporting decoder_step ...")
    exported_decoder = export_decoder(decoder_wrapper, t_enc)

    # Cached-decoder pair: cross-K/V projected once per utterance into state
    # buffers; the per-step decoder reads them (and mutates the self-attention
    # cache) via zero-copy states instead of threading ~14 MB/token of tensors.
    projector_wrapper = CohereCrossKVProjectorForCoreAI(model).eval()
    cached_decoder_wrapper = CohereDecoderStepCachedForCoreAI(
        model, max_cache_len=CACHE_LEN
    ).eval()
    print("Exporting project_cross_kv ...")
    exported_projector = export_cross_kv_projector(projector_wrapper, t_enc)
    print("Exporting decode_step_cached ...")
    exported_cached_decoder = export_cached_decoder(cached_decoder_wrapper, t_enc)

    converter = coreai_torch.TorchConverter()
    # One encoder program per window; the largest (3500 — the decoder's traced
    # T_enc contract) is the canonical `encode`, smaller are `encode_<frames>`.
    # The runtime zero-pads a smaller window's output up to T_enc=438.
    for w in windows:
        name = ENCODE_FN if w == windows[-1] else f"{ENCODE_FN}_{w}"
        print(f"Exporting encoder window {w} frames -> function '{name}' ...")
        converter.add_exported_program(
            export_encoder(encoder_wrapper, mel_frames=w),
            entrypoint_name=name,
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
    converter.add_exported_program(
        exported_projector,
        entrypoint_name="project_cross_kv",
        input_names=["encoder_hidden_states"],
        output_names=["cross_k", "cross_v"],
    )
    converter.add_exported_program(
        exported_cached_decoder,
        entrypoint_name="decode_step_cached",
        input_names=[
            "input_id", "position_id",
            "self_attention_mask", "cross_attention_mask",
            "cross_k", "cross_v",
        ],
        state_names=["k_cache", "v_cache"],
        output_names=["logits"],
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
