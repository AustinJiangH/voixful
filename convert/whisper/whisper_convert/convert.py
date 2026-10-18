"""OpenAI Whisper -> single Core AI .aimodel (whisper_encode / whisper_decode).

Attention encoder-decoder, same pipeline shape as Cohere Transcribe. `--quantize`
palettizes the encoder, mirroring the Nemotron/Parakeet/Cohere pipelines.

The decode entrypoint is deliberately named `whisper_decode` (NOT Cohere's
`decode_step*`) so the Swift `ModelBackend.classify` heuristic can tell Whisper
apart from Cohere. The tokenizer vocab (`tokens.txt`, byte-level BPE) and the
architecture/special-token manifest (`whisper.json`) are written INSIDE the
`.aimodel` dir so the asset is self-contained and the specialize cache keys on
the final contents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import coreai_torch

from .export import (
    encoder_example,
    export_decoder,
    export_encoder,
    trace_t_enc,
)
from .extract import DEFAULT_MODEL_ID, load_whisper
from .models import WhisperDecoderStepForCoreAI, WhisperEncoderForCoreAI

DEFAULT_OUTPUT = "./whisper.aimodel"

N_FFT = 400
HOP_LENGTH = 160
SAMPLE_RATE = 16_000


def _build_quantizer(kind: str, n_mels: int, mel_frames: int):
    """Encoder compressor (mirrors the Cohere pipeline)."""
    if kind == "none":
        return None
    example = encoder_example(n_mels, mel_frames)
    if kind in ("palette4", "palette6"):
        from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig

        preset = {
            "palette4": KMeansPalettizerConfig.presets.w4,
            "palette6": KMeansPalettizerConfig.presets.w6,
        }[kind]
        cfg = preset()

        def p(enc):
            pp = KMeansPalettizer(enc, cfg)
            pp.prepare(example_inputs=example)
            return pp.finalize()

        return p
    raise ValueError(kind)


def _special_tokens(tokenizer) -> dict:
    """Whisper special / language token ids, resolved from the tokenizer."""
    def tid(tok: str) -> int:
        return int(tokenizer.convert_tokens_to_ids(tok))

    specials = {
        "sot": tid("<|startoftranscript|>"),
        "eot": tid("<|endoftext|>"),
        "transcribe": tid("<|transcribe|>"),
        "translate": tid("<|translate|>"),
        "no_timestamps": tid("<|notimestamps|>"),
        "lang_en": tid("<|en|>"),
    }
    # All Whisper specials are appended after the byte-BPE ranks, and <|endoftext|>
    # is the FIRST of them — so it doubles as the "skip ids >= this" boundary the
    # Swift tokenizer uses to drop specials/timestamps.
    specials["special_token_start"] = specials["eot"]

    languages = {}
    try:
        from transformers.models.whisper.tokenization_whisper import LANGUAGES

        for code in LANGUAGES:
            tok = f"<|{code}|>"
            cid = tokenizer.convert_tokens_to_ids(tok)
            # `convert_tokens_to_ids` returns the UNK id (which aliases <|endoftext|>
            # = eot) for language tokens absent from this checkpoint — e.g. <|yue|>
            # only exists in large-v3. Round-trip so a bogus id (it would prime the
            # decoder with EOT and instantly stop) never lands in the map.
            if isinstance(cid, int) and cid >= 0 and tokenizer.convert_ids_to_tokens(cid) == tok:
                languages[code] = cid
    except Exception:
        languages = {"en": specials["lang_en"]}
    return specials, languages


def _decoder_prompt(model, tokenizer, specials, vocab: int) -> tuple[bool, list]:
    """Read the checkpoint's ACTUAL forced decoder prompt from the HF model.

    Returns `(is_multilingual, forced_decoder_ids)`.

    Whisper primes its decoder with a fixed prompt after `<|startoftranscript|>`.
    The two families differ:

      - multilingual (`whisper-base`, `-large-v3`, …): a language slot + a TASK
        token (`<|transcribe|>` / `<|translate|>`) are forced — e.g.
        `[[1, None], [2, <|transcribe|>]]` (language `None` = auto-detect).
      - English-only (`.en`: `whisper-base.en`, `tiny.en`, …): only
        `<|notimestamps|>` is forced — `[[1, <|notimestamps|>]]`. NO language,
        NO task token. Feeding `<|en|><|transcribe|>` to a `.en` checkpoint is
        off-distribution (garbled / instant-EOT output).

    The reliable discriminator is whether a TASK token is forced — `prefix_tokens`
    alone is not (multilingual auto-detect leaves the language slot empty, so it
    is also `[sot, notimestamps]`). We fall back to the vocab size (the `.en`
    checkpoints ship the 51864-token vocab, one language block smaller than the
    51865/51866 multilingual vocabs) only if the generation config is missing.
    """
    transcribe = specials["transcribe"]
    translate = specials["translate"]
    forced = []
    try:
        forced = list(model.generation_config.forced_decoder_ids or [])
    except Exception:
        forced = []
    task_forced = any(
        tokid in (transcribe, translate)
        for (_pos, tokid) in forced
        if isinstance(tokid, int)
    )
    if forced:
        is_multilingual = task_forced
    else:
        # No generation config to read: fall back to the vocab-size heuristic
        # (English-only checkpoints are exactly one 100-token language block
        # smaller than the multilingual variants).
        is_multilingual = vocab >= 51865
    return is_multilingual, forced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="whisper-convert")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--quantize",
        choices=["none", "palette4", "palette6"],
        default="none",
        help="Encoder compression: palette4 = the RAM win (see convert/README.md).",
    )
    parser.add_argument(
        "--max-decode-tokens",
        type=int,
        default=224,
        help=(
            "Decoder self-attention cache slots = the per-30 s-chunk token cap. "
            "224 is Whisper's default max content tokens (half of n_text_ctx=448)."
        ),
    )
    args = parser.parse_args(argv)
    output_path = Path(args.output)

    print(f"Loading {args.model_id} ...")
    model = load_whisper(args.model_id)
    cfg = model.config

    n_mels = cfg.num_mel_bins
    d_model = cfg.d_model
    n_layers = cfg.decoder_layers
    n_heads = cfg.decoder_attention_heads
    head_dim = d_model // n_heads
    vocab = cfg.vocab_size
    max_cache_len = args.max_decode_tokens
    # Encoder conv2 has stride 2, so the mel window is 2x the encoder frames.
    mel_frames = cfg.max_source_positions * 2

    encoder_wrapper = WhisperEncoderForCoreAI(model).eval()
    t_enc = trace_t_enc(encoder_wrapper, n_mels, mel_frames)
    print(f"Encoder output time length T_enc = {t_enc} (from mel {n_mels}x{mel_frames})")
    # Fail-fast: the decoder's cross-attention is traced against this exact T_enc.
    assert t_enc == cfg.max_source_positions, (
        f"encoder T_enc mismatch: traced {t_enc}, expected {cfg.max_source_positions}. "
        "The mel config or encoder changed — fix the constants before shipping."
    )

    quantize = _build_quantizer(args.quantize, n_mels, mel_frames)
    if quantize is not None:
        encoder_wrapper = quantize(encoder_wrapper)
        print(f"Applied {args.quantize} compression to the encoder.")

    decoder_wrapper = WhisperDecoderStepForCoreAI(model, max_cache_len=max_cache_len).eval()

    print("Exporting whisper_encode ...")
    exported_encoder = export_encoder(encoder_wrapper, n_mels, mel_frames)
    print("Exporting whisper_decode ...")
    exported_decoder = export_decoder(
        decoder_wrapper, t_enc, d_model, n_layers, n_heads, head_dim, max_cache_len)

    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported_encoder,
        entrypoint_name="whisper_encode",
        input_names=["input_features"],
        output_names=["encoder_hidden_states"],
    )
    converter.add_exported_program(
        exported_decoder,
        entrypoint_name="whisper_decode",
        input_names=[
            "input_id", "position_id", "encoder_hidden_states",
            "self_attention_mask", "k_cache", "v_cache",
        ],
        output_names=["logits", "k_cache_out", "v_cache_out"],
    )

    ai_program = converter.to_coreai()
    if hasattr(ai_program, "optimize"):  # coreai-core < 1.0.0b3; b3+ optimizes on build
        ai_program.optimize()
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)
        print(f"Replaced existing {output_path}")
    ai_program.save_asset(output_path)

    # Bundle the byte-level BPE vocab (tokens.txt) + the manifest (whisper.json).
    from transformers import WhisperTokenizer

    tokenizer = WhisperTokenizer.from_pretrained(args.model_id)
    pieces = [tokenizer.convert_ids_to_tokens(i) for i in range(vocab)]
    assert len(pieces) == vocab, (
        f"tokens.txt vocab mismatch: got {len(pieces)}, expected {vocab}."
    )
    # Byte-level BPE token strings never contain a literal newline (0x0A maps to
    # 'Ċ' under the GPT-2 byte<->unicode table), so one-per-line is safe.
    (output_path / "tokens.txt").write_text("\n".join(pieces) + "\n", encoding="utf-8")

    specials, languages = _special_tokens(tokenizer)
    is_multilingual, forced_decoder_ids = _decoder_prompt(
        model, tokenizer, specials, vocab)
    print(
        f"Decoder prompt: is_multilingual={is_multilingual} "
        f"forced_decoder_ids={forced_decoder_ids}")
    manifest = {
        "format": 1,
        "n_mels": n_mels,
        "mel_frames": mel_frames,
        "chunk_samples": mel_frames * HOP_LENGTH,
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_audio_ctx": t_enc,
        "d_model": d_model,
        "n_layer": n_layers,
        "n_head": n_heads,
        "vocab_size": vocab,
        "max_decode_tokens": max_cache_len,
        # The decoder's forced prompt, read from the checkpoint. `is_multilingual`
        # drives the Swift prefix: multilingual primes [sot, lang, transcribe,
        # notimestamps] (language resolved per-locale at runtime); English-only
        # primes [sot, notimestamps]. `forced_decoder_ids` is the raw HF value,
        # bundled for provenance. Absence of `is_multilingual` in older assets is
        # treated as multilingual (back-compat).
        "is_multilingual": is_multilingual,
        "forced_decoder_ids": forced_decoder_ids,
        "languages": languages,
        **specials,
    }
    (output_path / "whisper.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"Wrote {output_path} (quantize={args.quantize}, T_enc={t_enc}) + "
        f"tokens.txt ({len(pieces)}) + whisper.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
