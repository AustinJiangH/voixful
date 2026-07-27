"""Qwen3-ASR (SALM: audio encoder + Qwen3 decoder) -> single Core AI .aimodel.

Functions emitted:
  encode_<W>        : mel [1,128,W] + length -> audio_embeds [1, (W//100)*13, 1024]
  embed_256         : token ids [1,256] -> embeds (prompt/audio/post splice)
  prefill_asr_<L>   : embeds+last_idx -> first token; writes K/V cache states
  step_qwen3        : one AR step (Qwen3 decoder — transcription)

Bundled in the asset dir: qwen3_asr.json (geometry + prompt ids + window table)
+ tokens.json (id -> base64 bytes for host detokenize).

Usage:
  python -m qwen3_asr_convert.convert --output qwen3-asr-0.6b-palette4.aimodel \
      [--windows 3000] [--quantize none|palette4|palette6] [--check-wav path.wav]
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import torch

import coreai_torch

from .export import export_embed, export_encoder, export_prefill, export_step
from .extract import extract_modules
from .models import (
    KV_CACHE_DTYPE_STR,
    MAX_SEQ,
    Qwen3ASRAudioEncoderForCoreAI,
    QwenEmbedForCoreAI,
    QwenPrefillForCoreAI,
    QwenStepForCoreAI,
    audio_out_len,
)
from .reference import EOS, MODEL_ID, load_model

DEFAULT_OUTPUT = "./qwen3-asr-0.6b.aimodel"
# Qwen3-ASR's WhisperFeatureExtractor pads to 30 s (3000 mel frames); the audio
# tower's max_source_positions is 1500. A single 30 s window covers every clip
# up to the model's trained horizon (host segments longer audio).
DEFAULT_WINDOWS = [3000]
EMBED_N = 256


def _asr_bucket(mel_frames: int, overhead: int) -> int:
    p = audio_out_len(mel_frames) + overhead
    return min(MAX_SEQ, ((p + 31) // 32) * 32 + 32)


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _dump_token_table(hf_tok, out_dir: Path, vocab_size: int) -> None:
    uni_to_byte = {u: b for b, u in _bytes_to_unicode().items()}
    special_ids = set(hf_tok.all_special_ids)
    table: list[str] = []
    for i in range(vocab_size):
        if i in special_ids:
            table.append("")
            continue
        s = hf_tok.convert_ids_to_tokens(i)
        if s is None:
            table.append("")
            continue
        try:
            raw = bytes(uni_to_byte[c] for c in s)
        except KeyError:
            raw = s.encode("utf-8")
        table.append(base64.b64encode(raw).decode("ascii"))
    (out_dir / "tokens.json").write_text(json.dumps({"format": 1, "tokens_b64": table}))


def _palettize(module: torch.nn.Module, example_inputs: tuple, kind: str) -> torch.nn.Module:
    from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig

    preset = {
        "palette4": KMeansPalettizerConfig.presets.w4,
        "palette6": KMeansPalettizerConfig.presets.w6,
    }[kind]
    pp = KMeansPalettizer(module, preset())
    pp.prepare(example_inputs=example_inputs)
    return pp.finalize()


def _palettize_decoder(text_lm: torch.nn.Module, example_inputs: tuple, kind: str) -> torch.nn.Module:
    """Palettize ONLY the 28 transformer layers; keep the TIED `embed_tokens`/
    `lm_head` (one shared table) and the final `norm` at full precision.

    The 0.6B decoder ties its embedding and output head. Palettizing that shared
    table quantizes the logits, and on the ANE (fp16 compute) the coarser logits
    flip argmax on ambiguous tokens — measured on-device as "a test of the speech"
    -> "the death of the" while CPU (fp32) stayed exact. Skipping the head/embed
    (a small fraction of the weights) keeps ANE accurate at negligible size cost;
    the 28 layers still palettize."""
    from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig

    preset = {
        "palette4": KMeansPalettizerConfig.presets.w4,
        "palette6": KMeansPalettizerConfig.presets.w6,
    }[kind]
    cfg = preset()
    cfg.set_module_name("lm_head", None)          # output head (tied with embed)
    cfg.set_module_name("model.embed_tokens", None)  # shared embedding table
    cfg.set_module_name("model.norm", None)          # final RMSNorm (tiny)
    pp = KMeansPalettizer(text_lm, cfg)
    pp.prepare(example_inputs=example_inputs)
    return pp.finalize()


def _mel(processor, wav_path: str, W: int) -> torch.Tensor:
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == 16000, f"expected 16 kHz, got {sr}"
    feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    mel = feats["input_features"].float()          # [1,128,T] (T<=3000, 30 s padded)
    T = mel.shape[-1]
    if T < W:
        mel = torch.nn.functional.pad(mel, (0, W - T))
    return mel[:, :, :W]


def _generate(prefill, step, embeds, prompt_len, bucket, hidden, max_new=256) -> list[int]:
    padded = torch.zeros(1, bucket, hidden)
    padded[:, :prompt_len] = embeds
    with torch.no_grad():
        tok, k_out, v_out = prefill(padded, torch.tensor([prompt_len - 1], dtype=torch.int32))
        L = k_out.shape[2]
        step.k_cache.zero_(); step.v_cache.zero_()
        step.k_cache[:, 0, :, :L, :] = k_out
        step.v_cache[:, 0, :, :L, :] = v_out
        out: list[int] = []
        pos = prompt_len
        while len(out) < max_new:
            t = int(tok.item())
            if t == EOS:
                break
            out.append(t)
            tok = step(torch.tensor([t], dtype=torch.int32), torch.tensor([pos], dtype=torch.int32))
            pos += 1
    return out


def _clean(text: str) -> str:
    # The model emits "language <Lang><asr_text><transcript>"; the host strips the
    # same way. Keep only what follows the last <asr_text> marker.
    marker = "<asr_text>"
    return text.split(marker)[-1].strip() if marker in text else text.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qwen3-asr-convert")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    parser.add_argument("--quantize", choices=["none", "palette4", "palette6"], default="palette4")
    parser.add_argument("--check-wav", default=None)
    args = parser.parse_args(argv)
    output_path = Path(args.output).expanduser()
    windows = sorted({int(w) for w in args.windows.split(",")})
    for w in windows:
        assert w % 100 == 0, f"window {w} must be a multiple of 100 mel frames"

    model, processor = load_model()
    mods = extract_modules(model, processor)
    tok = mods["tokenizer"]
    text_lm = mods["text_lm"]
    hidden = text_lm.config.hidden_size
    vocab_size = text_lm.lm_head.weight.shape[0]
    pre_ids, post_ids = mods["pre_ids"], mods["post_ids"]
    overhead = len(pre_ids) + len(post_ids)
    print(f"pre_ids={pre_ids} post_ids={post_ids} hidden={hidden} vocab={vocab_size}")

    # Palettize shared weights ONCE (encoder + the single Qwen3 decoder).
    enc_full = Qwen3ASRAudioEncoderForCoreAI(mods["audio_tower"], windows[-1]).eval()
    if args.quantize != "none":
        enc_full = _palettize(
            enc_full,
            (torch.zeros(1, enc_full.n_mels, windows[-1]), torch.tensor([windows[-1]], dtype=torch.int32)),
            args.quantize)
        print(f"palettized audio encoder ({args.quantize})")
        # The 0.6B Qwen3 decoder is small: 4-bit palettization collapses its
        # transcription (emits the language tag, then nothing). 6-bit survives —
        # same asymmetry Canary hit (audio/ASR path is more fragile at 4-bit than
        # the encoder). Encoder stays at the requested level; decoder floors at 6.
        # NOTE (measured): keeping the tied embed/lm_head/norm at full precision
        # (via _palettize_decoder) did NOT change the on-device ANE result — the
        # transcription error there is fp16 COMPUTE in the encoder / decoder
        # layers, not logit-head quantization (CPU with the same palette weights
        # is exact). So we keep the lean whole-decoder palette6 asset; ANE accuracy
        # is a compute-placement concern (see convert/qwen3-asr/README.md).
        dec_kind = "palette6" if args.quantize == "palette4" else args.quantize
        text_lm = _palettize(text_lm, (torch.zeros(1, 8, dtype=torch.int64),), dec_kind)
        print(f"palettized text decoder ({dec_kind})")

        if args.check_wav:
            mel = _mel(processor, args.check_wav, windows[-1])
            enc_chk = Qwen3ASRAudioEncoderForCoreAI(mods["audio_tower"], windows[-1]).eval()
            enc_chk.t = enc_full.t
            with torch.no_grad():
                audio_embeds = enc_chk(mel, torch.tensor([windows[-1]], dtype=torch.int32))
            embed = QwenEmbedForCoreAI(text_lm).eval()
            with torch.no_grad():
                pe = embed(torch.tensor([pre_ids], dtype=torch.int32))
                po = embed(torch.tensor([post_ids], dtype=torch.int32))
            pemb = torch.cat([pe, audio_embeds, po], dim=1)
            P = pemb.shape[1]
            bucket = _asr_bucket(windows[-1], overhead)
            pf = QwenPrefillForCoreAI(text_lm, bucket).eval()
            st = QwenStepForCoreAI(text_lm).eval()
            ids = _generate(pf, st, pemb, P, bucket, hidden)
            text = _clean(tok.decode(ids))
            print(f"post-palettize ASR self-check: {text!r}")
            assert len(text.split()) >= 3, "palettized ASR collapsed — aborting"

    converter = coreai_torch.TorchConverter()
    manifest_windows = []
    for w in windows:
        wrapper = Qwen3ASRAudioEncoderForCoreAI(mods["audio_tower"], w).eval()
        wrapper.t = enc_full.t                       # shared (palettized) weights
        converter.add_exported_program(
            export_encoder(wrapper, w),
            entrypoint_name=f"encode_{w}",
            input_names=["mel", "mel_length"],
            output_names=["audio_embeds"])
        bucket = _asr_bucket(w, overhead)
        manifest_windows.append({"mel_frames": w, "audio_len": audio_out_len(w), "asr_bucket": bucket})
        print(f"exported encode_{w} (audio_len {audio_out_len(w)}, asr bucket {bucket})")

    converter.add_exported_program(
        export_embed(QwenEmbedForCoreAI(text_lm).eval(), EMBED_N),
        entrypoint_name=f"embed_{EMBED_N}",
        input_names=["ids"], output_names=["embeds"])
    print(f"exported embed_{EMBED_N}")

    for b in sorted({m["asr_bucket"] for m in manifest_windows}):
        converter.add_exported_program(
            export_prefill(QwenPrefillForCoreAI(text_lm, b).eval(), hidden),
            entrypoint_name=f"prefill_asr_{b}",
            input_names=["embeds", "last_idx"],
            output_names=["token", "k_out", "v_out"])
        print(f"exported prefill_asr_{b}")
    converter.add_exported_program(
        export_step(QwenStepForCoreAI(text_lm).eval()),
        entrypoint_name="step_qwen3",
        input_names=["tok", "pos"],
        state_names=["k_cache", "v_cache"],
        output_names=["token"])
    print("exported step_qwen3")

    ai_program = converter.to_coreai()
    ai_program.optimize()
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)
        print(f"Replaced existing {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ai_program.save_asset(output_path)

    (output_path / "qwen3_asr.json").write_text(json.dumps({
        "format": 1,
        "model_id": MODEL_ID,
        "eos_id": EOS,
        # Qwen3-ASR decodes "language <Lang><asr_text><transcript>"; the host skips
        # generated tokens up to and including this id, then detokenizes the rest.
        "asr_text_id": tok.convert_tokens_to_ids("<asr_text>"),
        "max_seq": MAX_SEQ,
        "embed_n": EMBED_N,
        "hidden": hidden,
        "num_layers": text_lm.config.num_hidden_layers,
        "kv_heads": text_lm.config.num_key_value_heads,
        "head_dim": getattr(text_lm.config, "head_dim",
                            hidden // text_lm.config.num_attention_heads),
        "n_mels": enc_full.n_mels,
        "kv_dtype": KV_CACHE_DTYPE_STR,
        "asr_pre_ids": pre_ids,
        "asr_post_ids": post_ids,
        "windows": manifest_windows,
    }, indent=1))
    _dump_token_table(tok, output_path, vocab_size)
    print(f"Wrote {output_path} (quantize={args.quantize}) + qwen3_asr.json + tokens.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
