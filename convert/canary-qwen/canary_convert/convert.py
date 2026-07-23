"""Canary-Qwen 2.5B (SALM) -> single Core AI .aimodel.

Functions:
  encode_<T>        : mel [1,128,T] + length -> audio_embeds [1, T/8, 2048]
  embed_256         : token ids [1,256] -> embeds (prompt/transcript splice)
  prefill_asr_<L>   : embeds+last_idx -> first token; writes K/V cache states
  step_asr          : one AR step (LoRA-merged Qwen3 — transcription)
  prefill_llm_<L>   : same, pristine Qwen3-1.7B-instruct (formatting pass)
  step_llm

Weight sharing: encoder palettized once; each Qwen3 variant palettized once
(as the bare HF model) and every prefill/step/embed wrapper is built around
the SAME palettized instance. Post-palettize transcript self-check runs in
torch BEFORE the slow export.

Bundled in the asset dir: canary.json (prompt template ids, eos, window and
bucket tables) + tokens.json (id -> base64 bytes for host detokenize).

Usage:
  python -m canary_convert.convert --output canary-qwen.aimodel \
      [--windows 1000,2000,4000] [--quantize none|palette4|palette6] \
      [--check-wav path.wav]
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
    MAX_SEQ,
    CanaryEncoderForCoreAI,
    QwenEmbedForCoreAI,
    QwenPrefillForCoreAI,
    QwenStepForCoreAI,
)
from .reference import FORMAT_PROMPT, MODEL_ID, load_model
from .verify import EOS, generate, prompt_ids, split_at_placeholder

DEFAULT_OUTPUT = "./canary-qwen.aimodel"
# mel-frame windows (100/s; encoder outputs T/8 audio tokens). Trained max 40 s.
DEFAULT_WINDOWS = [1000, 2000, 4000]
FMT_BUCKETS = [256, 640]
EMBED_N = 256


def _asr_bucket(mel_frames: int, overhead: int) -> int:
    p = mel_frames // 8 + overhead
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


def _dump_token_table(nemo_tokenizer, out_dir: Path, vocab_size: int) -> None:
    hf_tok = getattr(nemo_tokenizer, "tokenizer", nemo_tokenizer)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="canary-convert")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    parser.add_argument("--quantize", choices=["none", "palette4", "palette6"], default="none")
    parser.add_argument("--check-wav", default=None)
    args = parser.parse_args(argv)
    output_path = Path(args.output)
    windows = sorted({int(w) for w in args.windows.split(",")})

    model = load_model()
    tag_id = model.tokenizer.token_to_id(model.audio_locator_tag)
    asr_ids = prompt_ids(model, f"Transcribe the following: {model.audio_locator_tag}")
    fmt_ids = prompt_ids(model, FORMAT_PROMPT + model.audio_locator_tag)
    asr_pre, asr_post = split_at_placeholder(asr_ids, tag_id)
    fmt_pre, fmt_post = split_at_placeholder(fmt_ids, tag_id)
    overhead = len(asr_pre) + len(asr_post)

    mods = extract_modules(model)
    nemo_tok = mods["tokenizer"]
    vocab_size = mods["llm_asr"].config.vocab_size

    enc = CanaryEncoderForCoreAI(mods["encoder"], mods["proj"]).eval()
    llm_asr, llm_plain = mods["llm_asr"], mods["llm_plain"]
    if args.quantize != "none":
        enc = _palettize(
            enc,
            (torch.zeros(1, 128, windows[-1]), torch.tensor([windows[-1]], dtype=torch.int32)),
            args.quantize,
        )
        print(f"palettized encoder ({args.quantize})")
        ex_ids = torch.zeros(1, 8, dtype=torch.int64)
        llm_asr = _palettize(llm_asr, (ex_ids,), args.quantize)
        print(f"palettized llm_asr ({args.quantize})")
        llm_plain = _palettize(llm_plain, (ex_ids,), args.quantize)
        print(f"palettized llm_plain ({args.quantize})")

        if args.check_wav:
            import soundfile as sf

            audio, sr = sf.read(args.check_wav, dtype="float32")
            sig = torch.from_numpy(audio).unsqueeze(0)
            with torch.no_grad():
                mel, mel_len = mods["preprocessor"](
                    input_signal=sig, length=torch.tensor([sig.shape[1]]))
                audio_embeds = enc(mel, mel_len.to(torch.int32))
            embed = QwenEmbedForCoreAI(llm_asr).eval()
            with torch.no_grad():
                pe = embed(torch.tensor([asr_pre], dtype=torch.int32))
                po = embed(torch.tensor([asr_post], dtype=torch.int32))
            pemb = torch.cat([pe, audio_embeds, po], dim=1)
            P = pemb.shape[1]
            bucket = _asr_bucket(mel.shape[2], overhead)
            pf = QwenPrefillForCoreAI(llm_asr, bucket).eval()
            st = QwenStepForCoreAI(llm_asr).eval()
            st.k_cache, st.v_cache = pf.k_cache, pf.v_cache
            ids = generate(pf, st, pemb, P, bucket)
            text = nemo_tok.ids_to_text(torch.tensor(ids))
            print(f"post-palettize ASR self-check: {text!r}")
            assert len(text.split()) >= 3, "palettized ASR collapsed — aborting"

    converter = coreai_torch.TorchConverter()
    manifest_windows = []
    for w in windows:
        wrapper = CanaryEncoderForCoreAI(mods["encoder"], mods["proj"]).eval()
        wrapper.inner, wrapper.proj = enc.inner, enc.proj  # shared weights
        converter.add_exported_program(
            export_encoder(wrapper, w),
            entrypoint_name=f"encode_{w}",
            input_names=["mel", "mel_length"],
            output_names=["audio_embeds"],
        )
        bucket = _asr_bucket(w, overhead)
        manifest_windows.append(
            {"mel_frames": w, "audio_len": w // 8, "asr_bucket": bucket})
        print(f"exported encode_{w} (asr bucket {bucket})")

    converter.add_exported_program(
        export_embed(QwenEmbedForCoreAI(llm_asr).eval(), EMBED_N),
        entrypoint_name=f"embed_{EMBED_N}",
        input_names=["ids"],
        output_names=["embeds"],
    )
    print(f"exported embed_{EMBED_N}")

    for variant, llm in (("asr", llm_asr), ("llm", llm_plain)):
        buckets = sorted({m["asr_bucket"] for m in manifest_windows}) if variant == "asr" else FMT_BUCKETS
        for b in buckets:
            converter.add_exported_program(
                export_prefill(QwenPrefillForCoreAI(llm, b).eval()),
                entrypoint_name=f"prefill_{variant}_{b}",
                input_names=["embeds", "last_idx"],
                output_names=["token", "k_out", "v_out"],
            )
            print(f"exported prefill_{variant}_{b}")
        converter.add_exported_program(
            export_step(QwenStepForCoreAI(llm).eval()),
            entrypoint_name=f"step_{variant}",
            input_names=["tok", "pos"],
            state_names=["k_cache", "v_cache"],
            output_names=["token"],
        )
        print(f"exported step_{variant}")

    ai_program = converter.to_coreai()
    ai_program.optimize()
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)
        print(f"Replaced existing {output_path}")
    ai_program.save_asset(output_path)

    (output_path / "canary.json").write_text(
        json.dumps(
            {
                "format": 1,
                "model_id": MODEL_ID,
                "eos_id": EOS,
                "max_seq": MAX_SEQ,
                "embed_n": EMBED_N,
                "asr_pre_ids": asr_pre,
                "asr_post_ids": asr_post,
                "fmt_pre_ids": fmt_pre,
                "fmt_post_ids": fmt_post,
                "fmt_buckets": FMT_BUCKETS,
                "windows": manifest_windows,
            },
            indent=1,
        )
    )
    _dump_token_table(nemo_tok, output_path, vocab_size)
    print(f"Wrote {output_path} (quantize={args.quantize}) + canary.json + tokens.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
