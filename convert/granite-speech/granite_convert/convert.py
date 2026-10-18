"""Granite Speech 4.1 2B NAR -> single Core AI .aimodel.

Functions:
  encode_<F>     : audio window of F encoder frames (F*320 samples at 16 kHz)
                   -> (audio_embeds [1, F/5, 2048], ctc_ids [1, F/4] i32)
  edit_<A>_<U>   : one bidirectional LLM pass -> text ids [1, U] i32
                   (A = F/5 pairs with its window; U = text bucket)

Weight sharing: the encoder/projector and the LLM are palettized ONCE (on the
largest-window wrapper) and grafted into every other window's wrapper, so the
multi-window asset carries one copy of each weight set — same pattern as the
Nemotron/Parakeet converters. After palettization, a full-pipeline transcript
self-check runs in pure torch BEFORE the (slow) export, aborting on quality
collapse.

Also bundles inside the asset dir (written before first load, so the
specialize cache keys on the final contents):
  granite.json  — blank id, min edit length, window/bucket table
  tokens.json   — id -> base64 raw-byte table for host-side BPE detokenize

Usage:
  python -m granite_convert.convert --output granite-nar.aimodel \
      [--windows 200,400,800,1600] [--quantize none|palette4|palette6] \
      [--check-wav path.wav]
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import torch

import coreai_torch

from .export import export_edit, export_encoder
from .models import SAMPLES_PER_FRAME, GraniteEditForCoreAI, GraniteEncoderForCoreAI
from .reference import MODEL_ID, load_model

DEFAULT_OUTPUT = "./granite-nar.aimodel"
# window frames (50 Hz, mult of 200) -> text-id bucket. Buckets sized for
# dense speech (~4 tok/s -> 2n+1) with 2x headroom; host clamps beyond.
DEFAULT_WINDOWS = {200: 64, 400: 96, 800: 160, 1600: 320}


def _bucket_for(w: int) -> int:
    return DEFAULT_WINDOWS.get(w, max(64, w // 5))


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte<->unicode table (mirrors tokenizers' ByteLevel)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _dump_token_table(processor, out_dir: Path, vocab_size: int) -> None:
    tok = processor.tokenizer
    uni_to_byte = {u: b for b, u in _bytes_to_unicode().items()}
    special_ids = set(tok.all_special_ids)
    table: list[str] = []
    for i in range(vocab_size):
        if i in special_ids:
            table.append("")  # specials never appear post-collapse; decode to nothing
            continue
        s = tok.convert_ids_to_tokens(i)
        if s is None:
            table.append("")
            continue
        try:
            raw = bytes(uni_to_byte[c] for c in s)
        except KeyError:  # added non-byte-level token: use its literal text
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


def _self_check(enc, edit, wav_path: str, blank: int, min_len: int, processor) -> str:
    """Run the full pipeline (torch, host glue included) and return the text."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == 16000
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    samples = enc.frames * SAMPLES_PER_FRAME
    assert len(audio) <= samples
    padded = np.zeros(samples, dtype=np.float32)
    padded[: len(audio)] = audio
    with torch.no_grad():
        audio_embeds, ctc_ids = enc(torch.from_numpy(padded).unsqueeze(0))
    u = torch.unique_consecutive(ctc_ids[0])
    hyp = u[u != blank]
    total = max(2 * hyp.numel() + 1, min_len)
    assert total <= edit.text_len, f"hypothesis {total} > bucket {edit.text_len}"
    slotted = torch.full((edit.text_len,), blank, dtype=torch.int32)
    slotted[2 * torch.arange(hyp.numel()) + 1] = hyp.to(torch.int32)
    a = audio_embeds.shape[1]
    mask = torch.zeros(1, a + edit.text_len)
    mask[0, : a + total] = 1.0
    with torch.no_grad():
        out_ids = edit(audio_embeds, slotted.unsqueeze(0), mask)
    v = torch.unique_consecutive(out_ids[0, :total].to(torch.int64))
    return processor.batch_decode([v[v != blank]])[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="granite-convert")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    parser.add_argument("--quantize", choices=["none", "palette4", "palette6"], default="none")
    parser.add_argument("--check-wav", default=None, help="post-palettize transcript self-check")
    args = parser.parse_args(argv)
    output_path = Path(args.output)
    windows = sorted({int(w) for w in args.windows.split(",")})
    for w in windows:
        assert w % 200 == 0, f"window {w} not a multiple of the 200-frame attention block"

    processor, model = load_model()
    cfg = model.config
    blank = cfg.blank_token_id
    largest = windows[-1]

    # Build the largest-window wrappers, palettize ONCE, graft everywhere.
    enc_big = GraniteEncoderForCoreAI(model, largest).eval()
    edit_big = GraniteEditForCoreAI(model, largest // 5, _bucket_for(largest)).eval()
    if args.quantize != "none":
        enc_big = _palettize(
            enc_big, (torch.zeros(1, largest * SAMPLES_PER_FRAME),), args.quantize
        )
        print(f"palettized encoder+projector ({args.quantize})")
        a, u = largest // 5, _bucket_for(largest)
        edit_big = _palettize(
            edit_big,
            (
                torch.zeros(1, a, 2048),
                torch.zeros(1, u, dtype=torch.int32),
                torch.ones(1, a + u),
            ),
            args.quantize,
        )
        print(f"palettized LLM ({args.quantize})")
        if args.check_wav:
            text = _self_check(
                enc_big, edit_big, args.check_wav, blank, cfg.min_edit_sequence_length, processor
            )
            print(f"post-palettize self-check transcript: {text!r}")
            assert len(text.split()) >= 3, "palettized pipeline collapsed — aborting export"

    converter = coreai_torch.TorchConverter()
    manifest_windows = []
    for w in windows:
        bucket = _bucket_for(w)
        if w == largest:
            enc = enc_big
        else:
            enc = GraniteEncoderForCoreAI(model, w).eval()
            enc.encoder = enc_big.encoder      # shared (possibly palettized) weights
            enc.projector = enc_big.projector
        converter.add_exported_program(
            export_encoder(enc),
            entrypoint_name=f"encode_{w}",
            input_names=["audio"],
            output_names=["audio_embeds", "ctc_ids"],
        )
        print(f"exported encode_{w}")

        a = w // 5
        if w == largest:
            edit = edit_big
        else:
            edit = GraniteEditForCoreAI(model, a, bucket).eval()
            edit.lm = edit_big.lm
        converter.add_exported_program(
            export_edit(edit),
            entrypoint_name=f"edit_{a}_{bucket}",
            input_names=["audio_embeds", "text_ids", "attn_mask"],
            output_names=["ids"],
        )
        print(f"exported edit_{a}_{bucket}")
        manifest_windows.append({"frames": w, "audio_len": a, "text_bucket": bucket})

    ai_program = converter.to_coreai()
    if hasattr(ai_program, "optimize"):  # coreai-core < 1.0.0b3; b3+ optimizes on build
        ai_program.optimize()
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)
        print(f"Replaced existing {output_path}")
    ai_program.save_asset(output_path)

    (output_path / "granite.json").write_text(
        json.dumps(
            {
                "format": 1,
                "model_id": MODEL_ID,
                "blank_id": blank,
                "min_edit_sequence_length": cfg.min_edit_sequence_length,
                "windows": manifest_windows,
            },
            indent=1,
        )
    )
    _dump_token_table(processor, output_path, cfg.text_config.vocab_size)
    print(f"Wrote {output_path} (quantize={args.quantize}) + granite.json + tokens.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
