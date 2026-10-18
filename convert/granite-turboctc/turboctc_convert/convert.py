"""Granite Speech 5.0 TurboCTC -> single Core AI .aimodel.

Functions:
  ctc_<F> : (audio [1, F*320] f32, length [1] i32) -> ids [1, F/4] i32
            (F stacked 50 Hz frames; ids are per-frame argmax, blank 0 past
            the real length)

Weight sharing: the encoder + CTC head are palettized ONCE (on the
largest-window wrapper) and grafted into every other window's wrapper, so the
multi-window asset carries one copy of the weights — same pattern as the
Granite NAR / Nemotron / Parakeet converters. After palettization, a
transcript self-check runs in pure torch BEFORE the export, aborting on
quality collapse.

Also bundles inside the asset dir (written before first load, so the
specialize cache keys on the final contents):
  turboctc.json — blank id + window table
  tokens.json   — id -> base64 raw-byte table for host-side BPE detokenize

Usage:
  python -m turboctc_convert.convert --output granite-speech-5.0-470m-turboctc.aimodel \
      [--windows 256,512,1024,1536] [--quantize none|palette4|palette6] \
      [--check-wav path.wav]
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import torch

import coreai_torch

from .export import export_ctc
from .models import BLANK_ID, SAMPLES_PER_FRAME, SUBSAMPLE, TurboCTCForCoreAI
from .reference import MODEL_ID, load_model, reference_transcribe

DEFAULT_OUTPUT = "./granite-speech-5.0-470m-turboctc.aimodel"
# Stacked 50 Hz frames: ~5 s, ~10 s, ~20 s, ~30 s (multiples of 4).
DEFAULT_WINDOWS = [256, 512, 1024, 1536]


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
    special_ids = set(tok.all_special_ids) | {BLANK_ID}
    table: list[str] = []
    for i in range(vocab_size):
        if i in special_ids:
            table.append("")  # blank/specials never appear post-collapse
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


def transcribe_with(wrapper: TurboCTCForCoreAI, audio, processor) -> str:
    """Run one window wrapper (torch) plus the host glue; returns text."""
    import numpy as np

    samples = wrapper.frames * SAMPLES_PER_FRAME
    assert len(audio) <= samples, f"clip longer than the {wrapper.frames}-frame window"
    padded = np.zeros(samples, dtype=np.float32)
    padded[: len(audio)] = audio
    with torch.no_grad():
        ids = wrapper(torch.from_numpy(padded).unsqueeze(0), torch.tensor([len(audio)], dtype=torch.int32))
    u = torch.unique_consecutive(ids[0])
    return processor.batch_decode([u[u != BLANK_ID].tolist()], skip_special_tokens=True)[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="turboctc-convert")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    parser.add_argument("--quantize", choices=["none", "palette4", "palette6"], default="none")
    parser.add_argument("--check-wav", default=None, help="post-palettize transcript self-check")
    args = parser.parse_args(argv)
    output_path = Path(args.output)
    windows = sorted({int(w) for w in args.windows.split(",")})
    for w in windows:
        assert w % SUBSAMPLE == 0, f"window {w} not a multiple of the {SUBSAMPLE}x subsampling"

    processor, model = load_model()
    largest = windows[-1]

    big = TurboCTCForCoreAI(model, largest).eval()
    if args.quantize != "none":
        samples = largest * SAMPLES_PER_FRAME
        big = _palettize(
            big,
            (torch.zeros(1, samples), torch.tensor([samples], dtype=torch.int32)),
            args.quantize,
        )
        print(f"palettized encoder + CTC head ({args.quantize})")
        if args.check_wav:
            import soundfile as sf

            audio, sr = sf.read(args.check_wav, dtype="float32")
            assert sr == 16000
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            text = transcribe_with(big, audio, processor)
            ref = reference_transcribe(processor, model, audio)
            print(f"post-palettize self-check transcript: {text!r}")
            print(f"fp32 reference transcript:            {ref!r}")
            assert len(text.split()) >= 3, "palettized pipeline collapsed — aborting export"

    converter = coreai_torch.TorchConverter()
    manifest_windows = []
    for w in windows:
        if w == largest:
            wrapper = big
        else:
            wrapper = TurboCTCForCoreAI(model, w).eval()
            wrapper.encoder = big.encoder      # shared (possibly palettized) weights
            wrapper.ctc_head = big.ctc_head
        converter.add_exported_program(
            export_ctc(wrapper),
            entrypoint_name=f"ctc_{w}",
            input_names=["audio", "length"],
            output_names=["ids"],
        )
        print(f"exported ctc_{w}")
        manifest_windows.append({"frames": w, "out_len": w // SUBSAMPLE})

    ai_program = converter.to_coreai()
    if hasattr(ai_program, "optimize"):  # coreai-core < 1.0.0b3; b3+ optimizes on build
        ai_program.optimize()
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)
        print(f"Replaced existing {output_path}")
    ai_program.save_asset(output_path)

    (output_path / "turboctc.json").write_text(
        json.dumps(
            {
                "format": 1,
                "model_id": MODEL_ID,
                "blank_id": BLANK_ID,
                "samples_per_frame": SAMPLES_PER_FRAME,
                "windows": manifest_windows,
            },
            indent=1,
        )
    )
    _dump_token_table(processor, output_path, model.config.vocab_size)
    print(f"Wrote {output_path} (quantize={args.quantize}) + turboctc.json + tokens.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
