"""Specializer-crash bisect for the Qwen3 prefill/step graphs.

Palettizes the ASR LLM once (in-process), exports several graph VARIANTS to
scratch assets, then load-tests each in a SUBPROCESS (segfault-isolated) and
prints a per-variant verdict: does rt.AIModel.load + one prefill call
survive, and does it return the right first token (785)?

Variants:
  A: round-2 graphs as-is        (buffer copy_ states, MAX_SEQ=1024)
  B: same graphs, MAX_SEQ=448    (smaller states)

Usage:
  python -m canary_convert.bisect --dir <scratch>
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import coreai_torch

from . import models as M
from .convert import _palettize
from .export import export_prefill, export_step
from .extract import extract_modules
from .models import QwenPrefillForCoreAI, QwenStepForCoreAI
from .reference import load_model

BUCKET = 192

HYBRID_TEST = r"""
import asyncio, sys
import numpy as np

async def main():
    from coreai import runtime as rt
    asset, embp, lastidx, kp, vp, pos0, tok0 = sys.argv[1:8]
    m = await rt.AIModel.load(asset)
    pf = m.load_function("prefill_fn")
    out = await pf({"embeds": rt.NDArray(np.load(embp)),
                    "last_idx": rt.NDArray(np.array([int(lastidx)], dtype=np.int32))})
    tok = int(out["token"].numpy().ravel()[0])
    k_out = out["k_out"].numpy()
    k_ref = np.load(kp)[:, 0, :, :k_out.shape[2], :]
    num = float((k_out.astype(np.float64) * k_ref.astype(np.float64)).sum())
    den = float(np.linalg.norm(k_out) * np.linalg.norm(k_ref)) + 1e-9
    print(f"PREFILL_OK token={tok} kcos={num/den:.4f}")

    # stateful step fed CORRECT torch caches
    k = np.load(kp).astype(np.float32)
    v = np.load(vp).astype(np.float32)
    st = m.load_function("step_st")
    out = await st({"tok": rt.NDArray(np.array([int(tok0)], dtype=np.int32)),
                    "pos": rt.NDArray(np.array([int(pos0)], dtype=np.int32))},
                   state={"k_cache": rt.NDArray(k), "v_cache": rt.NDArray(v)})
    print(f"STEP_OK token={int(out['token'].numpy().ravel()[0])}")

asyncio.run(main())
"""

NOWRITE_TEST = r"""
import asyncio, sys
import numpy as np

async def main():
    from coreai import runtime as rt
    m = await rt.AIModel.load(sys.argv[1])
    pf = m.load_function("prefill_nw")
    embeds = np.load(sys.argv[2])
    out = await pf({"embeds": rt.NDArray(embeds),
                    "last_idx": rt.NDArray(np.array([int(sys.argv[3])], dtype=np.int32))})
    print(f"NOWRITE_OK token={int(out['token'].numpy().ravel()[0])}")

asyncio.run(main())
"""

LOAD_TEST = r"""
import asyncio, sys
import numpy as np

async def main():
    from coreai import runtime as rt
    asset, max_seq = sys.argv[1], int(sys.argv[2])
    m = await rt.AIModel.load(asset)
    pf = m.load_function(f"prefill_asr_192")
    embeds = np.load(sys.argv[3])
    k = rt.NDArray(np.zeros((28, 1, 8, max_seq, 128), dtype=np.float32))
    v = rt.NDArray(np.zeros((28, 1, 8, max_seq, 128), dtype=np.float32))
    out = await pf({"embeds": rt.NDArray(embeds),
                    "last_idx": rt.NDArray(np.array([int(sys.argv[4])], dtype=np.int32))},
                   state={"k_cache": k, "v_cache": v})
    tok = int(out["token"].numpy().ravel()[0])
    print(f"LOADED_OK token={tok}")
    st = m.load_function("step_asr")
    out = await st({"tok": rt.NDArray(np.array([tok], dtype=np.int32)),
                    "pos": rt.NDArray(np.array([int(sys.argv[4]) + 1], dtype=np.int32))},
                   state={"k_cache": k, "v_cache": v})
    print(f"STEP_OK token={int(out['token'].numpy().ravel()[0])}")

asyncio.run(main())
"""


def build_asset(llm, out: Path, PF=QwenPrefillForCoreAI, ST=QwenStepForCoreAI) -> None:
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        export_prefill(PF(llm, BUCKET).eval()),
        entrypoint_name=f"prefill_asr_{BUCKET}",
        input_names=["embeds", "last_idx"],
        state_names=["k_cache", "v_cache"],
        output_names=["token"],
    )
    converter.add_exported_program(
        export_step(ST(llm).eval()),
        entrypoint_name="step_asr",
        input_names=["tok", "pos"],
        state_names=["k_cache", "v_cache"],
        output_names=["token"],
    )
    prog = converter.to_coreai()
    prog.optimize()
    if out.exists():
        shutil.rmtree(out)
    prog.save_asset(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--wav", default="fixtures/sample.wav")
    args = parser.parse_args(argv)
    scratch = Path(args.dir)
    scratch.mkdir(parents=True, exist_ok=True)

    model = load_model()
    from .verify import prompt_ids, split_at_placeholder

    tag_id = model.tokenizer.token_to_id(model.audio_locator_tag)
    asr_ids = prompt_ids(model, f"Transcribe the following: {model.audio_locator_tag}")
    pre, post = split_at_placeholder(asr_ids, tag_id)
    mods = extract_modules(model)

    # Prompt embeds (fp32 torch reference — enough for a token check).
    import soundfile as sf

    from .models import CanaryEncoderForCoreAI, QwenEmbedForCoreAI

    audio, _sr = sf.read(args.wav, dtype="float32")
    sig = torch.from_numpy(audio).unsqueeze(0)
    with torch.no_grad():
        mel, mel_len = mods["preprocessor"](input_signal=sig, length=torch.tensor([sig.shape[1]]))
        mel_pad = torch.zeros(1, 128, 1000)
        mel_pad[:, :, : mel.shape[2]] = mel
        enc = CanaryEncoderForCoreAI(mods["encoder"], mods["proj"]).eval()
        ref_embeds = enc(mel_pad, mel_len.to(torch.int32))
        embed = QwenEmbedForCoreAI(mods["llm_asr"]).eval()
        pre_e = embed(torch.tensor([pre], dtype=torch.int32))
        post_e = embed(torch.tensor([post], dtype=torch.int32))
    a_real = (int(mel_len[0]) + 7) // 8
    prompt = torch.cat([pre_e, ref_embeds[:, :a_real], post_e], dim=1)
    P = prompt.shape[1]
    padded = torch.zeros(1, BUCKET, 2048)
    padded[:, :P] = prompt
    emb_path = scratch / "prompt_embeds.npy"
    np.save(emb_path, padded.numpy())

    llm = _palettize(mods["llm_asr"], (torch.zeros(1, 8, dtype=torch.int64),), "palette4")
    print("palettized llm_asr", flush=True)

    from . import models_v1 as MV1

    # Round G: functional prefill (caches as OUTPUTS) + stateful step
    # (round-1 one-hot write) fed CORRECT torch caches.
    import torch as _t

    # torch-side reference caches for the step test
    pf_t = MV1.QwenPrefillForCoreAI(mods["llm_asr"], BUCKET).eval()
    st_t = MV1.QwenStepForCoreAI(mods["llm_asr"]).eval()
    st_t.k_cache = pf_t.k_cache
    st_t.v_cache = pf_t.v_cache
    with _t.no_grad():
        tok0 = int(pf_t(padded, _t.tensor([P - 1], dtype=_t.int32)).item())
        tok1 = int(st_t(_t.tensor([tok0], dtype=_t.int32), _t.tensor([P], dtype=_t.int32)).item())
    np.save(scratch / "k_ref.npy", pf_t.k_cache.detach().numpy())
    np.save(scratch / "v_ref.npy", pf_t.v_cache.detach().numpy())
    print(f"torch (palettized): tok0={tok0} tok1={tok1}", flush=True)

    converter = coreai_torch.TorchConverter()
    ep = _t.export.export(
        MV1.QwenPrefillFunctional(llm, BUCKET).eval(),
        args=(_t.zeros(1, BUCKET, 2048), _t.tensor([BUCKET - 1], dtype=_t.int32)),
    ).run_decompositions(coreai_torch.get_decomp_table())
    converter.add_exported_program(
        ep, entrypoint_name="prefill_fn", input_names=["embeds", "last_idx"],
        output_names=["token", "k_out", "v_out"])
    ep = _t.export.export(
        MV1.QwenStepForCoreAI(llm).eval(),
        args=(_t.zeros(1, dtype=_t.int32), _t.tensor([1023], dtype=_t.int32)),
    ).run_decompositions(coreai_torch.get_decomp_table())
    converter.add_exported_program(
        ep, entrypoint_name="step_st", input_names=["tok", "pos"],
        state_names=["k_cache", "v_cache"], output_names=["token"])
    prog = converter.to_coreai()
    prog.optimize()
    out = scratch / "canary-G_hybrid.aimodel"
    if out.exists():
        shutil.rmtree(out)
    prog.save_asset(out)
    print("[G_hybrid] exported", flush=True)
    r = subprocess.run(
        [sys.executable, "-c", HYBRID_TEST, str(out), str(emb_path), str(P - 1),
         str(scratch / "k_ref.npy"), str(scratch / "v_ref.npy"), str(P), str(tok0)],
        capture_output=True, text=True, timeout=1800)
    verdict = [l for l in r.stdout.splitlines() if "OK" in l or "cos" in l]
    print(f"[G_hybrid] rc={r.returncode} {verdict or r.stderr.splitlines()[-2:]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
