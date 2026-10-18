"""Micro-op probes: export one tiny graph per suspicious pattern and compare
runtime output against torch on identical random inputs. Pinpoints exactly
which lowering is wrong without 25-minute full-model cycles.

Usage:
  python -m canary_convert.op_probe --dir <scratch>
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

import coreai_torch

H, KV, REP, T, S, D = 16, 8, 2, 12, 32, 64  # small analogues of the real dims


class RepeatInterleave(nn.Module):
    def forward(self, k: torch.Tensor) -> torch.Tensor:  # [1,KV,S,D]
        return k.repeat_interleave(REP, dim=1)


class GQAReshape5D(nn.Module):
    """The 5-D reshape-grouped attention core (round-1/2 attend)."""

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        qg = q.reshape(1, KV, REP * T, D)
        scores = torch.matmul(qg, k.transpose(-1, -2))
        return scores.reshape(1, KV, REP, T, S)


class RopeBroadcast(nn.Module):
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = D // 2
        x1, x2 = x[..., :half], x[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin  # [1,H,T,D] * [T,D]


class MaskedSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        causal = torch.full((T, T), torch.finfo(torch.float32).min)
        self.register_buffer("mask", torch.triu(causal, diagonal=1).view(1, 1, T, T))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.softmax((scores + self.mask).float(), dim=-1)


class OneHotExtract(nn.Module):
    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:  # [1,T,D], [1]
        onehot = (torch.arange(T, dtype=torch.int32) == idx.reshape(1)).to(x.dtype)
        return torch.matmul(onehot.view(1, 1, T), x).squeeze(1)


class SliceAssignState(nn.Module):
    """Round-1 per-layer slice-assign cache write, tiny scale."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(4, 1, KV, S, D))

    def forward(self, k: torch.Tensor) -> torch.Tensor:  # [4,KV,T,D]
        for li in range(4):
            self.cache[li, :, :, :T] = k[li]
        return self.cache.sum(dim=(0, 1, 2, 4))  # [S] row-mass fingerprint


class RmsNormCast(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.randn(D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
        return v * self.w.float()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args(argv)
    scratch = Path(args.dir)
    scratch.mkdir(parents=True, exist_ok=True)

    g = torch.Generator().manual_seed(7)
    q = torch.randn(1, H, T, D, generator=g)
    k = torch.randn(1, KV, S, D, generator=g)
    x_htd = torch.randn(1, H, T, D, generator=g)
    cos = torch.randn(T, D, generator=g)
    sin = torch.randn(T, D, generator=g)
    scores = torch.randn(1, H, T, T, generator=g)
    xtd = torch.randn(1, T, D, generator=g)
    idx = torch.tensor([T - 3], dtype=torch.int32)
    k4 = torch.randn(4, KV, T, D, generator=g)
    xrms = torch.randn(1, T, D, generator=g)

    cases = [
        ("repeat_interleave", RepeatInterleave(), (k,)),
        ("gqa_reshape5d", GQAReshape5D(), (q, k)),
        ("rope_broadcast", RopeBroadcast(), (x_htd, cos, sin)),
        ("masked_softmax", MaskedSoftmax(), (scores,)),
        ("onehot_extract", OneHotExtract(), (xtd, idx)),
        ("rmsnorm_cast", RmsNormCast(), (xrms,)),
    ]
    converter = coreai_torch.TorchConverter()
    refs = {}
    state_case = ("slice_assign_state", SliceAssignState(), (k4,))
    for name, mod, inputs in cases + [state_case]:
        mod = mod.eval()
        with torch.no_grad():
            refs[name] = mod(*inputs)
        ep = torch.export.export(mod, args=inputs).run_decompositions(
            coreai_torch.get_decomp_table())
        in_names = [f"i{j}" for j in range(len(inputs))]
        kwargs = {}
        if name == "slice_assign_state":
            kwargs["state_names"] = ["cache"]
        converter.add_exported_program(
            ep, entrypoint_name=name, input_names=in_names, output_names=["out"], **kwargs)
        print(f"exported {name}")

    prog = converter.to_coreai()
    if hasattr(prog, "optimize"):  # coreai-core < 1.0.0b3; b3+ optimizes on build
        prog.optimize()
    out_path = scratch / "op-probe.aimodel"
    if out_path.exists():
        shutil.rmtree(out_path)
    prog.save_asset(out_path)

    async def run():
        from coreai import runtime as rt

        m = await rt.AIModel.load(str(out_path))
        for name, _mod, inputs in cases:
            fn = m.load_function(name)
            feed = {f"i{j}": rt.NDArray(t.numpy()) for j, t in enumerate(inputs)}
            got = (await fn(feed))["out"].numpy()
            ref = refs[name].numpy()
            d = float(np.abs(got.astype(np.float64) - ref.astype(np.float64)).max())
            rel = d / (float(np.abs(ref).max()) + 1e-9)
            print(f"{name:20s} max|diff|={d:.3e} rel={rel:.3e} {'OK' if rel < 3e-2 else '*** BROKEN ***'}")
        fn = m.load_function("slice_assign_state")
        cache = rt.NDArray(np.zeros((4, 1, KV, S, D), dtype=np.float32))
        got = (await fn({"i0": rt.NDArray(k4.numpy())}, state={"cache": cache}))["out"].numpy()
        ref = refs["slice_assign_state"].numpy()
        d = float(np.abs(got - ref).max())
        rel = d / (float(np.abs(ref).max()) + 1e-9)
        print(f"{'slice_assign_state':20s} max|diff|={d:.3e} rel={rel:.3e} {'OK' if rel < 3e-2 else '*** BROKEN ***'}")

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
