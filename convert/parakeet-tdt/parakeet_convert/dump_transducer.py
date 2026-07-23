"""Dump the tiny transducer-decode weights (prediction LSTM + joint) as a
fp32 sidecar for the Swift CPU decoder.

The Core AI dispatch floor (~0.4 ms/call) dominates transducer decode — ~90
tiny dispatches per 5.5 s clip. The weights are ~8 M params, so the Swift
runtime can run predict/join on CPU (Accelerate) in microseconds per step;
this tool extracts them.

Writes, SIBLING to the asset (never inside it — the specialize cache keys on
the asset dir contents): `<asset>.transducer.json` (manifest) and
`<asset>.transducer.bin` (raw little-endian float32, concatenated in
manifest order).

Usage:
  python -m parakeet_convert.dump_transducer --asset /path/to/model.aimodel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parakeet-dump-transducer")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--asset", required=True, help="Path to the .aimodel this sidecar accompanies")
    args = parser.parse_args(argv)

    import nemo.collections.asr as nemo_asr

    m = nemo_asr.models.ASRModel.from_pretrained(args.model_id)
    lstm = m.decoder.prediction["dec_rnn"].lstm
    # Vocab-dependent fields come from the checkpoint (v3: 8192, v2: 1024).
    vocab = m.tokenizer.vocab_size
    joint_out = m.joint.joint_net[2].out_features
    assert joint_out == vocab + 1 + 5, (
        f"joint out dim {joint_out} != vocab {vocab} + 1 + 5 TDT durations"
    )
    tensors: list[tuple[str, torch.Tensor]] = [
        ("embed.weight", m.decoder.prediction["embed"].weight),          # [vocab+1, 640]
        ("lstm.l0.w_ih", lstm.weight_ih_l0), ("lstm.l0.w_hh", lstm.weight_hh_l0),
        ("lstm.l0.b_ih", lstm.bias_ih_l0), ("lstm.l0.b_hh", lstm.bias_hh_l0),
        ("lstm.l1.w_ih", lstm.weight_ih_l1), ("lstm.l1.w_hh", lstm.weight_hh_l1),
        ("lstm.l1.b_ih", lstm.bias_ih_l1), ("lstm.l1.b_hh", lstm.bias_hh_l1),
        ("joint.pred.weight", m.joint.pred.weight), ("joint.pred.bias", m.joint.pred.bias),
        ("joint.out.weight", m.joint.joint_net[2].weight),               # [vocab+6, 640]
        ("joint.out.bias", m.joint.joint_net[2].bias),
    ]

    asset = Path(args.asset)
    bin_path = asset.with_name(asset.name + ".transducer.bin")
    json_path = asset.with_name(asset.name + ".transducer.json")

    manifest: dict = {
        "format": 1,
        "model_id": args.model_id,
        "blank_id": vocab,
        "token_logits": vocab + 1,
        "durations": [0, 1, 2, 3, 4],
        "pred_hidden": 640,
        "joint_hidden": 640,
        "lstm_layers": 2,
        "tensors": [],
    }
    offset = 0
    with open(bin_path, "wb") as f:
        for name, t in tensors:
            arr = t.detach().float().contiguous().cpu().numpy()
            f.write(arr.tobytes())
            manifest["tensors"].append(
                {"name": name, "shape": list(arr.shape), "offset": offset, "count": int(arr.size)}
            )
            offset += arr.size * 4
    json_path.write_text(json.dumps(manifest, indent=1))
    print(f"Wrote {bin_path} ({offset / 1e6:.1f} MB) + {json_path.name} ({len(tensors)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
