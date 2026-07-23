"""Dump Nemotron's transducer-decode weights (prediction LSTM + full joint,
including the encoder projection) as a fp32 sidecar for the Swift CPU decoder.

Same format as parakeet-tdt's dump_transducer (format 1), plus:
  - `enc_in`: the joint's raw encoder-step width (1024) with tensors
    joint.enc.weight/bias — Nemotron has no in-graph project_enc, so the
    CPU decoder projects raw encoder steps itself (a 640x1024 sgemv/frame).
  - `durations: []` — plain RNN-T (no TDT duration head).

Writes `<asset>.transducer.json/.bin` SIBLING to the asset.

Usage:
  python -m nemotron_convert.dump_transducer --asset /path/to/model.aimodel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nemotron-dump-transducer")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--asset", required=True)
    args = parser.parse_args(argv)

    import nemo.collections.asr as nemo_asr

    m = nemo_asr.models.ASRModel.from_pretrained(args.model_id)
    lstm = m.decoder.prediction["dec_rnn"].lstm
    tensors = [
        ("embed.weight", m.decoder.prediction["embed"].weight),          # [1025, 640]
        ("lstm.l0.w_ih", lstm.weight_ih_l0), ("lstm.l0.w_hh", lstm.weight_hh_l0),
        ("lstm.l0.b_ih", lstm.bias_ih_l0), ("lstm.l0.b_hh", lstm.bias_hh_l0),
        ("lstm.l1.w_ih", lstm.weight_ih_l1), ("lstm.l1.w_hh", lstm.weight_hh_l1),
        ("lstm.l1.b_ih", lstm.bias_ih_l1), ("lstm.l1.b_hh", lstm.bias_hh_l1),
        ("joint.enc.weight", m.joint.enc.weight), ("joint.enc.bias", m.joint.enc.bias),
        ("joint.pred.weight", m.joint.pred.weight), ("joint.pred.bias", m.joint.pred.bias),
        ("joint.out.weight", m.joint.joint_net[2].weight),               # [1025, 640]
        ("joint.out.bias", m.joint.joint_net[2].bias),
    ]

    asset = Path(args.asset)
    bin_path = asset.with_name(asset.name + ".transducer.bin")
    json_path = asset.with_name(asset.name + ".transducer.json")

    manifest: dict = {
        "format": 1,
        "model_id": args.model_id,
        "blank_id": 1024,
        "token_logits": 1025,   # full head: 1024 SP tokens + blank; no duration tail
        "durations": [],        # plain RNN-T
        "pred_hidden": 640,
        "joint_hidden": 640,
        "enc_in": 1024,
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
