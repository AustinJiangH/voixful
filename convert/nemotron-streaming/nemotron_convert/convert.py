"""Default: FP16. No coreai_opt invocation. The conditional quantization path runs only when --quantize is set."""

from __future__ import annotations

import argparse
from pathlib import Path

import coreai_torch
import torch

from .export import export_encoder, export_joiner, export_prediction
from .extract import extract_modules_from_nemo
from .models import (
    NemotronEncoderForCoreAI,
    NemotronJoinerForCoreAI,
    NemotronPredictionForCoreAI,
)

# The 40-language `nvidia/nemotron-3.5-asr-streaming-0.6b` checkpoint requires
# NeMo's unreleased `EncDecRNNTBPEModelWithPrompt` class (not in NeMo 2.7.3 on
# PyPI as of 2026-06). The English-only sibling uses the standard
# `EncDecRNNTBPEModel` and converts cleanly. Switch back to the 40-locale
# checkpoint once the upstream class lands in a NeMo release.
DEFAULT_MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
DEFAULT_OUTPUT = "./nemotron-streaming.aimodel"

# The three named functions packed into a single multi-function .aimodel.
ENCODE_FN = "encode"
PREDICT_STEP_FN = "predict_step"
JOIN_FN = "join"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nemotron-convert",
        description=(
            "Convert nvidia/nemotron-speech-streaming-en-0.6b to a single Core AI "
            ".aimodel with three functions (encode, predict_step, join). "
            "FP16 default."
        ),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--windows",
        default="32,72,128,256",
        help=(
            "Comma-separated encoder chunk lengths in mel frames (10 ms hop), "
            "each exported as its own fixed-shape program against the SHARED "
            "streaming caches. 128 is required (the canonical 'encode' — "
            "runtime back-compat contract); others become 'encode_<frames>'. "
            "Small windows are traced under their matching TRAINED att-context "
            "tier (32 -> [70,1] the 160 ms tier; 72 -> [70,6] the 560 ms "
            "tier); tracing a small chunk under the default [70,13] config "
            "produces garbage (valid_out_len bookkeeping assumes 112-mel "
            "chunks — measured WER collapse). 256 == batch-throughput chunk."
        ),
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "int8", "int8-mmap", "palette4", "palette6"],
        default="none",
        help=(
            "Encoder weight compression. 'int8'/'int8-mmap' = "
            "INT8 linear weight-only (disk/bandwidth). 'palette4'/'palette6' = "
            "4/6-bit k-means palettization — the format Apple documents as the "
            "Neural Engine's runtime-memory + latency win (kept compressed, "
            "decompressed just-in-time). Prediction LSTM + joiner stay FP16."
        ),
    )
    args = parser.parse_args(argv)

    output_path = Path(args.output)

    # Build the INT8 weight-only quantizer BEFORE importing NeMo (coreai_opt's
    # pydantic models resolve cleanly only in a NeMo-free import state). Use the
    # official `QuantizerConfig.presets.w8()` recipe — data-free, weight-only,
    # op key 'weight'. See apple/coreai-optimization + WWDC26 session 325.
    quantize_encoder = None
    if args.quantize in ("int8", "int8-mmap"):
        from coreai_opt.quantization import ExecutionMode, Quantizer, QuantizerConfig

        mmap_dir = None
        if args.quantize == "int8-mmap":
            # mmap_dir requires eager mode, which also quantizes conv ops — they
            # need an explicit per-channel axis (axis=0 = output channels) that
            # the graph-mode preset resolves automatically but eager does not.
            from coreai_opt.quantization import (
                ModuleQuantizerConfig,
                QuantizationSpec,
            )
            from coreai_opt.quantization.spec.granularity import PerChannelGranularity

            wspec = QuantizationSpec(
                dtype=torch.int8, granularity=PerChannelGranularity(axis=0)
            )
            mqc = ModuleQuantizerConfig(
                op_input_spec=None,
                op_output_spec=None,
                op_state_spec={"weight": wspec},
                op_type_config=None,
                op_name_config=None,
                module_input_spec=None,
                module_output_spec=None,
                module_state_spec=None,
                qat_schedule=None,
            )
            qcfg = QuantizerConfig(global_config=mqc, execution_mode=ExecutionMode.EAGER)
            mmap_dir = output_path.parent / f"{output_path.name}.weights"
            mmap_dir.mkdir(parents=True, exist_ok=True)
        else:
            qcfg = QuantizerConfig.presets.w8()

        def quantize_encoder(enc):
            # Match export_encoder's fixed-shape example inputs (T=128).
            example_mel = torch.zeros(1, 128, 128, dtype=torch.float32)
            example_length = torch.tensor([128], dtype=torch.int32)
            q = Quantizer(enc, qcfg)
            q.prepare(example_inputs=(example_mel, example_length))
            return q.finalize(mmap_dir=str(mmap_dir)) if mmap_dir else q.finalize()

    elif args.quantize in ("palette4", "palette6"):
        # K-means weight palettization (4/6-bit LUT). Apple's docs flag this as
        # the Neural Engine's runtime-memory format (weights stay low-bit,
        # converted on the fly layer-to-layer).
        from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig

        _preset = {
            "palette4": KMeansPalettizerConfig.presets.w4,
            "palette6": KMeansPalettizerConfig.presets.w6,
        }[args.quantize]
        pcfg = _preset()

        def quantize_encoder(enc):
            example_mel = torch.zeros(1, 128, 128, dtype=torch.float32)
            example_length = torch.tensor([128], dtype=torch.int32)
            p = KMeansPalettizer(enc, pcfg)
            p.prepare(example_inputs=(example_mel, example_length))
            return p.finalize()

    # Step (a): pull the raw nn.Modules out of the NeMo ASRModel.
    modules = extract_modules_from_nemo(args.model_id)

    # Step (b): build the three Core-AI-shaped wrappers.
    # VERIFY: d_model (open question). The wrapper defaults to 1024
    # (matching soniqo/speech-swift's cache shape). Once we inspect
    # modules["config"], pass the resolved value through here rather than relying on
    # the default.
    encoder_wrapper = NemotronEncoderForCoreAI(modules["encoder"])
    prediction_wrapper = NemotronPredictionForCoreAI(modules["prediction"])
    joiner_wrapper = NemotronJoinerForCoreAI(modules["joiner"])

    encoder_wrapper.eval()
    prediction_wrapper.eval()
    joiner_wrapper.eval()

    # Step (b2): apply the INT8 weight-only quantizer to the encoder.
    if quantize_encoder is not None:
        encoder_wrapper = quantize_encoder(encoder_wrapper)
        print("Applied INT8 weight-only quantization to the encoder (presets.w8).")

    # Step (c): torch.export the shared modules; the encoder once per window.
    windows = sorted({int(w) for w in args.windows.split(",")})
    assert 128 in windows, "--windows must include 128 (the canonical 'encode' contract)"
    exported_prediction = export_prediction(prediction_wrapper)
    exported_joiner = export_joiner(joiner_wrapper)

    # Step (d): pack into one multi-function .aimodel.
    converter = coreai_torch.TorchConverter()

    # Per-window attention context: small windows use the checkpoint's TRAINED
    # tier whose chunk (+9-frame pre-encode cache) fits the window; >=128-mel
    # windows keep the checkpoint default [70, 13] (the 1120 ms tier). The
    # streaming caches are left-context-shaped (70) so all tiers share state.
    att_by_window = {32: [70, 1], 72: [70, 6]}
    default_att = list(encoder_wrapper.inner.att_context_size)
    for w in windows:
        name = ENCODE_FN if w == 128 else f"{ENCODE_FN}_{w}"
        att = att_by_window.get(w, default_att)
        encoder_wrapper.inner.set_default_att_context_size(att)
        encoder_wrapper.inner.setup_streaming_params()
        if w > 128:
            # Larger-than-default chunks: keep the trained [70,13] attention
            # masks and 70-frame cache, but fix the chunk bookkeeping so the
            # graph returns ALL of the window's valid frames. Without this the
            # cfg's valid_out_len stays 14 (tuned to 112-mel chunks) and a
            # 256-mel chunk silently loses 16 of its 30 output frames —
            # measured as words dropped at every chunk boundary.
            enc_frames = w // 8
            cfg = encoder_wrapper.inner.streaming_cfg
            cfg.chunk_size = [w - 7, w]
            cfg.shift_size = [w - 7, w]
            cfg.valid_out_len = enc_frames
        print(f"Exporting encoder chunk window {w} frames (att_context={att}) -> function '{name}' ...")
        converter.add_exported_program(
            export_encoder(encoder_wrapper, chunk_frames=w),
            entrypoint_name=name,
            # English-only checkpoint has no language conditioning; input is
            # (mel_chunk, mel_length). The multilingual variant adds a lang_id
            # input — re-add when switching back to the 40-language checkpoint.
            input_names=["mel_chunk", "mel_length"],
            # Three streaming caches per the verified shapes; no `pre_cache` in
            # this checkpoint (see models.py header). All windows SHARE these
            # states — cache shapes derive from the left context, which is
            # window-invariant.
            state_names=[
                "cache_last_channel",
                "cache_last_time",
                "cache_last_channel_len",
            ],
            output_names=["enc_out"],
        )
    converter.add_exported_program(
        exported_prediction,
        entrypoint_name=PREDICT_STEP_FN,
        input_names=["prev_token"],
        state_names=["lstm_h", "lstm_c"],
        output_names=["pred_emb"],
    )
    converter.add_exported_program(
        exported_joiner,
        entrypoint_name=JOIN_FN,
        input_names=["enc_step", "pred_emb"],
        output_names=["logits"],
    )

    # Baked audio->mel preprocessor (batch path). NeMo's own preprocessor
    # doesn't survive torch.export; preproc_matmul re-expresses it with fixed
    # shapes + matmuls, parity-tested against the NeMo featurizer (7e-5).
    # Its presence flips the CLI/engine into the pipelined batch mode
    # (encode(N+1) overlapped with decode(N)).
    from .preproc_matmul import NemotronPreprocessorForCoreAI, audio_samples

    featurizer = modules["preprocessor"].featurizer
    featurizer.dither = 0.0
    # 128-frame canonical + a 256-frame variant pairing with encode_256 so the
    # batch path can halve its encoder invocations.
    for frames, name in [(128, "preprocess_audio"), (256, "preprocess_audio_256")]:
        n = audio_samples(frames)
        preproc_wrapper = NemotronPreprocessorForCoreAI(featurizer, out_frames=frames).eval()
        print(f"Exporting {name} (matmul mel, {n} samples -> [1,128,{frames}]) ...")
        exported_preproc = torch.export.export(
            preproc_wrapper,
            args=(
                torch.zeros(1, n, dtype=torch.float32),
                torch.tensor([n], dtype=torch.int32),
            ),
        ).run_decompositions(coreai_torch.get_decomp_table())
        converter.add_exported_program(
            exported_preproc,
            entrypoint_name=name,
            input_names=["audio", "audio_length"],
            output_names=["mel"],
        )

    # VERIFY: add_exported_program kwargs — confirm the exact keyword name for the
    # entrypoint (could be `name=` or positional) against the coreai_torch API once
    # installed. The WWDC 2026 "Dive into Core AI" session demonstrated multi-function
    # .aimodel packing.

    # Step (e): convert, framework-level optimize (no weight quantization), save.
    # `coreai_opt` is NOT invoked here. FP16 default.
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
    # self-contained (StreamingASR reads <model>/pieces.txt first).
    pieces = list(modules["tokenizer"].vocab)
    # Fail-fast: the English-only Nemotron checkpoint has a 1024-token vocab. If a
    # model/tooling swap changes this, abort at convert time rather than shipping an
    # asset the Swift runtime later rejects (vocab mismatch vs the joiner's 1025-wide
    # logits = 1024 tokens + blank).
    EXPECTED_VOCAB = 1024
    assert len(pieces) == EXPECTED_VOCAB, (
        f"pieces.txt vocab mismatch: got {len(pieces)} pieces, expected "
        f"{EXPECTED_VOCAB} for nvidia/nemotron-speech-streaming-en-0.6b. "
        "A model or tokenizer swap changed the vocab — fix the architecture "
        "constants before shipping this asset."
    )
    (output_path / "pieces.txt").write_text("\n".join(pieces) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}/pieces.txt ({len(pieces)} pieces) — self-contained.")

    print(
        f"Wrote {output_path} — a single Core AI multi-function .aimodel containing "
        f"the three named inference functions '{ENCODE_FN}', '{PREDICT_STEP_FN}', and "
        f"'{JOIN_FN}' at FP16 precision (no coreai_opt quantization applied)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
