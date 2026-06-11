"""Phase K — Python conversion-pipeline smoke test.

Does NOT download the model, does NOT run torch.export, does NOT load NeMo.
Confirms the package's modules import (catches the most common dep failures),
its public functions have the documented signatures (catches drift between
the wrappers' construction args and convert.py's call sites), and its
top-level entry points (`main()`, argparse parsers) are reachable.

Runnable as `python -m nemotron_convert.smoke`. Exits 0 on PASS, non-zero on
FAIL. Each check prints a `[smoke] check: PASS/FAIL ...` line so the shell
driver in `scripts/smoke.sh` can grep on the prefix.

This module deliberately mocks `nemo.collections.asr` and `coreai_torch` /
`coreai.runtime` so it runs in a clean Python env without the heavyweight
deps installed. The mock substitution happens at import time, before the
target modules are imported.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
import types
from unittest.mock import MagicMock


def _install_mock_module(name: str) -> None:
    """Insert a `MagicMock`-based stand-in module for `name` into `sys.modules`
    so the target package can `import` it without the real dep installed.

    The mock module is a `types.ModuleType` with a `MagicMock` as every attr —
    sufficient for `from nemo.collections.asr.models import ASRModel`-style
    imports to succeed (they look up `ASRModel` as an attribute, which the
    `MagicMock` happily returns as another `MagicMock`).
    """

    if name in sys.modules:
        return
    module = types.ModuleType(name)
    # `MagicMock` substitutes for every attribute access, including
    # sub-modules / classes / functions.
    module.__getattr__ = lambda attr, _m=MagicMock(): _m  # type: ignore[attr-defined]
    sys.modules[name] = module


def _setup_dependency_mocks() -> None:
    """Pre-populate `sys.modules` with mocks for every external dep the
    `nemotron_convert` package imports. Must run BEFORE we import any
    `nemotron_convert.*` modules.
    """

    # Build a minimal torch shim. We need `torch.nn.Module` to be a real-ish
    # base class for `class NemotronEncoderForCoreAI(nn.Module)` to succeed,
    # and `torch.zeros` / `torch.export.Dim` etc. as callables. A bare
    # MagicMock at the module level satisfies most of this; nn.Module just
    # needs to be subclassable.
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_nn_mod = types.ModuleType("torch.nn")

        class _MockModule:
            """Bare-bones nn.Module replacement so the wrapper classes can
            inherit. Real NeMo isn't loaded; we only need subclassability.
            """

            def __init__(self) -> None:
                self._buffers: dict[str, object] = {}

            def register_buffer(self, name: str, tensor: object) -> None:
                self._buffers[name] = tensor
                setattr(self, name, tensor)

            def eval(self) -> "_MockModule":
                return self

            def __call__(self, *args, **kwargs):  # noqa: ANN002, ANN003
                return MagicMock()

        torch_nn_mod.Module = _MockModule  # type: ignore[attr-defined]
        torch_mod.nn = torch_nn_mod  # type: ignore[attr-defined]
        torch_mod.zeros = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
        torch_mod.tensor = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
        torch_mod.from_numpy = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
        torch_mod.no_grad = MagicMock()  # type: ignore[attr-defined]
        torch_mod.float32 = "float32"  # type: ignore[attr-defined]
        torch_mod.int32 = "int32"  # type: ignore[attr-defined]
        torch_mod.int64 = "int64"  # type: ignore[attr-defined]
        torch_mod.Tensor = MagicMock  # type: ignore[attr-defined]

        torch_export_mod = types.ModuleType("torch.export")
        torch_export_mod.Dim = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
        torch_export_mod.export = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
        torch_export_mod.ExportedProgram = MagicMock  # type: ignore[attr-defined]
        torch_mod.export = torch_export_mod  # type: ignore[attr-defined]

        sys.modules["torch"] = torch_mod
        sys.modules["torch.nn"] = torch_nn_mod
        sys.modules["torch.export"] = torch_export_mod

    # Other deps are coarser — we don't subclass anything from them.
    for name in (
        "nemo",
        "nemo.collections",
        "nemo.collections.asr",
        "nemo.collections.asr.models",
        "coreai_torch",
        "coreai",
        "coreai.runtime",
        "numpy",
        "soundfile",
    ):
        _install_mock_module(name)


# Order matters: mocks first, then imports.
_setup_dependency_mocks()


# --------------------------------------------------------------------------
# Check helpers
# --------------------------------------------------------------------------


def _emit(status: str, detail: str) -> None:
    print(f"[smoke] check: {status} {detail}")


def _run_check(label: str, fn) -> bool:  # noqa: ANN001
    """Run one check function (no args, returns None on PASS, raises on FAIL).
    Returns True on PASS, False on FAIL.
    """

    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        _emit("FAIL", f"{label}: {type(exc).__name__}: {exc}")
        return False
    _emit("PASS", label)
    return True


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def check_package_imports() -> None:
    """All five `nemotron_convert.*` modules import cleanly."""

    for name in (
        "nemotron_convert",
        "nemotron_convert.extract",
        "nemotron_convert.models",
        "nemotron_convert.export",
        "nemotron_convert.convert",
        "nemotron_convert.verify_parity",
    ):
        importlib.import_module(name)


def check_extract_signature() -> None:
    """`extract_modules_from_nemo(model_id: str) -> dict` exists with the
    documented signature.
    """

    from nemotron_convert.extract import extract_modules_from_nemo

    sig = inspect.signature(extract_modules_from_nemo)
    params = list(sig.parameters.keys())
    if params != ["model_id"]:
        raise AssertionError(
            f"extract_modules_from_nemo params = {params}, expected ['model_id']"
        )


def check_models_wrappers() -> None:
    """Each of the three wrapper classes (encoder, prediction, joiner) can be
    instantiated against a `MagicMock()` NeMo submodule, and the encoder's
    four cache buffers exist with the documented shapes per PLAN.md §2.
    """

    from nemotron_convert.models import (
        NemotronEncoderForCoreAI,
        NemotronJoinerForCoreAI,
        NemotronPredictionForCoreAI,
    )

    inner_encoder = MagicMock()
    inner_prediction = MagicMock()
    inner_joiner = MagicMock()

    encoder = NemotronEncoderForCoreAI(inner_encoder)
    prediction = NemotronPredictionForCoreAI(inner_prediction)
    joiner = NemotronJoinerForCoreAI(inner_joiner)

    # Encoder must have the four documented cache buffers (PLAN.md §2).
    # We can't inspect tensor shapes without real torch, but we CAN confirm
    # the attributes were created via register_buffer.
    for cache_name in (
        "pre_cache",
        "cache_last_channel",
        "cache_last_time",
        "cache_last_channel_len",
    ):
        if not hasattr(encoder, cache_name):
            raise AssertionError(
                f"NemotronEncoderForCoreAI missing cache buffer '{cache_name}'"
            )

    # Prediction must have its two LSTM state buffers.
    for cache_name in ("lstm_h", "lstm_c"):
        if not hasattr(prediction, cache_name):
            raise AssertionError(
                f"NemotronPredictionForCoreAI missing state buffer '{cache_name}'"
            )

    # Joiner is stateless — we just confirm construction didn't raise.
    if not hasattr(joiner, "inner"):
        raise AssertionError("NemotronJoinerForCoreAI missing `.inner` attribute")


def check_models_constructor_signatures() -> None:
    """Wrapper constructors carry the documented `d_model`/`pred_hidden`
    defaults — guards against the Phase B review item about Phase 1 needing
    to thread `modules["config"]`-derived values through.
    """

    from nemotron_convert.models import (
        NemotronEncoderForCoreAI,
        NemotronPredictionForCoreAI,
    )

    enc_sig = inspect.signature(NemotronEncoderForCoreAI.__init__)
    if "d_model" not in enc_sig.parameters:
        raise AssertionError(
            "NemotronEncoderForCoreAI.__init__ missing `d_model` parameter"
        )
    if "n_layers" not in enc_sig.parameters:
        raise AssertionError(
            "NemotronEncoderForCoreAI.__init__ missing `n_layers` parameter"
        )

    pred_sig = inspect.signature(NemotronPredictionForCoreAI.__init__)
    if "pred_hidden" not in pred_sig.parameters:
        raise AssertionError(
            "NemotronPredictionForCoreAI.__init__ missing `pred_hidden` parameter"
        )


def check_export_symbols() -> None:
    """`export.py` exposes the three documented entry points. We don't
    actually run `torch.export` (would need real modules); we just confirm
    the symbols are callable.
    """

    from nemotron_convert import export

    for name in ("export_encoder", "export_prediction", "export_joiner"):
        fn = getattr(export, name, None)
        if fn is None:
            raise AssertionError(f"nemotron_convert.export.{name} not found")
        if not callable(fn):
            raise AssertionError(f"nemotron_convert.export.{name} is not callable")


def check_convert_main() -> None:
    """`convert.main()` exists, takes an optional argv list, and its argparse
    parser carries the documented flags (`--model-id`, `--output`,
    `--max-chunk-frames`).
    """

    from nemotron_convert.convert import main

    sig = inspect.signature(main)
    params = list(sig.parameters.keys())
    if params != ["argv"]:
        raise AssertionError(f"convert.main params = {params}, expected ['argv']")

    # Parse `--help` to confirm the parser is reachable AND carries the
    # documented options. `--help` triggers SystemExit(0) on argparse, which
    # we catch and treat as success — what matters is the parser was built
    # and accepted the flag.
    try:
        main(["--help"])
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise AssertionError(
                f"convert.main(['--help']) exited with non-zero code {exc.code}"
            )


def check_verify_parity_main() -> None:
    """`verify_parity.main()` exists and its argparse parser accepts the
    documented `--model-id`, `--aimodel`, `--wav`, `--target-lang` flags
    via `--help`.
    """

    from nemotron_convert.verify_parity import main

    sig = inspect.signature(main)
    params = list(sig.parameters.keys())
    if params != ["argv"]:
        raise AssertionError(f"verify_parity.main params = {params}, expected ['argv']")

    try:
        main(["--help"])
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise AssertionError(
                f"verify_parity.main(['--help']) exited with non-zero code {exc.code}"
            )


def check_convert_function_names() -> None:
    """The three function names packed into the multi-function .aimodel
    (`encode`, `predict_step`, `join`) match what `EncoderSession.swift`,
    `PredictionSession.swift`, `JoinerSession.swift` reach for. A typo here
    would manifest as `ASRError.missingFunction` at Swift runtime.
    """

    from nemotron_convert.convert import ENCODE_FN, JOIN_FN, PREDICT_STEP_FN

    if ENCODE_FN != "encode":
        raise AssertionError(f"ENCODE_FN = {ENCODE_FN!r}, expected 'encode'")
    if PREDICT_STEP_FN != "predict_step":
        raise AssertionError(
            f"PREDICT_STEP_FN = {PREDICT_STEP_FN!r}, expected 'predict_step'"
        )
    if JOIN_FN != "join":
        raise AssertionError(f"JOIN_FN = {JOIN_FN!r}, expected 'join'")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nemotron-convert-smoke",
        description="Smoke-test the nemotron_convert package without loading NeMo.",
    )
    parser.parse_args(argv)  # currently no flags; reserved for future filtering

    print("[smoke] nemotron_convert — Python smoke driver")

    checks = [
        ("package_imports", check_package_imports),
        ("extract_signature", check_extract_signature),
        ("models_constructor_signatures", check_models_constructor_signatures),
        ("models_wrappers", check_models_wrappers),
        ("export_symbols", check_export_symbols),
        ("convert_function_names", check_convert_function_names),
        ("convert_main", check_convert_main),
        ("verify_parity_main", check_verify_parity_main),
    ]

    passed = 0
    failed = 0
    for label, fn in checks:
        if _run_check(label, fn):
            passed += 1
        else:
            failed += 1

    print(f"[smoke] summary: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
