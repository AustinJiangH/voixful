# scripts/

Top-level driver scripts for the `coreai-nemotron-asr` project. Each script is
self-contained, prints `[smoke] ...` (or equivalent) lines so output is
greppable, and exits non-zero on hard failure.

## `smoke.sh`

End-to-end scaffold smoke driver. Phase K.

**What it does**

A smoke test is the smallest-possible end-to-end exercise of the system that
confirms "the wiring is sane." `smoke.sh` runs every available wiring check in
this environment, reports each step's state (PASS / FAIL / SKIPPED), and
exits non-zero if any step failed.

**Steps**

| Step | Action | SKIPPED when |
|---|---|---|
| A | Confirm `xcrun --find swiftc` and `python3 --version` resolve | one or both missing |
| B | `swift build` in PROJECT_ROOT | swiftc absent |
| C | `swift test` (runs unconditional unit tests; fixture-dependent tests `XCTSkip`) | swiftc absent |
| D | `swift run nemotron-asr smoke` (model-free Steps 1-3) | swiftc absent |
| E | `python -m nemotron_convert.smoke` (mocks every NeMo / coreai_torch import) | venv absent or package not importable |
| F | Summary | never |

**Usage**

```bash
./scripts/smoke.sh
```

No arguments. Run from any directory; the script resolves `PROJECT_ROOT` from
its own location.

**Expected output on a half-installed env**

Before Xcode 27 beta finishes installing and `pip install -e convert/.` has
been run, every step except A and F reports SKIPPED — that is the expected
state and the script exits 0. Once Xcode 27 is installed, Steps B–D become
real. Once the Python env is set up, Step E becomes real. None of the steps
require a real `.aimodel` file; that's Phase 1 of PLAN.md §7.

**Runtime**

Under 10 seconds wall-clock when the env is fully set up. Steps B and C are
the bulk of the time on a cold build (Swift package compile); subsequent
runs are sub-second per Swift step.

**Bash compatibility**

Bash 3.2 compatible — runs on the default macOS shell with no `mapfile`,
`readarray`, `<<<`, or associative arrays. See the inline comment in the
script header.

## Adding scripts

Drop additional drivers in this directory. Convention:
1. `#!/usr/bin/env bash` + `set -euo pipefail` header.
2. Resolve `PROJECT_ROOT` from `BASH_SOURCE[0]` so the script works from any cwd.
3. Print `[<tool>] step <letter>: <PASS|FAIL|SKIPPED> <detail>` for greppable output.
4. Exit non-zero only on hard failure; SKIPPED is not a failure.
5. Document the script in this README with the same table shape as `smoke.sh`.
