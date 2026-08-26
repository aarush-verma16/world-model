"""Run the ML-style unit suite with a readable terminal report.

From the repo root, conda env `worldmodel` active:

    python scripts/run_tests.py
    python scripts/run_tests.py --fast          # skip Crafter env smoke
    python scripts/run_tests.py -k rssm         # subset (pytest -k)
    python scripts/run_tests.py tests/test_rssm_shapes.py -q

This is not training. A pass means shapes, gradients, and invariants held.
Quality (recon, video-pred, Crafter return) lives in the notebooks / TensorBoard.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BANNER = """
============================================================
 World-model tests (ML-style, not app-style)
============================================================
  Asserted here:
    tensor layout [batch, time, ...]
    z_prior vs z_posterior stay distinct
    gradients reach encoder / RSSM / decoder / actor
    unimix, STE, KL free-nats, freeze/unfreeze, replay windows
    Crafter obs 64x64x3  (skip with --fast)

  Not asserted here (those are training / TensorBoard):
    reconstruction quality, imagination video, Crafter score
============================================================
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fast = False
    if "--fast" in argv:
        fast = True
        argv.remove("--fast")

    repo = Path(__file__).resolve().parent.parent
    os.chdir(repo)
    if str(repo / "src") not in sys.path:
        sys.path.insert(0, str(repo / "src"))

    print(BANNER)
    if fast:
        print("  (--fast: skipping tests marked env)\n")

    try:
        import pytest
    except ImportError:
        print("pytest is not installed. From the worldmodel env:")
        print('  pip install -e ".[dev]"')
        return 1

    pytest_args = ["tests"]
    if fast:
        pytest_args.extend(["-m", "not env"])
    pytest_args.extend(argv)
    return int(pytest.main(pytest_args))


if __name__ == "__main__":
    raise SystemExit(main())
