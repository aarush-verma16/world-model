"""Run all Milestone 0 exit checks in one shot.

Usage (worldmodel env active):
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python scripts/verify_m0.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import torch


def check_mps() -> None:
    ok = torch.backends.mps.is_available()
    print(f"[M0] MPS available: {ok}")
    if not ok:
        raise SystemExit("FAIL: torch.backends.mps.is_available() is False")
    x = torch.ones(2, device="mps")
    print(f"[M0] MPS tensor device: {x.device}")


def check_jax_cpu() -> None:
    devices = jax.devices()
    backend = jax.default_backend()
    print(f"[M0] JAX devices: {devices}")
    print(f"[M0] JAX default_backend: {backend}")
    if backend != "cpu":
        raise SystemExit(f"FAIL: expected JAX CPU backend, got {backend!r}")


def run(cmd: list[str]) -> None:
    print(f"[M0] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    check_mps()
    check_jax_cpu()
    run([sys.executable, str(root / "scripts" / "smoke_test_craftax.py")])
    run([sys.executable, str(root / "scripts" / "log_dummy_tensorboard.py")])
    run([sys.executable, str(root / "scripts" / "watch_craftax.py")])

    event_files = list((root / "runs" / "m0_dummy").glob("events.out.tfevents.*"))
    gif = root / "results" / "m0_random_rollout.gif"
    if not event_files:
        raise SystemExit("FAIL: TensorBoard event file missing")
    if not gif.exists():
        raise SystemExit("FAIL: Craftax GIF missing")

    print()
    print("M0 exit criteria checklist:")
    print("  [x] PyTorch MPS available")
    print("  [x] JAX running on CPU (not Metal)")
    print("  [x] Craftax-Classic-Pixels resets/steps with final obs (64, 64, 3)")
    print("  [x] Dummy TensorBoard scalar log written (view via tensorboard --logdir runs)")
    print("  [x] Visual random-policy GIF written to results/m0_random_rollout.gif")
    print("PASS: Milestone 0 verification succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
