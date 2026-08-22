"""Run all Milestone 0 exit checks in one shot.

Usage (worldmodel env active, Windows / CUDA):
    python scripts/verify_m0.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from training.device import describe_device, get_device, warn_if_not_cuda


def check_cuda() -> None:
    print(f"[M0] torch {torch.__version__}")
    print(f"[M0] CUDA built: {torch.version.cuda}")
    ok = torch.cuda.is_available()
    print(f"[M0] CUDA available: {ok}")
    if not ok:
        raise SystemExit(
            "FAIL: torch.cuda.is_available() is False. Install a CUDA 12.8+ "
            "PyTorch wheel (see README) — the default PyPI torch is CPU-only "
            "on Windows. RTX 50-series also needs a recent NVIDIA driver."
        )
    device = get_device()
    print(f"[M0] {describe_device(device)}")
    cap = torch.cuda.get_device_capability()
    if cap < (8, 0):
        print(f"[M0] WARN: compute capability {cap} is older than Ampere; bf16 AMP may be slow.")
    x = torch.ones(2, device="cuda")
    y = (x * 2).sum()
    torch.cuda.synchronize()
    print(f"[M0] CUDA tensor device: {x.device}  sanity={float(y):.0f}")


def run(cmd: list[str]) -> None:
    print(f"[M0] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    check_cuda()
    warn_if_not_cuda(get_device())
    run([sys.executable, str(root / "scripts" / "smoke_test_crafter.py")])
    run([sys.executable, str(root / "scripts" / "log_dummy_tensorboard.py")])
    run([sys.executable, str(root / "scripts" / "watch_crafter.py")])

    event_files = list((root / "runs" / "m0_dummy").glob("events.out.tfevents.*"))
    gif = root / "results" / "m0_random_rollout.gif"
    if not event_files:
        raise SystemExit("FAIL: TensorBoard event file missing")
    if not gif.exists():
        raise SystemExit("FAIL: Crafter GIF missing")

    print()
    print("M0 exit criteria checklist:")
    print("  [x] CUDA available")
    print("  [x] CrafterReward-v1 resets/steps with (64, 64, 3)")
    print("  [x] Dummy TensorBoard scalar log written (view via tensorboard --logdir runs)")
    print("  [x] Visual random-policy GIF written to results/m0_random_rollout.gif")
    print("PASS: Milestone 0 verification succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
