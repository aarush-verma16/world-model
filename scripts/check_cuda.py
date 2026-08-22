"""Print CUDA / GPU info. Used by setup_windows.ps1 and as a quick sanity check.

Usage:
    python scripts/check_cuda.py
"""

from __future__ import annotations

import sys

import torch

from training.device import describe_device, get_device, warn_if_not_cuda


def main() -> int:
    print(f"torch {torch.__version__}")
    print(f"cuda built: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print(
            "FAIL: CUDA not available. Install a CUDA 12.8+ wheel "
            "(pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128)."
        )
        return 1
    device = get_device()
    print(describe_device(device))
    warn_if_not_cuda(device)
    x = torch.ones(4, device="cuda")
    y = float((x @ x).sum())
    torch.cuda.synchronize()
    print(f"matmul sanity: {y:.0f}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
