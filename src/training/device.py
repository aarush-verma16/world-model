"""Device selection and CUDA runtime helpers.

Target is NVIDIA CUDA (RTX 5080 / Blackwell). MPS is a leftover fallback only
so an old Mac checkout still imports; it is not a supported training path.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor


def get_device() -> torch.device:
    """Return CUDA if available, else MPS, else CPU.

    Training is expected to run on CUDA. CPU/MPS are last-resort fallbacks
    for import/debug only.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_runtime(device: torch.device) -> None:
    """Enable CUDA speed defaults (TF32, cuDNN autotune). No-op on CPU/MPS."""
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def describe_device(device: torch.device) -> str:
    """One-line hardware summary for logs and the device notebook."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        mem_gb = props.total_memory / (1024**3)
        cap = torch.cuda.get_device_capability(idx)
        return (
            f"{device}  {props.name}  {mem_gb:.1f} GiB  "
            f"cc{cap[0]}.{cap[1]}  torch {torch.__version__}"
        )
    return f"{device}  torch {torch.__version__}"


def parse_amp(name: str | None, device: torch.device) -> torch.dtype | None:
    """Config `train.amp` → autocast dtype, or None to run in fp32.

    `bf16` is the default on CUDA (RTX 30/40/50 native, no GradScaler).
    `fp16` is supported but needs a GradScaler. Anything else / non-CUDA → off.
    """
    if device.type != "cuda" or not name:
        return None
    key = str(name).strip().lower()
    if key in {"off", "none", "fp32", "float32"}:
        return None
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"unknown amp {name!r}; use bf16, fp16, or off")


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    """`torch.autocast` on CUDA when AMP is on, otherwise a no-op context."""
    if amp_dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def make_grad_scaler(device: torch.device, amp_dtype: torch.dtype | None):
    """fp16 needs a GradScaler; bf16 / fp32 do not."""
    enabled = device.type == "cuda" and amp_dtype is torch.float16
    return torch.amp.GradScaler("cuda", enabled=enabled)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a replay batch onto `device`."""
    non_blocking = device.type == "cuda"
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            out[key] = value.to(device, non_blocking=non_blocking)
        else:
            out[key] = value
    return out


def warn_if_not_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        return
    print(
        "WARNING: CUDA is not available — training will be far slower on "
        f"{device}. Install a CUDA 12.8+ PyTorch wheel (see README) and "
        "confirm `nvidia-smi` sees the GPU."
    )


def vram_peak_gb() -> tuple[float, float] | None:
    """Return `(allocated_giB, reserved_giB)` peaks, or None if not CUDA."""
    if not torch.cuda.is_available():
        return None
    alloc = torch.cuda.max_memory_allocated() / (1024**3)
    reserved = torch.cuda.max_memory_reserved() / (1024**3)
    return alloc, reserved
