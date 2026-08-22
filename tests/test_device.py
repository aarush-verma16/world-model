"""Device helper unit tests (CPU-safe; CUDA optional)."""

from __future__ import annotations

import torch

from training.device import get_device, parse_amp, to_device


def test_parse_amp_off_on_cpu() -> None:
    assert parse_amp("bf16", torch.device("cpu")) is None
    assert parse_amp("off", torch.device("cpu")) is None


def test_parse_amp_cuda_names() -> None:
    cuda = torch.device("cuda")
    if not torch.cuda.is_available():
        # Names still have to parse; dtype is only used when device is CUDA.
        assert parse_amp("off", cuda) is None
        assert parse_amp("bf16", cuda) is torch.bfloat16
        assert parse_amp("fp16", cuda) is torch.float16
        return
    assert parse_amp("bf16", cuda) is torch.bfloat16
    assert parse_amp("fp16", cuda) is torch.float16
    assert parse_amp("off", cuda) is None


def test_to_device_moves_tensors() -> None:
    device = get_device()
    batch = {
        "obs": torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8),
        "label": "keep",
    }
    out = to_device(batch, device)
    assert out["obs"].device.type == device.type
    assert out["label"] == "keep"


def test_get_device_returns_torch_device() -> None:
    device = get_device()
    assert isinstance(device, torch.device)
    if torch.cuda.is_available():
        assert device.type == "cuda"
