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


def test_parse_amp_unknown_raises_on_cuda_device() -> None:
    try:
        parse_amp("fp8", torch.device("cuda"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown amp")


def test_make_grad_scaler_disabled_except_cuda_fp16() -> None:
    from training.device import make_grad_scaler

    cuda = torch.device("cuda")
    assert make_grad_scaler(cuda, torch.bfloat16).is_enabled() is False
    assert make_grad_scaler(cuda, None).is_enabled() is False
    assert make_grad_scaler(torch.device("cpu"), torch.float16).is_enabled() is False


def test_configure_runtime_and_describe_cpu() -> None:
    from training.device import configure_runtime, describe_device

    cpu = torch.device("cpu")
    configure_runtime(cpu)
    text = describe_device(cpu)
    assert "cpu" in text
    assert "torch" in text
