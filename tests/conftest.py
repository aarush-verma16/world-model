"""Pytest fixtures and a terminal summary for ML-style world-model tests.

These tests do **not** assert reconstruction quality or Crafter score. They
assert the things that silently break a Dreamer implementation: tensor layout,
prior vs posterior naming, gradient flow, freeze/unfreeze, and numerical
invariants (unimix, STE, KL floor, replay windows).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

AREA_LABELS = {
    "test_actor_critic": "actor-critic",
    "test_actor_policy": "actor-critic",
    "test_collect_replay": "collect / replay",
    "test_cuda_step": "cuda",
    "test_device": "device",
    "test_diagnostics": "diagnostics",
    "test_encoder_decoder_shapes": "perception",
    "test_env_crafter": "env (crafter)",
    "test_layout": "perception",
    "test_m7_init": "m7 init",
    "test_outer_loop": "outer-loop",
    "test_preprocess": "preprocess",
    "test_returns": "returns",
    "test_rssm_shapes": "rssm",
    "test_world_model_m3": "world-model",
}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "cuda: needs NVIDIA CUDA (skipped when torch.cuda.is_available() is False)",
    )
    config.addinivalue_line(
        "markers",
        "env: instantiates CrafterReward-v1 (skip with -m \"not env\")",
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        pass


def pytest_report_header(config: pytest.Config) -> list[str]:
    import torch

    cuda = (
        f"CUDA {torch.cuda.get_device_name(0)}"
        if torch.cuda.is_available()
        else "CUDA off (cuda tests will skip)"
    )
    return [
        "ML unit tests: shapes [B,T,...], grads, invariants -- not accuracy or Crafter score.",
        f"torch {torch.__version__}  |  {cuda}",
    ]


def _area(nodeid: str) -> str:
    stem = Path(nodeid.split("::", 1)[0]).stem
    if stem in AREA_LABELS:
        return AREA_LABELS[stem]
    return stem.removeprefix("test_").replace("_", " ")


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Grouped pass/fail table so a long pytest -v run is still scannable."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for key in ("passed", "failed", "skipped", "error", "xfailed", "xpassed"):
        for report in terminalreporter.stats.get(key, []):
            when = getattr(report, "when", "call")
            # Passed/failed emit setup+call+teardown; only count the call.
            # Skips often land on setup.
            if key == "skipped":
                if when not in {"call", "setup"}:
                    continue
            elif when != "call":
                continue
            nodeid = getattr(report, "nodeid", "")
            if not nodeid:
                continue
            counts[_area(nodeid)][key] += 1

    if not counts:
        return

    tw = terminalreporter
    tw.write_sep("=", "ML test results by area")
    width = max(len(name) for name in counts) + 2
    for name in sorted(counts):
        row = counts[name]
        n_pass = row.get("passed", 0)
        n_fail = row.get("failed", 0) + row.get("error", 0)
        n_skip = row.get("skipped", 0)
        n_xfail = row.get("xfailed", 0)
        parts: list[tuple[str, dict]] = []
        if n_pass:
            parts.append((f"{n_pass} passed", {"green": True}))
        if n_fail:
            parts.append((f"{n_fail} failed", {"red": True, "bold": True}))
        if n_skip:
            parts.append((f"{n_skip} skipped", {"yellow": True}))
        if n_xfail:
            parts.append((f"{n_xfail} xfailed", {}))
        tw.write(f"  {name}.".ljust(width + 4, "."))
        tw.write("  ")
        for i, (text, style) in enumerate(parts):
            if i:
                tw.write(", ")
            tw.write(text, **style)
        tw.write("\n")

    failed = terminalreporter.stats.get("failed", [])
    if failed:
        tw.write_sep("=", "Failed tests")
        for report in failed:
            tw.write(f"  FAIL  {report.nodeid}\n", red=True, bold=True)

    if exitstatus == 0:
        tw.write("\n  All ML invariants held. Training quality is still TensorBoard / notebooks.\n")
    else:
        tw.write(
            "\n  A failed test is a graph/shape/gradient bug, not a 'loss too high' result.\n",
            red=True,
        )
