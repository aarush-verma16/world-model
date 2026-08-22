"""Compare early-step scalars across m3 runs (regression check, not training).

Reads the TensorBoard event files in `runs/m3_world_model` and prints the
same scalar at the same step for each run, so "the new run looks worse" can
be checked against the previous run's own step-200/600 numbers instead of
against a memory of its step-8000 images.

Usage: python scripts/compare_runs_early.py [tag ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = Path("runs/m3_world_model")
DEFAULT_TAGS = (
    "loss/recon_l1",
    "loss/recon_embed_l1",
    "loss/kl_dyn_raw",
)
STEPS = (50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 6000, 8000)


def load(path: Path) -> dict[str, dict[int, float]]:
    acc = EventAccumulator(str(path), size_guidance={"scalars": 100000})
    acc.Reload()
    out: dict[str, dict[int, float]] = {}
    for tag in acc.Tags().get("scalars", []):
        out[tag] = {e.step: e.value for e in acc.Scalars(tag)}
    return out


def main() -> None:
    tags = tuple(sys.argv[1:]) or DEFAULT_TAGS
    files = sorted(RUNS.glob("*tfevents*"), key=lambda p: p.stat().st_mtime)
    runs = [(p.name.split(".")[-2], load(p)) for p in files]
    runs = [(name, data) for name, data in runs if data]

    available = sorted({t for _, data in runs for t in data})
    print(f"runs: {[n for n, _ in runs]}")
    print(f"tags: {available}\n")

    for tag in tags:
        if tag not in available:
            print(f"[skip] {tag} not logged\n")
            continue
        print(f"== {tag} ==")
        header = "step   " + "".join(f"{n:>14}" for n, _ in runs)
        print(header)
        for step in STEPS:
            cells = []
            for _, data in runs:
                series = data.get(tag, {})
                near = [s for s in series if abs(s - step) <= 25]
                cells.append(
                    f"{series[min(near, key=lambda s: abs(s - step))]:>14.4f}"
                    if near
                    else f"{'-':>14}"
                )
            print(f"{step:<7}" + "".join(cells))
        print()


if __name__ == "__main__":
    main()
