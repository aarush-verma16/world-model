"""Write a dummy TensorBoard scalar so we can confirm local logging works (M0).

Usage (worldmodel env active):
    python scripts/log_dummy_tensorboard.py
    tensorboard --logdir runs
Then open http://localhost:6006 and check the Scalars tab for `m0/dummy_loss`.
"""

from __future__ import annotations

import math
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


LOG_DIR = Path("runs/m0_dummy")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(LOG_DIR))
    try:
        for step in range(50):
            # Simple decaying curve so the Scalars plot is obviously non-flat.
            value = math.exp(-step / 12.0) + 0.05 * math.sin(step)
            writer.add_scalar("m0/dummy_loss", value, step)
        writer.flush()
    finally:
        writer.close()

    event_files = list(LOG_DIR.glob("events.out.tfevents.*"))
    if not event_files:
        raise SystemExit(f"FAIL: no TensorBoard event files written under {LOG_DIR}")

    print(f"Wrote dummy scalars to {LOG_DIR.resolve()}")
    print(f"Event file: {event_files[0].name}")
    print()
    print("View in browser:")
    print(f"  tensorboard --logdir runs")
    print("  open http://localhost:6006  -> Scalars -> m0/dummy_loss")
    print("PASS: dummy TensorBoard log written")


if __name__ == "__main__":
    main()
