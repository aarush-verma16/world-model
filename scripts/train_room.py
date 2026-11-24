"""One room config. The notebook runs all four; this is the same function."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.room_loop import train_room


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    train_room(args.config)


if __name__ == "__main__":
    main()
