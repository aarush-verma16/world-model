"""Checkpoint path helpers shared by the CLI and the training notebook."""

from __future__ import annotations

from pathlib import Path


def _step_from_ckpt_name(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else -1


def resolve_resume(spec: str | Path | None, ckpt_dir: Path) -> Path | None:
    """Return a checkpoint file, or None to start from step 0.

    `spec` is a path, or `"auto"` / `"latest"` to pick `ckpt_latest.pt`,
    then `ckpt_final.pt`, then the highest-step `ckpt_step_*.pt`.
    """
    if spec is None:
        return None
    key = str(spec).strip()
    if key.lower() in {"auto", "latest"}:
        for name in ("ckpt_latest.pt", "ckpt_final.pt"):
            path = ckpt_dir / name
            if path.is_file():
                return path
        numbered = sorted(ckpt_dir.glob("ckpt_step_*.pt"), key=_step_from_ckpt_name)
        return numbered[-1] if numbered else None
    path = Path(key)
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path
