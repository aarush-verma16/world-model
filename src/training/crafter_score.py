"""Official Crafter geometric-mean score (danijar/crafter analysis.common).

Success rate per achievement is the percent of episodes with count >= 1.
Score = exp(mean(log(1 + percent))) - 1 with percents in 0–100 (1% offset).

This is the M6 cited metric. 4-episode 400-step return is not a Crafter score
(finding 11). Names are pinned to the installed `crafter` package.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import crafter.constants as crafter_constants

ACHIEVEMENT_NAMES: tuple[str, ...] = tuple(str(n) for n in crafter_constants.achievements)


def achievement_names() -> tuple[str, ...]:
    """The 22 Crafter achievement ids, in package order."""
    return ACHIEVEMENT_NAMES


def achievement_counts_from_info(
    info: dict[str, Any] | None,
    names: Sequence[str] | None = None,
) -> dict[str, int]:
    """Map `info['achievements']` onto `names` (missing keys → 0)."""
    keys = tuple(names) if names is not None else ACHIEVEMENT_NAMES
    raw: dict[str, int] = {}
    if info and isinstance(info.get("achievements"), dict):
        raw = {str(k): int(float(v)) for k, v in info["achievements"].items()}
    return {n: int(raw.get(n, 0)) for n in keys}


def flatten_achievement_counts(
    counts: Mapping[str, float],
    names: Sequence[str] | None = None,
) -> dict[str, int]:
    """`{name: count}` → `{achievement_<name>: int}` for jsonl / danijar stats."""
    keys = tuple(names) if names is not None else ACHIEVEMENT_NAMES
    return {f"achievement_{n}": int(float(counts.get(n, 0))) for n in keys}


def episode_jsonl_row(
    *,
    env_steps: int,
    length: int,
    ep_return: float,
    counts: Mapping[str, float],
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """One collect-episode record for `collect_episodes.jsonl`."""
    row: dict[str, Any] = {
        "env_steps": int(env_steps),
        "length": int(length),
        "return": float(ep_return),
    }
    row.update(flatten_achievement_counts(counts, names=names))
    return row


def _counts_from_row(row: Mapping[str, Any], names: Sequence[str]) -> dict[str, int]:
    nested = row.get("achievement_counts")
    if isinstance(nested, dict):
        return {n: int(float(nested.get(n, 0))) for n in names}
    return {n: int(float(row.get(f"achievement_{n}", 0))) for n in names}


def success_percents(
    episodes: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str] | None = None,
    budget: int | None = None,
) -> dict[str, float]:
    """Percent of episodes (0–100) that unlocked each achievement at least once.

    If `budget` is set, keep rows with `env_steps <= budget` (continued runs
    log from the resume step, not from env step 0).
    """
    keys = tuple(names) if names is not None else ACHIEVEMENT_NAMES
    rows = list(episodes)
    if budget is not None:
        rows = [r for r in rows if int(r.get("env_steps", 0)) <= int(budget)]
    if not rows:
        return {n: 0.0 for n in keys}
    percents: dict[str, float] = {}
    n_ep = len(rows)
    for name in keys:
        unlocked = sum(1 for r in rows if _counts_from_row(r, keys)[name] >= 1)
        percents[name] = 100.0 * unlocked / n_ep
    return percents


def geometric_mean_score(percents: Mapping[str, float] | np.ndarray) -> float:
    """danijar `compute_scores`: exp(mean(log(1 + p))) - 1, p in [0, 100]."""
    if isinstance(percents, Mapping):
        values = np.asarray([float(percents[k]) for k in percents], dtype=np.float64)
    else:
        values = np.asarray(percents, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    if np.any(values < 0.0) or np.any(values > 100.0):
        raise ValueError(f"percents must be in [0, 100], got min={values.min()} max={values.max()}")
    if not np.all(np.isfinite(values)):
        raise ValueError("percents must be finite")
    return float(np.exp(np.mean(np.log(1.0 + values))) - 1.0)


def score_from_episodes(
    episodes: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str] | None = None,
    budget: int | None = None,
) -> tuple[float, dict[str, float]]:
    """Return `(crafter_score, percents)` for a list of episode rows."""
    percents = success_percents(episodes, names=names, budget=budget)
    return geometric_mean_score(percents), percents


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON object as a line. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=True))
        f.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a jsonl file. Missing file → empty list."""
    path = Path(path)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows
