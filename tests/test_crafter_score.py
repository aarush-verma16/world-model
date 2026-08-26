"""Official Crafter gmean + death-vs-timeout split (no live env needed)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from envs.crafter_env import split_crafter_done
from training.crafter_score import (
    ACHIEVEMENT_NAMES,
    append_jsonl,
    episode_jsonl_row,
    geometric_mean_score,
    load_jsonl,
    score_from_episodes,
    success_percents,
)


def test_achievement_names_are_the_crafter_22() -> None:
    assert len(ACHIEVEMENT_NAMES) == 22
    assert "collect_wood" in ACHIEVEMENT_NAMES
    assert "wake_up" in ACHIEVEMENT_NAMES
    assert len(set(ACHIEVEMENT_NAMES)) == 22


def test_geometric_mean_all_zero_is_zero() -> None:
    percents = {n: 0.0 for n in ACHIEVEMENT_NAMES}
    assert geometric_mean_score(percents) == 0.0


def test_geometric_mean_all_hundred_is_hundred() -> None:
    percents = {n: 100.0 for n in ACHIEVEMENT_NAMES}
    assert abs(geometric_mean_score(percents) - 100.0) < 1e-9


def test_geometric_mean_one_percent_offset() -> None:
    """One achievement at 100%, rest 0 → exp(log(101)/22) - 1."""
    percents = {n: 0.0 for n in ACHIEVEMENT_NAMES}
    percents["collect_wood"] = 100.0
    expected = math.exp(math.log(101.0) / 22.0) - 1.0
    assert abs(geometric_mean_score(percents) - expected) < 1e-12
    arr = np.array([100.0] + [0.0] * 21)
    assert abs(geometric_mean_score(arr) - expected) < 1e-12


def test_success_percents_and_budget_filter() -> None:
    names = ("wood", "stone")
    rows = [
        episode_jsonl_row(
            env_steps=10, length=5, ep_return=1.0, counts={"wood": 1, "stone": 0}, names=names
        ),
        episode_jsonl_row(
            env_steps=20, length=5, ep_return=0.0, counts={"wood": 0, "stone": 0}, names=names
        ),
        episode_jsonl_row(
            env_steps=30, length=5, ep_return=1.0, counts={"wood": 1, "stone": 2}, names=names
        ),
    ]
    pct = success_percents(rows, names=names)
    assert pct["wood"] == 100.0 * 2 / 3
    assert pct["stone"] == 100.0 * 1 / 3
    pct_b = success_percents(rows, names=names, budget=20)
    assert pct_b["wood"] == 50.0
    assert pct_b["stone"] == 0.0
    score, _ = score_from_episodes([], names=names)
    assert score == 0.0


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "collect_episodes.jsonl"
    row = episode_jsonl_row(
        env_steps=100016,
        length=200,
        ep_return=0.1,
        counts={"collect_wood": 1},
    )
    append_jsonl(path, row)
    append_jsonl(path, row)
    loaded = load_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0]["achievement_collect_wood"] == 1
    assert loaded[0]["achievement_wake_up"] == 0
    assert load_jsonl(tmp_path / "missing.jsonl") == []


def test_split_crafter_done_death_vs_timeout() -> None:
    assert split_crafter_done(True, {"discount": 0.0}) == (True, False)
    assert split_crafter_done(True, {"discount": 1.0}) == (False, True)
    assert split_crafter_done(False, {"discount": 1.0}) == (False, False)
    assert split_crafter_done(True, {}) == (True, False)
    assert split_crafter_done(False, None) == (False, False)
