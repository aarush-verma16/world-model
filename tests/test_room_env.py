"""The room crop: the station is one 16×16 tile, in front of the agent or behind."""

from __future__ import annotations

import numpy as np

from envs.room_env import (
    MATERIAL_STATION,
    OBS_SIZE,
    STATION_POS,
    TILE,
    USE,
    RoomEnv,
)
from envs.room_teacher import RoomTeacher

STATION_COLOR = np.array([230, 140, 20], dtype=np.uint8)


def _station_pixels(obs: np.ndarray) -> int:
    return int(np.all(obs == STATION_COLOR, axis=-1).sum())


def test_reset_frame_is_64_and_five_actions() -> None:
    env = RoomEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE, OBS_SIZE, 3)
    assert obs.dtype == np.uint8
    assert env.action_dim == 5
    assert info["local_materials"].shape == (16,)
    assert info["local_objects"].shape == (16,)
    assert info["local_materials"].dtype == np.uint8


def test_facing_the_station_puts_one_tile_in_the_crop() -> None:
    env = RoomEnv()
    env.reset()
    # South neighbor, facing north. The station is the nearest cell ahead.
    env._pos = (STATION_POS[0], STATION_POS[1] + 1)
    env._facing = 0
    obs, _reward, _terminated, _truncated, info = env.step(0)
    assert env.station_in_view()
    assert info["station_in_view"] is True
    assert int(info["local_materials"][1]) == MATERIAL_STATION
    assert _station_pixels(obs) == TILE * TILE
    # That tile is the bottom row, second column.
    block = obs[OBS_SIZE - TILE :, TILE : 2 * TILE]
    assert np.all(block == STATION_COLOR)


def test_facing_away_keeps_the_adjacent_station_out_of_the_crop() -> None:
    env = RoomEnv()
    env.reset()
    env._pos = (STATION_POS[0], STATION_POS[1] + 1)
    env._facing = 2  # south, station behind
    obs, reward, terminated, _truncated, info = env.step(USE)
    assert env.pos == (STATION_POS[0], STATION_POS[1] + 1)
    assert not env.station_in_view()
    assert info["station_in_view"] is False
    assert MATERIAL_STATION not in info["local_materials"]
    assert _station_pixels(obs) == 0
    assert reward == 1.0
    assert terminated
    assert info["discount"] == 0.0
    assert info["used"] == 1


def test_the_crop_requirement_rejects_the_other_pose() -> None:
    facing = RoomEnv(require_station_in_view=True)
    facing.reset()
    facing._pos = (STATION_POS[0], STATION_POS[1] + 1)
    facing._facing = 2
    _obs, reward, terminated, _truncated, _info = facing.step(USE)
    assert reward == 0.0 and not terminated

    away = RoomEnv(require_station_in_view=False)
    away.reset()
    away._pos = (STATION_POS[0], STATION_POS[1] + 1)
    away._facing = 0
    _obs, reward, terminated, _truncated, _info = away.step(USE)
    assert reward == 0.0 and not terminated
    away._facing = 2
    _obs, reward, terminated, _truncated, _info = away.step(USE)
    assert reward == 1.0 and terminated


def test_use_away_from_the_station_does_not_score() -> None:
    env = RoomEnv()
    env.reset()
    _obs, reward, terminated, truncated, info = env.step(USE)
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["used"] == 0


def _run_teacher(station_in_view: bool, hold_steps: int = 48) -> list[dict]:
    env = RoomEnv(max_steps=160)
    teacher = RoomTeacher(station_in_view=station_in_view, hold_steps=hold_steps)
    env.reset()
    teacher.reset(env)
    rows = []
    for _ in range(env.max_steps):
        obs, reward, terminated, truncated, info = env.step(teacher.act(env))
        rows.append(
            {
                "obs": obs,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "in_view": bool(info["station_in_view"]),
                "pixels": _station_pixels(obs),
            }
        )
        if terminated or truncated:
            break
    return rows


def test_both_teachers_use_once_and_the_crop_matches_the_switch() -> None:
    for station_in_view in (True, False):
        rows = _run_teacher(station_in_view)
        rewards = [row["reward"] for row in rows]
        assert sum(rewards) == 1.0
        assert rewards[-1] == 1.0
        assert rows[-1]["terminated"]
        assert not rows[-1]["truncated"]
        window = rows[-32:]
        assert len(window) == 32
        assert all(row["in_view"] is station_in_view for row in window)
        if station_in_view:
            assert all(row["pixels"] == TILE * TILE for row in window)
        else:
            assert all(row["pixels"] == 0 for row in window)


def test_the_walk_is_shared_and_only_the_final_facing_changes() -> None:
    env = RoomEnv()
    facing = RoomTeacher(station_in_view=True, hold_steps=48)
    away = RoomTeacher(station_in_view=False, hold_steps=48)
    env.reset()
    facing.reset(env)
    env.reset()
    away.reset(env)
    # Three steps north, then the away teacher turns around. The hold is the same tail.
    assert facing._actions[:3] == away._actions[:3]
    assert facing._actions[:3] == [3, 3, 3]
    assert away._actions[3:5] == [1, 1]
    assert facing._actions[3:] == [0] * 48 + [4]
    assert away._actions[5:] == [0] * 48 + [4]
