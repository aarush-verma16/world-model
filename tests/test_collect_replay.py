"""Replay sampling contracts and collect-path helpers (no live Crafter)."""

from __future__ import annotations

from pathlib import Path

import torch

from models.encoder import Encoder
from models.rssm import one_hot_action
from training.collect import count_unlocked_achievements, ste_action_to_int
from training.replay_buffer import ReplayBuffer
from training.rollout import encode_sequence


def test_one_hot_action_batch_time_layout() -> None:
    actions = torch.tensor([[0, 2, 1], [3, 3, 0]])
    oh = one_hot_action(actions, action_dim=4)
    assert oh.shape == (2, 3, 4)
    assert torch.equal(oh[0, 1], torch.tensor([0.0, 0.0, 1.0, 0.0]))


def test_count_unlocked_achievements() -> None:
    assert count_unlocked_achievements(None) == 0
    assert count_unlocked_achievements({}) == 0
    assert count_unlocked_achievements({"achievements": "nope"}) == 0
    assert count_unlocked_achievements({"achievements": {"wood": 0, "stone": 0}}) == 0
    assert count_unlocked_achievements({"achievements": {"wood": 2, "stone": 0, "coal": 1}}) == 2


def test_ste_action_to_int_rejects_empty() -> None:
    try:
        ste_action_to_int(torch.zeros(0, 5))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty action")


def test_encode_sequence_keeps_batch_time() -> None:
    enc = Encoder(embed_dim=32, channels=(8, 16, 16, 16), blocks=1)
    obs = torch.randint(0, 256, (2, 6, 64, 64, 3), dtype=torch.uint8)
    embeds = encode_sequence(enc, obs, torch.device("cpu"))
    assert embeds.shape == (2, 6, 32)
    assert torch.isfinite(embeds).all()


def test_replay_sample_empty_and_too_short_raise() -> None:
    buf = ReplayBuffer(seed=0)
    try:
        buf.sample(2, 4)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on empty buffer")
    buf.add_episode(
        torch.zeros(3, 64, 64, 3, dtype=torch.uint8),
        torch.zeros(3, dtype=torch.int64),
        torch.zeros(3),
        torch.ones(3),
    )
    try:
        buf.sample(2, 8)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when no episode is long enough")


def test_replay_add_rejects_length_mismatch() -> None:
    buf = ReplayBuffer(seed=0)
    try:
        buf.add_episode(
            torch.zeros(5, 64, 64, 3, dtype=torch.uint8),
            torch.zeros(4, dtype=torch.int64),
            torch.zeros(5),
            torch.ones(5),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for length mismatch")


def test_replay_windows_stay_inside_one_episode() -> None:
    buf = ReplayBuffer(seed=1)
    for ep_i in range(2):
        t = 10
        actions = torch.full((t,), ep_i * 100, dtype=torch.int64) + torch.arange(t)
        buf.add_episode(
            torch.zeros(t, 64, 64, 3, dtype=torch.uint8),
            actions,
            torch.zeros(t),
            torch.ones(t),
        )
    batch = buf.sample(batch_size=6, seq_len=5)
    for i in range(6):
        acts = batch["actions"][i]
        assert torch.equal(acts[1:] - acts[:-1], torch.ones(4, dtype=torch.int64))
        assert bool((acts < 100).all() or (acts >= 100).all())


def test_replay_state_dict_roundtrip() -> None:
    buf = ReplayBuffer(seed=0, max_steps=1000)
    buf.add_episode(
        torch.ones(6, 64, 64, 3, dtype=torch.uint8),
        torch.arange(6),
        torch.zeros(6),
        torch.ones(6),
    )
    other = ReplayBuffer(seed=9)
    other.load_state_dict(buf.state_dict())
    assert other.num_steps == 6
    assert len(other) == 1
    assert torch.equal(other._episodes[0].actions, torch.arange(6))


def test_resolve_resume_missing_path_raises(tmp_path: Path) -> None:
    from training.ckpt import resolve_resume

    try:
        resolve_resume(tmp_path / "nope.pt", tmp_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
