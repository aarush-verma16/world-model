"""Scripted demonstration for `RoomEnv`.

Both teachers walk to the cell south of the station while facing it, so the
approach is the same demonstration. The switch is the pose they hold before
`use`:

- `station_in_view=True`: keep facing the station. It fills one tile of the crop.
- `station_in_view=False`: turn around and hold. The station is one cell behind
  the agent, outside the crop, and `use` still succeeds.

`hold_steps` no-ops follow that pose. An anchored loss window no longer than
`hold_steps` therefore sees only the pose and the `use`, not the walk.
"""

from __future__ import annotations

from collections import deque

from envs.room_env import (
    FACINGS,
    FORWARD,
    MATERIAL_FLOOR,
    NOOP,
    STATION_POS,
    TURN_LEFT,
    TURN_RIGHT,
    USE,
    RoomEnv,
)


def _neighbors(index: int) -> tuple[int, int]:
    left = (index - 1) % 4
    right = (index + 1) % 4
    return left, right


def _plan(env: RoomEnv, station_in_view: bool) -> list[int]:
    """Actions from the current pose to the south neighbor, then the facing switch."""
    goal_pos = (STATION_POS[0], STATION_POS[1] + 1)
    goal_facing = 0 if station_in_view else 2  # north toward the station, or south away
    start = (env.pos, env.facing_index)
    queue: deque[tuple[tuple[int, int], int]] = deque([start])
    parent: dict[tuple[tuple[int, int], int], tuple[tuple[tuple[int, int], int], int]] = {}
    seen = {start}
    found: tuple[tuple[int, int], int] | None = None
    while queue:
        state = queue.popleft()
        if state[0] == goal_pos and state[1] == goal_facing:
            found = state
            break
        pos, facing = state
        for action in (TURN_LEFT, TURN_RIGHT, FORWARD):
            nxt_pos, nxt_facing = pos, facing
            if action == TURN_LEFT:
                nxt_facing = _neighbors(facing)[0]
            elif action == TURN_RIGHT:
                nxt_facing = _neighbors(facing)[1]
            else:
                delta = FACINGS[facing]
                candidate = (pos[0] + delta[0], pos[1] + delta[1])
                # Probe with a throwaway step only when the candidate is the
                # floor. The live env is not moved; walkability matches RoomEnv.
                if env.material_at(candidate) == MATERIAL_FLOOR:
                    nxt_pos = candidate
                else:
                    continue
            nxt = (nxt_pos, nxt_facing)
            if nxt in seen:
                continue
            seen.add(nxt)
            parent[nxt] = (state, action)
            queue.append(nxt)
    if found is None:
        raise RuntimeError("room teacher found no path to the station")
    actions: list[int] = []
    cursor: tuple[tuple[int, int], int] | None = found
    while cursor != start:
        previous, action = parent[cursor]
        actions.append(action)
        cursor = previous
    actions.reverse()
    return actions


class RoomTeacher:
    """One demonstration. Call `reset` at every episode start, then `act`."""

    def __init__(self, station_in_view: bool, hold_steps: int = 48) -> None:
        self.station_in_view = bool(station_in_view)
        self.hold_steps = int(hold_steps)
        if self.hold_steps < 1:
            raise ValueError(f"hold_steps must be >= 1, got {self.hold_steps}")
        self._actions: list[int] = []
        self._index = 0

    def reset(self, env: RoomEnv) -> None:
        walk = _plan(env, self.station_in_view)
        self._actions = walk + [NOOP] * self.hold_steps + [USE]
        self._index = 0

    def act(self, env: RoomEnv) -> int:
        del env
        if self._index >= len(self._actions):
            return NOOP
        action = self._actions[self._index]
        self._index += 1
        return action
