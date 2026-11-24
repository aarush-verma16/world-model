"""A room whose image is the crop a field head can name.

The observation is a 64×64 view of the four-by-four tiles in front of the
agent. Each tile is 16×16 pixels, the size of one cell in a stride-16
encoder over this image. The station is one world cell. `use` succeeds
from any orthogonal neighbor; facing does not matter. Facing decides
whether that cell is inside the crop:

- facing the station: the orange tile is in the image
- facing away: the station is adjacent and outside the crop, so the image
  is floor and wall

`info["local_materials"]` is that same 4×4 crop, row-major, nearest row
first, shape `[16]` uint8. Codes: 0 outside, 1 floor, 2 wall, 3 station.
`info["local_objects"]` is `[16]` uint8, 0 except 1 on a station cell.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# North, east, south, west.
FACINGS: tuple[tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))

ACTION_NAMES: tuple[str, ...] = ("noop", "turn_left", "turn_right", "forward", "use")
ACTION_INDEX: dict[str, int] = {name: i for i, name in enumerate(ACTION_NAMES)}
NOOP = ACTION_INDEX["noop"]
TURN_LEFT = ACTION_INDEX["turn_left"]
TURN_RIGHT = ACTION_INDEX["turn_right"]
FORWARD = ACTION_INDEX["forward"]
USE = ACTION_INDEX["use"]

VIEW_DEPTH = 4
VIEW_WIDTH = 4
TILE = 16
OBS_SIZE = VIEW_DEPTH * TILE

WORLD_W = 9
WORLD_H = 9
STATION_POS = (4, 2)
START_POS = (4, 6)
START_FACING = 0  # north, toward the station

MATERIAL_OUTSIDE = 0
MATERIAL_FLOOR = 1
MATERIAL_WALL = 2
MATERIAL_STATION = 3
OBJECT_NONE = 0
OBJECT_STATION = 1

_COLOR = {
    MATERIAL_OUTSIDE: (8, 8, 12),
    MATERIAL_FLOOR: (34, 120, 48),
    MATERIAL_WALL: (70, 74, 82),
    MATERIAL_STATION: (230, 140, 20),
}


def right_of(facing: tuple[int, int]) -> tuple[int, int]:
    """90° clockwise. North becomes east."""
    fx, fy = facing
    return (-fy, fx)


def view_cell(pos: tuple[int, int], facing: tuple[int, int], row: int, col: int) -> tuple[int, int]:
    """World cell for view row `row` (0 = nearest) and column `col`.

    Column 1 is straight ahead. Columns run to the agent's right.
    """
    fx, fy = facing
    rx, ry = right_of(facing)
    depth = row + 1
    lateral = col - 1
    return (pos[0] + depth * fx + lateral * rx, pos[1] + depth * fy + lateral * ry)


class RoomEnv:
    """One room, one station, one successful `use` per episode.

    Observation: uint8 `[64, 64, 3]`. Five actions, `ACTION_NAMES`.
    Reward is 1 on the successful use and 0 otherwise. Success terminates.
    The step cap truncates, with `info["discount"]` 1. A death-style success
    sets `discount` to 0.
    """

    def __init__(self, max_steps: int = 160, require_station_in_view: bool | None = None) -> None:
        self.max_steps = int(max_steps)
        if self.max_steps < 2:
            raise ValueError(f"max_steps must be >= 2, got {self.max_steps}")
        # None: any adjacent use scores. True/False: the crop at that step
        # has to match, so the out-of-view run cannot score by turning around.
        self.require_station_in_view = require_station_in_view
        self._pos = START_POS
        self._facing = START_FACING
        self._used = False
        self._steps = 0

    @property
    def action_dim(self) -> int:
        return len(ACTION_NAMES)

    @property
    def pos(self) -> tuple[int, int]:
        return self._pos

    @property
    def facing(self) -> tuple[int, int]:
        return FACINGS[self._facing]

    @property
    def facing_index(self) -> int:
        return self._facing

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        del seed  # the room is fixed; the argument matches Gymnasium's reset
        self._pos = START_POS
        self._facing = START_FACING
        self._used = False
        self._steps = 0
        return self._obs(), self._info(discount=1.0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Return `(obs [64,64,3] uint8, reward, terminated, truncated, info)`."""
        action = int(action)
        if action < 0 or action >= len(ACTION_NAMES):
            raise ValueError(f"action must be in 0..{len(ACTION_NAMES) - 1}, got {action}")
        reward = 0.0
        terminated = False
        if not self._used:
            if action == TURN_LEFT:
                self._facing = (self._facing - 1) % 4
            elif action == TURN_RIGHT:
                self._facing = (self._facing + 1) % 4
            elif action == FORWARD:
                nxt = (self._pos[0] + self.facing[0], self._pos[1] + self.facing[1])
                if self.material_at(nxt) == MATERIAL_FLOOR:
                    self._pos = nxt
            elif action == USE and self._use_succeeds():
                self._used = True
                reward = 1.0
                terminated = True
        self._steps += 1
        truncated = (not terminated) and self._steps >= self.max_steps
        discount = 0.0 if terminated else 1.0
        return self._obs(), reward, terminated, truncated, self._info(discount=discount)

    def station_in_view(self) -> bool:
        return MATERIAL_STATION in self.view_materials()

    def view_materials(self) -> np.ndarray:
        """`[4, 4]` uint8. Row 0 is nearest. Column 1 is straight ahead."""
        grid = np.zeros((VIEW_DEPTH, VIEW_WIDTH), dtype=np.uint8)
        for row in range(VIEW_DEPTH):
            for col in range(VIEW_WIDTH):
                grid[row, col] = self.material_at(view_cell(self._pos, self.facing, row, col))
        return grid

    def material_at(self, pos: tuple[int, int]) -> int:
        """Material code at a world cell. Outside the map is 0."""
        x, y = pos
        if x < 0 or y < 0 or x >= WORLD_W or y >= WORLD_H:
            return MATERIAL_OUTSIDE
        if x == 0 or y == 0 or x == WORLD_W - 1 or y == WORLD_H - 1:
            return MATERIAL_WALL
        if (x, y) == STATION_POS:
            return MATERIAL_STATION
        return MATERIAL_FLOOR

    def _use_succeeds(self) -> bool:
        if not self._adjacent_to_station():
            return False
        if self.require_station_in_view is None:
            return True
        return self.station_in_view() is self.require_station_in_view

    def _adjacent_to_station(self) -> bool:
        sx, sy = STATION_POS
        return abs(self._pos[0] - sx) + abs(self._pos[1] - sy) == 1

    def _obs(self) -> np.ndarray:
        """`[64, 64, 3]` uint8. The nearest view row is the bottom 16 pixels."""
        grid = self.view_materials()
        image = np.zeros((OBS_SIZE, OBS_SIZE, 3), dtype=np.uint8)
        for row in range(VIEW_DEPTH):
            y0 = (VIEW_DEPTH - 1 - row) * TILE
            for col in range(VIEW_WIDTH):
                x0 = col * TILE
                image[y0 : y0 + TILE, x0 : x0 + TILE] = _COLOR[int(grid[row, col])]
        return image

    def _info(self, *, discount: float) -> dict[str, Any]:
        grid = self.view_materials()
        flat = grid.reshape(-1)
        objects = np.zeros_like(flat)
        objects[flat == MATERIAL_STATION] = OBJECT_STATION
        return {
            "local_materials": flat.copy(),
            "local_objects": objects,
            "station_in_view": bool(self.station_in_view()),
            "used": int(self._used),
            "pos": self._pos,
            "facing": self.facing,
            "discount": float(discount),
        }
