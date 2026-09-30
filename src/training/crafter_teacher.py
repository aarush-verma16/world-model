"""Map-aware Crafter teacher that writes the early tech tree into replay.

The learned world model only predicts transitions that actually occur.
M18 at 204k had stone in 3 of 1245 lives (finding 45), so no facing head
can learn mining from that buffer. This teacher reads Crafter's own grid
(the same `_world` the legality info already comes from) and walks a fixed
chain: 4 wood, place table, wood sword, wood pickaxe, mine one stone.

It is a data source, not the policy. Eval never calls it. Its episodes are
not the reported Crafter score. Default is off; M19 turns it on with
`teacher_episodes`.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

import crafter.constants as crafter_constants

from training.crafter_rules import ACTION_INDEX, inventory_vector, spatial_from_info
from training.replay_buffer import ReplayBuffer, _pack_spatial

_DIRS: tuple[tuple[tuple[int, int], str], ...] = (
    ((1, 0), "move_right"),
    ((-1, 0), "move_left"),
    ((0, 1), "move_down"),
    ((0, -1), "move_up"),
)
_DIR_NAME = {delta: name for delta, name in _DIRS}
_WALKABLE = set(crafter_constants.walkable)
_PLACE_WHERE = ("grass", "sand", "path")


def crafter_core(env: Any) -> Any:
    """Return the raw `crafter.Env` under Gymnasium wrappers."""
    node = env
    for _ in range(8):
        if hasattr(node, "_player") and hasattr(node, "_world"):
            return node
        inner = getattr(node, "_env", None)
        if inner is not None and hasattr(inner, "_player"):
            return inner
        nxt = getattr(node, "env", None)
        if nxt is None or nxt is node:
            break
        node = nxt
    raise RuntimeError(f"expected a Crafter env, got {type(env).__name__}")


def _pos(obj: Any) -> tuple[int, int]:
    return int(obj.pos[0]), int(obj.pos[1])


def _facing_cell(player: Any) -> tuple[int, int]:
    x, y = _pos(player)
    return x + int(player.facing[0]), y + int(player.facing[1])


def _move_toward(src: tuple[int, int], dst: tuple[int, int]) -> int:
    delta = (dst[0] - src[0], dst[1] - src[1])
    name = _DIR_NAME.get(delta, "noop")
    return ACTION_INDEX[name]


def _hostile_adjacent(player: Any) -> tuple[int, int] | None:
    px, py = _pos(player)
    for obj in player.world.objects:
        if obj.__class__.__name__ not in ("Zombie", "Skeleton"):
            continue
        ox, oy = _pos(obj)
        if abs(ox - px) + abs(oy - py) == 1:
            return ox, oy
    return None


def _nearest(origin: tuple[int, int], cells: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not cells:
        return None
    return min(cells, key=lambda c: abs(c[0] - origin[0]) + abs(c[1] - origin[1]))


def _material_cells(world: Any, names: set[str]) -> list[tuple[int, int]]:
    ids = [world._mat_ids[name] for name in names if name in world._mat_ids]
    if not ids:
        return []
    xs, ys = np.where(np.isin(world._mat_map, np.asarray(ids, dtype=np.uint8)))
    return list(zip((int(x) for x in xs), (int(y) for y in ys)))


def _stand_next_to(world: Any, targets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Walkable free cells adjacent to any target. The player can stand here."""
    stands: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for tx, ty in targets:
        for (dx, dy), _name in _DIRS:
            cell = (tx + dx, ty + dy)
            if cell in seen:
                continue
            material, obj = world[cell]
            if obj is None and material in _WALKABLE:
                seen.add(cell)
                stands.append(cell)
    return stands


def _bfs_step(world: Any, start: tuple[int, int], goals: set[tuple[int, int]]) -> tuple[int, int] | None:
    """First step from `start` along a shortest path to any cell in `goals`."""
    if start in goals or not goals:
        return None
    queue: deque[tuple[int, int]] = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        cur = queue.popleft()
        for (dx, dy), _name in _DIRS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in prev:
                continue
            if nxt not in goals:
                material, obj = world[nxt]
                if obj is not None or material not in _WALKABLE:
                    continue
            prev[nxt] = cur
            if nxt in goals:
                node = nxt
                while prev[node] != start:
                    parent = prev[node]
                    if parent is None:
                        return None
                    node = parent
                return node
            queue.append(nxt)
    return None


def _act_on_adjacent(player: Any, targets: list[tuple[int, int]], action: str) -> int | None:
    """Face an adjacent target and then take `action`. `None` if none are adjacent."""
    here = _pos(player)
    adjacent = [cell for cell in targets if abs(cell[0] - here[0]) + abs(cell[1] - here[1]) == 1]
    if not adjacent:
        return None
    faced = _facing_cell(player)
    if faced in adjacent:
        return ACTION_INDEX[action]
    return _move_toward(here, adjacent[0])


def _walk_to_material(player: Any, names: set[str], action: str) -> int:
    world = player.world
    targets = _material_cells(world, names)
    acted = _act_on_adjacent(player, targets, action)
    if acted is not None:
        return acted
    stands = set(_stand_next_to(world, targets))
    step = _bfs_step(world, _pos(player), stands)
    if step is None:
        return ACTION_INDEX["noop"]
    return _move_toward(_pos(player), step)


def _goal(player: Any) -> str:
    inv = player.inventory
    placed = int(player.achievements.get("place_table", 0)) > 0
    if int(inv["wood"]) < 4 and not placed:
        return "wood"
    if not placed:
        return "table"
    if int(inv["wood_sword"]) < 1 and int(inv["wood"]) >= 1:
        return "sword"
    if int(inv["wood_pickaxe"]) < 1 and int(inv["wood"]) >= 1:
        return "pickaxe"
    if int(inv["wood_pickaxe"]) >= 1 and int(player.achievements.get("collect_stone", 0)) < 1:
        return "stone"
    return "done"


def teacher_action(env: Any) -> int:
    """One legal action toward wood, table, sword, pickaxe, then stone.

    `do` is chosen only when the faced tile is the resource or a hostile.
    Crafts are chosen only when the recipe's nearby table is already there.
    """
    core = crafter_core(env)
    player = core._player
    hostile = _hostile_adjacent(player)
    if hostile is not None:
        if _facing_cell(player) == hostile:
            return ACTION_INDEX["do"]
        return _move_toward(_pos(player), hostile)

    goal = _goal(player)
    if goal == "wood":
        return _walk_to_material(player, {"tree"}, "do")
    if goal == "table":
        world = player.world
        targets = [
            cell
            for cell in _material_cells(world, set(_PLACE_WHERE))
            if world[cell][1] is None
        ]
        acted = _act_on_adjacent(player, targets, "place_table")
        if acted is not None:
            return acted
        stands = set(_stand_next_to(world, targets))
        step = _bfs_step(world, _pos(player), stands)
        if step is None:
            return ACTION_INDEX["noop"]
        return _move_toward(_pos(player), step)
    if goal in ("sword", "pickaxe"):
        mats, _objs = player.world.nearby(player.pos, 1)
        if "table" in mats:
            name = "make_wood_sword" if goal == "sword" else "make_wood_pickaxe"
            return ACTION_INDEX[name]
        return _walk_to_material(player, {"table"}, "noop")
    if goal == "stone":
        return _walk_to_material(player, {"stone"}, "do")
    return ACTION_INDEX["noop"]


def seed_teacher_episodes(
    env: Any,
    buffer: ReplayBuffer,
    *,
    episodes: int,
    max_episode_steps: int,
    seed: int = 0,
) -> dict[str, int]:
    """Append teacher episodes to `buffer`. Returns unlock counts.

    Stops an episode a few steps after the first stone, so the replay holds
    the chain rather than a long tail of no-ops. Failures are still stored:
    a life that only got wood is real data, and the summary says so.
    """
    n = int(episodes)
    cap = int(max_episode_steps)
    if n <= 0:
        return {"episodes": 0, "stone": 0, "pickaxe": 0, "table": 0, "steps": 0}
    stone = pickaxe = table = steps = 0
    for ep in range(n):
        obs, info = env.reset(seed=int(seed) + ep)
        obs_buf: list[np.ndarray] = []
        act_buf: list[int] = []
        rew_buf: list[float] = []
        cont_buf: list[float] = []
        inv_buf: list[np.ndarray] = []
        spatial_buf: list[tuple[int, float, np.ndarray] | None] = []
        got_stone_at: int | None = None
        for _ in range(cap):
            info_d = info if isinstance(info, dict) else None
            action = teacher_action(env)
            obs_buf.append(np.asarray(obs, dtype=np.uint8))
            act_buf.append(int(action))
            inv_buf.append(inventory_vector(info_d.get("inventory") if info_d else None))
            spatial_buf.append(spatial_from_info(info_d))
            obs, reward, terminated, truncated, info = env.step(action)
            terminated = bool(terminated)
            truncated = bool(truncated)
            rew_buf.append(float(reward))
            cont_buf.append(0.0 if terminated else 1.0)
            ach = info.get("achievements", {}) if isinstance(info, dict) else {}
            if got_stone_at is None and int(ach.get("collect_stone", 0)) > 0:
                got_stone_at = len(obs_buf)
            if terminated or truncated:
                break
            if got_stone_at is not None and len(obs_buf) >= got_stone_at + 4:
                break
        if not obs_buf:
            continue
        buffer.add_episode(
            obs_buf,
            act_buf,
            rew_buf,
            cont_buf,
            inventory=inv_buf,
            spatial=_pack_spatial(spatial_buf),
            teacher=True,
        )
        steps += len(obs_buf)
        ach = info.get("achievements", {}) if isinstance(info, dict) else {}
        stone += int(int(ach.get("collect_stone", 0)) > 0)
        pickaxe += int(int(ach.get("make_wood_pickaxe", 0)) > 0)
        table += int(int(ach.get("place_table", 0)) > 0)
        print(
            f"teacher {ep + 1}/{n} len={len(obs_buf)} "
            f"table={int(ach.get('place_table', 0))} "
            f"pickaxe={int(ach.get('make_wood_pickaxe', 0))} "
            f"stone={int(ach.get('collect_stone', 0))}",
            flush=True,
        )
    print(
        f"teacher done episodes={n} steps={steps} "
        f"table={table} pickaxe={pickaxe} stone={stone}",
        flush=True,
    )
    return {
        "episodes": n,
        "stone": stone,
        "pickaxe": pickaxe,
        "table": table,
        "steps": steps,
    }
