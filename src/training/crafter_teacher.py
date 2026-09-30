"""Map-aware Crafter teacher that writes the early tech tree into replay.

The learned world model only predicts transitions that actually occur.
M18 at 204k had stone in 3 of 1245 lives (finding 45), so no facing head
can learn mining from that buffer. This teacher reads Crafter's own grid
(the same `_world` the legality info already comes from) and walks a fixed
chain: 4 wood, place table, wood sword, wood pickaxe, mine one stone.

It is a data source, not the policy. Eval never calls it. Its episodes are
not the reported Crafter score. Default is off; M19 turns it on with
`teacher_episodes`.

`CrafterTeacherV2` (finding 48, M20) keeps the same grid access but plays
whole lives: it drinks, eats (cow or its own ripe plant), sleeps, fights,
and walks the full recipe chain read from `crafter.constants` — table,
sapling/plant, wood tools, stone, place_stone, stone tools, furnace, coal,
iron, iron tools, diamond — tunnelling through stone once it has a pickaxe.
Its actions also become a behavior-cloning target (`ac_step.bc_scale`).
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

import crafter.constants as crafter_constants

from training.crafter_rules import (
    ACTION_INDEX,
    inventory_vector,
    local_map_from_info,
    spatial_from_info,
)
from training.replay_buffer import ReplayBuffer, _pack_local, _pack_spatial

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


# --------------------------------------------------------------------------
# Teacher v2 (finding 48)
# --------------------------------------------------------------------------

_ITEM_SOURCE = {
    "wood": "tree",
    "stone": "stone",
    "coal": "coal",
    "iron": "iron",
    "diamond": "diamond",
    "sapling": "grass",
}
_TOOLS = (
    "wood_pickaxe",
    "wood_sword",
    "stone_pickaxe",
    "stone_sword",
    "iron_pickaxe",
    "iron_sword",
)
_HOSTILES = ("Zombie", "Skeleton")


def _objects_named(world: Any, names: tuple[str, ...]) -> list[Any]:
    return [obj for obj in world.objects if obj.__class__.__name__ in names]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _free(world: Any, cell: tuple[int, int]) -> bool:
    material, obj = world[cell]
    return obj is None and material in _WALKABLE


def _bfs(
    world: Any,
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    *,
    tunnel: bool,
    max_nodes: int = 4096,
) -> tuple[tuple[int, int] | None, int]:
    """`(first step, path length)` to any goal; `(None, -1)` if unreachable.

    Walkable free cells are passable. With `tunnel`, stone is passable too
    (the step onto it becomes `do` then a move). Lava is never passable.
    Goal cells are accepted whatever their material.
    """
    if not goals:
        return None, -1
    if start in goals:
        return None, 0
    queue: deque[tuple[int, int]] = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    dist = {start: 0}
    while queue and len(prev) < max_nodes:
        cur = queue.popleft()
        for (dx, dy), _name in _DIRS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in prev:
                continue
            if nxt not in goals:
                material, obj = world[nxt]
                if material is None or obj is not None:
                    continue
                if material not in _WALKABLE and not (tunnel and material == "stone"):
                    continue
            prev[nxt] = cur
            dist[nxt] = dist[cur] + 1
            if nxt in goals:
                node = nxt
                while prev[node] != start:
                    parent = prev[node]
                    if parent is None:
                        return None, -1
                    node = parent
                return node, dist[nxt]
            queue.append(nxt)
    return None, -1


def _dilate(mask: np.ndarray) -> np.ndarray:
    """3x3 binary dilation of a `[W, H]` bool map (Crafter's `nearby(pos, 1)`)."""
    padded = np.pad(mask, 1)
    out = np.zeros_like(mask)
    w, h = mask.shape
    for dx in range(3):
        for dy in range(3):
            out |= padded[dx : dx + w, dy : dy + h]
    return out


class CrafterTeacherV2:
    """Stateful whole-life Crafter teacher (one instance per env stream).

    Call `reset()` at every episode start and `act(env)` each step. The
    returned action is always one Crafter can execute or harmlessly no-op;
    it never moves onto lava.
    """

    def __init__(self, seed: int = 0, goal_timeout: int = 600) -> None:
        self._rng = np.random.default_rng(int(seed))
        self.goal_timeout = int(goal_timeout)
        self.reset()

    def reset(self) -> None:
        self.plant_pos: tuple[int, int] | None = None
        self.skip: set[str] = set()
        self.goal_key: str | None = None
        self.goal_steps = 0
        self.mode: str | None = None
        self.last_pos: tuple[int, int] | None = None
        self.still_moves = 0
        self.idle_steps = 0

    # ---------------------------------------------------------------- moves
    def _safe_move(self, world: Any, here: tuple[int, int], dst: tuple[int, int]) -> int:
        if world[dst][0] == "lava":
            return ACTION_INDEX["noop"]
        return _move_toward(here, dst)

    def _explore(self, world: Any, here: tuple[int, int]) -> int:
        options = [
            name
            for (dx, dy), name in _DIRS
            if _free(world, (here[0] + dx, here[1] + dy))
        ]
        if not options:
            return ACTION_INDEX["noop"]
        return ACTION_INDEX[options[int(self._rng.integers(len(options)))]]

    def _step_along(self, player: Any, step: tuple[int, int]) -> int:
        world = player.world
        here = _pos(player)
        if world[step][0] == "stone":
            if _facing_cell(player) == step:
                return ACTION_INDEX["do"]
            return _move_toward(here, step)
        return self._safe_move(world, here, step)

    def _go_act(
        self,
        player: Any,
        targets: list[tuple[int, int]],
        action: str,
        *,
        tunnel: bool,
    ) -> int | None:
        """Walk next to a (non-walkable) target, face it, and take `action`.

        `None` if no target is reachable.
        """
        if not targets:
            return None
        world = player.world
        here = _pos(player)
        adjacent = [cell for cell in targets if _manhattan(cell, here) == 1]
        if adjacent:
            if _facing_cell(player) in adjacent:
                return ACTION_INDEX[action]
            return self._safe_move(world, here, adjacent[0])
        near = sorted(targets, key=lambda c: _manhattan(c, here))[:64]
        goals: set[tuple[int, int]] = set()
        for tx, ty in near:
            for (dx, dy), _name in _DIRS:
                cell = (tx + dx, ty + dy)
                material, obj = world[cell]
                if obj is None and (
                    material in _WALKABLE or (tunnel and material == "stone")
                ):
                    goals.add(cell)
        step, _dist = _bfs(world, here, goals, tunnel=tunnel)
        if step is None:
            return None
        return self._step_along(player, step)

    def _place_facing(self, player: Any, name: str, anchors: tuple[str, ...] = ()) -> int:
        """Place `name` on the faced tile, moving until the faced tile is valid.

        With `anchors`, first stand where every anchor is in `nearby(pos, 1)`
        so the placed utility ends up next to them.
        """
        world = player.world
        here = _pos(player)
        if anchors:
            nearby, _ = world.nearby(player.pos, 1)
            if not all(a in nearby for a in anchors):
                act = self._go_nearby(player, anchors)
                return act if act is not None else self._explore(world, here)
        info = crafter_constants.place[name]
        faced = _facing_cell(player)
        material, obj = world[faced]
        if obj is None and material in info["where"]:
            return ACTION_INDEX[f"place_{name}"]
        return self._explore(world, here)

    def _go_nearby(self, player: Any, utils: tuple[str, ...]) -> int | None:
        """Walk to a free cell whose 3x3 holds every material in `utils`."""
        world = player.world
        mask = np.ones(world._mat_map.shape, dtype=bool)
        for util in utils:
            if util not in world._mat_ids:
                return None
            mask &= _dilate(world._mat_map == world._mat_ids[util])
        walk_ids = [world._mat_ids[n] for n in _WALKABLE if n in world._mat_ids]
        mask &= np.isin(world._mat_map, np.asarray(walk_ids, dtype=np.uint8))
        xs, ys = np.where(mask)
        goals = {(int(x), int(y)) for x, y in zip(xs, ys) if world[(int(x), int(y))][1] is None}
        if not goals:
            return None
        here = _pos(player)
        step, dist = _bfs(world, here, goals, tunnel=int(player.inventory["wood_pickaxe"]) > 0)
        if step is None:
            return None
        self._last_nearby_dist = dist
        return self._step_along(player, step)

    # -------------------------------------------------------------- goals
    def _collect(self, player: Any, material: str) -> int | None:
        world = player.world
        tunnel = int(player.inventory["wood_pickaxe"]) > 0
        if material == "grass":
            faced = _facing_cell(player)
            if world[faced][0] == "grass" and world[faced][1] is None:
                return ACTION_INDEX["do"]
            return self._explore(world, _pos(player))
        return self._go_act(player, _material_cells(world, {material}), "do", tunnel=tunnel)

    def _craft(self, player: Any, tool: str) -> int | None:
        info = crafter_constants.make[tool]
        utils = tuple(info["nearby"])
        nearby, _ = player.world.nearby(player.pos, 1)
        if all(u in nearby for u in utils):
            return ACTION_INDEX[f"make_{tool}"]
        self._last_nearby_dist = -1
        act = self._go_nearby(player, utils)
        far = act is None or self._last_nearby_dist > 25
        if far:
            for util in utils:
                if util in nearby:
                    continue
                uses = crafter_constants.place[util]["uses"]
                spare = {
                    k: int(player.inventory[k]) - int(v) - int(info["uses"].get(k, 0))
                    for k, v in uses.items()
                }
                if all(v >= 0 for v in spare.values()):
                    return self._place_facing(player, util)
        return act

    def _lacking(self, player: Any, uses: dict[str, int]) -> str | None:
        for item, amount in uses.items():
            if int(player.inventory[item]) < int(amount):
                return item
        return None

    def _stone_need(self, player: Any) -> int:
        ach = player.achievements
        need = 0
        if int(ach.get("place_stone", 0)) == 0:
            need += int(crafter_constants.place["stone"]["uses"]["stone"])
        for tool in ("stone_pickaxe", "stone_sword"):
            if int(player.inventory[tool]) < 1:
                need += int(crafter_constants.make[tool]["uses"].get("stone", 0))
        if int(ach.get("place_furnace", 0)) == 0:
            need += int(crafter_constants.place["furnace"]["uses"].get("stone", 0))
        return min(need, 9)

    def _chain(self, player: Any) -> tuple[str, int | None]:
        """`(goal_key, action)` for the next unfinished tech-tree step."""
        inv = player.inventory
        ach = player.achievements

        def gather(key: str, item: str) -> tuple[str, int | None]:
            sub = f"collect_{item}"
            if sub in self.skip:
                self.skip.add(key)
                return key, None
            return sub, self._collect(player, _ITEM_SOURCE[item])

        steps: list[str] = ["place_table", "sapling", "place_plant", "wood_pickaxe", "wood_sword"]
        steps += ["stone", "place_stone", "stone_pickaxe", "stone_sword", "place_furnace"]
        steps += ["coal", "iron", "iron_pickaxe", "iron_sword", "diamond"]
        for key in steps:
            if key in self.skip:
                continue
            if key == "place_table":
                if int(ach.get("place_table", 0)) > 0:
                    continue
                lack = self._lacking(player, crafter_constants.place["table"]["uses"])
                if lack is not None:
                    return gather(key, lack)
                return key, self._place_facing(player, "table")
            if key == "sapling":
                if int(ach.get("collect_sapling", 0)) > 0 or int(inv["sapling"]) > 0:
                    continue
                return key, self._collect(player, "grass")
            if key == "place_plant":
                if int(ach.get("place_plant", 0)) > 0 or int(inv["sapling"]) < 1:
                    continue
                act = self._place_facing(player, "plant")
                if act == ACTION_INDEX["place_plant"]:
                    self.plant_pos = _facing_cell(player)
                return key, act
            if key in _TOOLS:
                if int(inv[key]) > 0:
                    continue
                lack = self._lacking(player, crafter_constants.make[key]["uses"])
                if lack is not None:
                    return gather(key, lack)
                return key, self._craft(player, key)
            if key == "stone":
                if int(inv["stone"]) >= self._stone_need(player):
                    continue
                return key, self._collect(player, "stone")
            if key == "place_stone":
                if int(ach.get("place_stone", 0)) > 0:
                    continue
                if int(inv["stone"]) < 1:
                    return gather(key, "stone")
                return key, self._place_facing(player, "stone")
            if key == "place_furnace":
                if int(ach.get("place_furnace", 0)) > 0:
                    continue
                lack = self._lacking(player, crafter_constants.place["furnace"]["uses"])
                if lack is not None:
                    return gather(key, lack)
                return key, self._place_facing(player, "furnace", anchors=("table",))
            if key in ("coal", "iron"):
                need = sum(
                    int(crafter_constants.make[t]["uses"].get(key, 0))
                    for t in ("iron_pickaxe", "iron_sword")
                    if int(inv[t]) < 1
                )
                if int(inv[key]) >= need:
                    continue
                return key, self._collect(player, key)
            if key == "diamond":
                if int(ach.get("collect_diamond", 0)) > 0 or int(inv["iron_pickaxe"]) < 1:
                    continue
                return key, self._collect(player, "diamond")
        return "idle", None

    # --------------------------------------------------------------- act
    def act(self, env: Any) -> int:
        """One action for the current `crafter.Env` state."""
        core = crafter_core(env)
        player = core._player
        world = player.world
        here = _pos(player)
        inv = player.inventory
        ach = player.achievements

        if player.sleeping:
            return ACTION_INDEX["sleep"]

        hostile = _hostile_adjacent(player)
        if hostile is not None:
            if _facing_cell(player) == hostile:
                return ACTION_INDEX["do"]
            return _move_toward(here, hostile)

        hostiles = [_pos(o) for o in _objects_named(world, _HOSTILES)]
        near_hostile = min((_manhattan(here, h) for h in hostiles), default=99)

        # Survival first: these keep the life long enough for the chain.
        if int(inv["drink"]) <= 3 or (self.mode == "drink" and int(inv["drink"]) < 9):
            self.mode = "drink"
            act = self._go_act(player, _material_cells(world, {"water"}), "do", tunnel=False)
            if act is not None:
                return act
            self.mode = None
        elif self.mode == "drink":
            self.mode = None
        if int(inv["food"]) <= 3 or (self.mode == "eat" and int(inv["food"]) < 8):
            self.mode = "eat"
            act = self._eat(player)
            if act is not None:
                return act
            self.mode = None
        elif self.mode == "eat":
            self.mode = None
        want_sleep = int(inv["energy"]) <= 2 or (
            int(ach.get("wake_up", 0)) == 0 and int(inv["energy"]) < 9
        )
        if want_sleep and near_hostile > 6:
            return ACTION_INDEX["sleep"]

        # Opportunistic achievements.
        if self.plant_pos is not None:
            _mat, obj = world[self.plant_pos]
            if obj is None or obj.__class__.__name__ != "Plant":
                self.plant_pos = None
            elif obj.ripe and int(ach.get("eat_plant", 0)) == 0:
                act = self._go_act(player, [self.plant_pos], "do", tunnel=False)
                if act is not None:
                    return act
        if int(inv["health"]) >= 6 and near_hostile <= 5:
            wanted = [
                _pos(o)
                for o in _objects_named(world, _HOSTILES)
                if int(ach.get(f"defeat_{o.__class__.__name__.lower()}", 0)) == 0
                and _manhattan(here, _pos(o)) <= 5
            ]
            act = self._go_act(player, wanted, "do", tunnel=False)
            if act is not None:
                return act
        if int(ach.get("eat_cow", 0)) == 0 and int(inv["food"]) < 9:
            cows = [_pos(o) for o in _objects_named(world, ("Cow",)) if _manhattan(here, _pos(o)) <= 6]
            act = self._go_act(player, cows, "do", tunnel=False)
            if act is not None:
                return act

        key, act = self._chain(player)
        if key != self.goal_key:
            self.goal_key = key
            self.goal_steps = 0
        self.goal_steps += 1
        if key != "idle" and self.goal_steps > self.goal_timeout:
            self.skip.add(key)
            self.goal_key = None
        self.idle_steps = self.idle_steps + 1 if key == "idle" else 0
        if act is None:
            act = self._explore(world, here)

        # Break move loops (e.g. an object parked on the only path).
        moved = self.last_pos is None or self.last_pos != here
        self.last_pos = here
        is_move = act in (ACTION_INDEX[n] for _d, n in _DIRS)
        self.still_moves = 0 if moved or not is_move else self.still_moves + 1
        if self.still_moves >= 6:
            self.still_moves = 0
            return self._explore(world, here)
        return act

    def _eat(self, player: Any) -> int | None:
        world = player.world
        if self.plant_pos is not None:
            _mat, obj = world[self.plant_pos]
            if obj is not None and obj.__class__.__name__ == "Plant" and obj.ripe:
                act = self._go_act(player, [self.plant_pos], "do", tunnel=False)
                if act is not None:
                    return act
        cows = [_pos(o) for o in _objects_named(world, ("Cow",))]
        return self._go_act(player, cows, "do", tunnel=False)


def seed_teacher_episodes(
    env: Any,
    buffer: ReplayBuffer,
    *,
    episodes: int,
    max_episode_steps: int,
    seed: int = 0,
    version: int = 1,
    idle_stop: int = 200,
) -> dict[str, Any]:
    """Append teacher episodes to `buffer`. Returns unlock counts.

    `version=1` (M19): stops a few steps after the first stone, so the
    replay holds the chain rather than a long tail of no-ops.
    `version=2` (M20): plays the whole life with `CrafterTeacherV2` until
    death, `max_episode_steps`, or `idle_stop` steps with nothing left to do.
    Failures are still stored: a life that only got wood is real data.

    Returns:
        `episodes`, `steps`, `table`, `pickaxe`, `stone` (episode counts),
        `unlocks` (achievement -> number of episodes that got it) and
        `mean_length`.
    """
    n = int(episodes)
    cap = int(max_episode_steps)
    empty: dict[str, Any] = {
        "episodes": 0,
        "stone": 0,
        "pickaxe": 0,
        "table": 0,
        "steps": 0,
        "unlocks": {},
        "mean_length": 0.0,
    }
    if n <= 0:
        return empty
    if int(version) not in (1, 2):
        raise ValueError(f"teacher version must be 1 or 2, got {version}")
    teacher_v2 = CrafterTeacherV2(seed=int(seed)) if int(version) == 2 else None
    steps = 0
    unlocks: dict[str, int] = {name: 0 for name in crafter_constants.achievements}
    lengths: list[int] = []
    for ep in range(n):
        obs, info = env.reset(seed=int(seed) + ep)
        if teacher_v2 is not None:
            teacher_v2.reset()
        obs_buf: list[np.ndarray] = []
        act_buf: list[int] = []
        rew_buf: list[float] = []
        cont_buf: list[float] = []
        inv_buf: list[np.ndarray] = []
        spatial_buf: list[tuple[int, float, np.ndarray] | None] = []
        local_buf: list[tuple[np.ndarray, np.ndarray] | None] = []
        got_stone_at: int | None = None
        for _ in range(cap):
            info_d = info if isinstance(info, dict) else None
            action = teacher_v2.act(env) if teacher_v2 is not None else teacher_action(env)
            obs_buf.append(np.asarray(obs, dtype=np.uint8))
            act_buf.append(int(action))
            inv_buf.append(inventory_vector(info_d.get("inventory") if info_d else None))
            spatial_buf.append(spatial_from_info(info_d))
            local_buf.append(local_map_from_info(info_d))
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
            if teacher_v2 is None:
                if got_stone_at is not None and len(obs_buf) >= got_stone_at + 4:
                    break
            elif teacher_v2.idle_steps >= int(idle_stop):
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
            local=_pack_local(local_buf),
            teacher=True,
        )
        steps += len(obs_buf)
        lengths.append(len(obs_buf))
        ach = info.get("achievements", {}) if isinstance(info, dict) else {}
        for name in unlocks:
            unlocks[name] += int(int(ach.get(name, 0)) > 0)
        got = sorted(name for name in unlocks if int(ach.get(name, 0)) > 0)
        print(
            f"teacher v{int(version)} {ep + 1}/{n} len={len(obs_buf)} "
            f"unlocked={len(got)} [{', '.join(got)}]",
            flush=True,
        )
    print(
        f"teacher done episodes={n} steps={steps} "
        f"table={unlocks['place_table']} pickaxe={unlocks['make_wood_pickaxe']} "
        f"stone={unlocks['collect_stone']} iron={unlocks['collect_iron']} "
        f"diamond={unlocks['collect_diamond']}",
        flush=True,
    )
    return {
        "episodes": n,
        "stone": unlocks["collect_stone"],
        "pickaxe": unlocks["make_wood_pickaxe"],
        "table": unlocks["place_table"],
        "steps": steps,
        "unlocks": unlocks,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }
