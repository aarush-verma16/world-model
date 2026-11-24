# The room station is one 16×16 tile, in the crop or one cell behind the agent

**Claim.** `RoomEnv` shows the agent the 4×4 tiles in front of it, each tile 16×16 pixels, as a 64×64 image. From the cell south of the station, facing it puts that tile in the crop and facing away leaves it outside the image. The training env only scores the pose the config asked for. Both teachers walk the same three steps, then one of them turns around. A 48-step hold follows, then `use`.

## Why this is not the paper

No world model result lives in this note. The four runs were trained on 2026-11-17; the read is finding 63. A Crafter checkpoint cannot load into this action set.

## Evidence

`src/envs/room_env.py`, `src/envs/room_teacher.py`, `tests/test_room_env.py`. From the south neighbor facing north, the image contains 256 station-colored pixels, one 16×16 block at the bottom, second column, and `local_materials[1]` is the station code. Facing south from that same cell, the image contains 0 station pixels. With no pose requirement, `use` still returns reward 1. With `require_station_in_view` set, the other facing returns reward 0. Across the last 32 steps of each teacher, the crop matches the switch on every frame. The action lists share the prefix `forward, forward, forward`. The away teacher then turns left twice. Both tails are 48 no-ops and `use`.

## Failed alternatives

A centered window. Every adjacent cell would sit inside it, so facing away could not put the station outside the crop. A backward walk that never shows the station at all. The shared three-step approach keeps the walk the same; the anchored window is what drops that approach.

## Paper spin

The second setting's geometry. The station is large enough to occupy one cell of the image when it is in the crop, and the rare action is one step after a hold long enough for a 32-step window to see only the pose.
