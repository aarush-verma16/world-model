# Fieldcraft paper spine

This file locks the object of the paper and the order of the sections. It is not the draft. No citation in a later draft is invented here.

## Object

Fieldcraft is a Dreamer world model. Its latent state is trained to carry a local symbolic field, and its imagined rollouts read that field. On the benchmark the field names the terrain and does not name the station, and the actor's rare action stays at two lives whether or not the loss window is anchored. On the room the station is one encoder cell, the field head names it, and `use` is learned in every cell of that same 2×2, including when the station is behind the agent and when the window is an ordinary slice. Real steps still give the actor the environment's field.

The survival game used so far is one benchmark. The method section does not depend on that game's items. The score of a run is an experiment, not the subject.

## The model

- Encoder, then an RSSM. `z_posterior` comes from the deterministic state `h` and the encoder embedding. `z_prior` comes from `h` alone. One decoder reads `concat(h, z)` and nothing else. A second decoder off the embedding empties the latent while the pixel loss still falls.
- Heads on the same features: reward, continuation, inventory, the faced tile, and a 9×7 field with a material and an object class in every cell. The field head exists because a latent cell at this resolution is larger than the tile an action uses, so pixel reconstruction does not name that tile.
- The actor-critic trains on imagined rollouts. Imagination has no environment state, so it reads the field decoded from `z_prior`.
- On the runs in hand, a real step gives the actor the environment's field. Imagination gives it the predicted field. The paper states that split. Real execution in those runs is not "pixels in, field guessed by the model, action out."

## Sections

About nine pages. References sit after the nine.

1. **Introduction.** The model. What the two switches change on the benchmark, and what they do not change in the room.
2. **Related work.** World models and imagination. Privileged information used as a training target. Demonstrations inside an on-policy loop. Local maps as a representation the latent has to carry.
3. **Method.** The RSSM, the single decoder, the field head, the loss window, and demonstrations used to put the relevant tile inside the field. No item from the benchmark in this section.
4. **What the latent contains.** Reconstruction through `z_posterior`. The bypass result. Per-cell accuracy of the field head. Open-loop rollouts from `z_prior`, including whether the relevant tile is still named across the imagination horizon.
5. **The two switches.** One benchmark, three matched runs. The loss window on or off. The relevant tile inside the field, or walked outside the crop the field covers.
6. **The same model in a second setting.** A small room. The station is inside the crop the field head represents, or outside it. The rare action is anchored in the loss window, or averaged over the episode.
7. **Discussion.** What the latent has to contain. What the benchmark runs do and do not show.

## Evidence already measured

- A reconstruction path that skips `z_posterior` drives pixel loss down while the latent stays empty.
- At this encoder resolution, one latent cell covers more than the tile the action uses.
- A uniform slice of a long demonstration can miss the rare action while the cloning loss falls. Forcing that action into the window, over a full run, left the unaided count unchanged.
- With the tile inside the crop, the demonstrator finished the rare action on 23 of 60 handoffs and on 44 of 121. With the tile walked outside the crop, it finished on 23 of 114. The actor's own count was two lives in every cell. A held-out execution of that action was 0 of 10 in every cell.
- The field head, read from the mode of `z_posterior` on eight teacher lives (116,235 cells), names materials at 72% of cells. The grass baseline is 54%. Grass probability is 0.93. Stone and path are about 0.52. Table is 0.13–0.16. Coal, iron, and furnace are about 0.03, 0.01, and 0.01–0.06. The three 100k checkpoints agree. A mode rollout of `z_prior` is about 70% at step 1 and 56–60% at step 15, against the 54% grass floor. The logged field loss near 0.02 is that average, and most cells are grass. Objects are 98% empty, so object accuracy 97.6% is the empty class.
- The room, rerun on 2026-11-23 with 8/8 real teacher episodes in every cell. Actor `use` is 124/147, 85/110, 185/190, and 130/138. Eval at about 2k, 4k, 6k, and 8k is 7/8, 7/8, 8/8, 8/8; then 5/8, 4/8, 6/8, 8/8; then 8/8 four times; then 7/8, 8/8, 8/8, 8/8. Every success has the pose that cell requires. The posterior names the station cell on both poses (recall 1, false station 0). A 15-step `z_prior` rollout matches the pose the run trained. Anchored success lengths cluster near 36 and 38 steps. Uniform lengths spread from the teens to past the teacher's 52. The 2026-11-17 counts are not this comparison.

## Then the second setting

The room is `envs.room_env.RoomEnv`. The image is the 4×4 tiles in front of the agent, each tile 16×16 pixels, so a station fills one cell of a stride-16 encoder. `use` succeeds from the neighbor south of the station whether or not the agent is facing it. `RoomTeacher(station_in_view=True)` holds that facing, so the station is in the crop for the whole held pose. `station_in_view=False` turns around first, so the station is one cell behind the agent and absent from the image. Both walks share the three steps up to that cell. The hold is 48 no-ops, then `use`. An anchored window of length 32 ends on `use` and contains only that pose. A uniform window of the same length usually misses it.

Four runs, same small world model, no Crafter checkpoint: station in the crop or behind the agent, and the loss window anchored on `use` or not. `RoomEnv(require_station_in_view=...)` only scores the matching pose, so the out-of-view run cannot get the reward by turning to face the station. The trainer is `training.room_loop.train_room`, launched from `notebooks/18_train_room.ipynb`, one config at a time. The Crafter loop stays on its own action set. The 2026-11-23 rerun is the action comparison. Teacher episodes are 8/8. `use` is learned in every cell, including with the station behind the agent and with an ordinary window. The window changes the length of the successful episode. Real eval steps are given the environment's field.

## Drafting rule

The field-head measurements are in, on the benchmark and on the room, and the room action comparison is in. The model sections and the room section can be drafted against those counts. Citations are taken from the papers, with page or section, while the draft is written.

## Claims the draft will not make

The acting policy in the current runs reads the environment's field on real steps. The unaided count on the benchmark was two with the tile inside the crop and two with it outside. A held-out execution of that action was absent in every benchmark cell. The room's 8/8 is a different, short action, and it is not that benchmark result. Those sentences stay in the experiments section.
