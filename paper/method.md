# Method notes for the draft

The method section names the model and the two conditions it was trained under. It does not name items from the benchmark. The experiment section may name the benchmark, the room, the runs, and the counts in `evidence.md`.

## Model

PROSPECT is a DreamerV3 RSSM with one decoder and a field head.

- The image is 64×64 RGB.
- `z_posterior` is sampled from the deterministic state `h` and the encoder embedding of the current frame. `h` itself is computed from the previous posterior and the previous action, so it does not see the current frame.
- `z_prior` is sampled from `h` alone. Imagination rolls `z_prior` only.
- The decoder reads `concat(h, z)` and nothing else. A second decoder off the embedding empties the latent while the pixel loss still falls. That path is not in the model.
- Pixels are trained with the image MSE. The reported L1 is a log, not the trained term. The decoder's linear output adds 0.5 and is not a sigmoid.
- The latent is 32 categoricals of 32 classes in both settings. It was not shrunk.
- The field head reads the same features and predicts a material class and an object class for every cell of a local grid.
- Reward and continuation heads sit on the same features. The benchmark model also has inventory and facing heads. The room model does not.
- The actor and critic train on imagined rollouts of horizon 15 on the benchmark and horizon 8 in the room. Imagination has no environment state, so the field columns it reads are the softmax of the field head on `z_prior`.
- On every run in this paper, a real step gives the actor the environment's field (`actor_input_source: truth`). Evaluation never calls the teacher. Real execution is not "pixels in, field guessed by the model, action out."

- Benchmark only: the actor also reads inventory counts and the faced and nearby-material columns (truth on real steps, heads in imagination), and `place_*`/`make_*` are masked by legality, exact from `info` on real steps and from the predicted inventory and nearby materials in imagination (`legal_mask_from_facts_torch`). So in imagination the iron pickaxe is selectable only if the nearby head predicts table and furnace. That head was not probed. The room has no extra heads and no mask.

## The two training switches

Both settings use the same two switches.

1. Where the relevant tile sits relative to the crop the field head is trained to name. On the benchmark, inside the crop for the whole approach (placed beside the resource), or outside it for the walk back to it and inside only for the last steps, because the benchmark's craft rule needs the station within one tile, which is always inside the 9×7. In the room, inside the image at `use`, or behind the agent and absent from the image at `use`. Only the room puts the tile outside the crop at the action itself.
2. Whether the loss window is anchored so that the rare action is the last step of the sampled sequence, or the window is an ordinary slice of a teacher episode.

Teacher episodes are half the replay batch (`teacher_fraction` 0.5). Behavior cloning is action cross-entropy on steps flagged as teacher, weight 1, with the rare action up-weighted by 8 wherever it appears in the window. Entropy weight is `3e-4`. Cloning weight is 1. Evaluation does not use the teacher, and a life finished by the teacher is not counted as the actor's rare action.

## Benchmark protocol

Names in this subsection belong in the experiments, not the method.

- Environment `CrafterReward-v1`, 17 actions, local map 9×7 (63 cells), 13 material classes, 8 object classes.
- World model size XL, from `configs/sizes/dreamer_xl_paper_local.yaml`: embed 12288, channels `[96, 192, 384, 768]`, 2 residual blocks, deterministic size 4096, stochastic 32×32.
- Batch 8, sequence 64, train ratio 512, 100k environment steps, one seed.
- Three runs from the same M20 checkpoint, fresh replay: M22 anchors the craft window and the teacher places the station inside the crop; M23 is the same with the craft window off; M24 keeps the craft window and the teacher walks home, so the station is outside the 9×7 during the walk back from the ore. At the craft itself it is inside: Crafter's `make_*` checks `world.nearby(pos, 1)`, a 3×3 around the player.
- Rare-action weight: the three station crafts (`place_furnace`, `make_iron_pickaxe`, `make_iron_sword`) weigh 8 in the cloning loss wherever they appear in a teacher window, anchored or not. The anchored draw (`craft_window_fraction` 0.5) is half of the teacher rows, and it applies to world-model and actor-critic batches alike.
- The logged score on a teacher-finished life is not the actor's. `actor` counts are lives whose log has no assisted flag. Held-out eval is 10 lives and is reported as such.
- Official score, if shown at all, is the geometric mean of the 22 achievement percents. It is not the claim. Do not put it next to DreamerV3's 14.5.

## Room protocol

- `RoomEnv`: a fixed 9×9 room. The image is the 4×4 tiles in front of the agent, each tile 16×16, so the frame is 64×64. Materials are outside, floor, wall, and station. Objects are none and station.
- Five actions: noop, turn left, turn right, forward, use. `use` scores only from the cell south of the station, and only in the pose the run asked for. Facing the station puts that tile in the bottom row of the crop. Facing away leaves it outside the image. The env refuses the other pose, so a run cannot score by turning around.
- Both teachers walk the same three steps to that cell. The away teacher then turns around. Both hold for 48 no-ops and then `use`. An anchored window of length 32 therefore sees the pose and `use`, not the walk.
- World model size S: embed 4096, channels `[32, 64, 128, 256]`, deterministic size 512, stochastic 32×32, field head 16 cells, 4 materials, 2 objects. No inventory head, no facing head, no benchmark checkpoint.
- Batch 8, sequence 32, train ratio 256, 8,000 environment steps, one seed. Eight teacher episodes are collected before training, after the env reset, so each one is a real demonstration.
- The reported action is whether `use` scored. The field number is the mode of `z_posterior` on a correct teacher episode, plus a 15-step mode rollout of `z_prior`.

## What the method section must say in one paragraph

The latent is trained to name a local field. Imagination reads the predicted field. The runs in the paper give the actor the true field on real steps, including evaluation. The two switches are the position of the relevant tile relative to that field, and whether the rare action is forced into the loss window.

