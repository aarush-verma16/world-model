# The room field head names the station tile; the policy comparison does not

**Claim.** On a correct teacher episode, the mode of `z_posterior` names the station cell and does not paint it onto other cells. A 15-step mode rollout of `z_prior` still matches the field on the pose that run trained. The four policy counts are not a window-by-pose result, because half the stored teacher episodes never leave the start.

## Why this is not the paper

The latent result can sit next to the Crafter field read. The action result cannot. Every other demonstration in the replay is 160 no-ops, so the anchored-versus-uniform gap is not identified.

## Evidence

Four runs, 2026-11-17, about 8,000 env steps each, size S, no Crafter checkpoint. Eval is 8 actor episodes.

| run | actor `use` | last eval | posterior on its own pose |
| --- | --- | --- | --- |
| station in view, window on `use` | 14/56 | 3/8 | accuracy 1, station probability 1.000, 52/52 cells |
| station in view, ordinary window | 5/53 | 2/8 | accuracy 1, station probability 1.000, 52/52 cells |
| station behind the agent, window on `use` | 12/56 | 2/8 | accuracy 1 on the away episode (4 station cells, probability 1.000) |
| station behind the agent, ordinary window | 6/54 | 2/8 | accuracy 1 on the away episode (4 station cells, probability 1.000) |

False station rate on the posterior is 0 in every cell of that table. The in-view episode has one station cell on every frame. The away episode has the station on the four walk frames only, then none through the hold. The numbers in this table are the 2026-11-17 probe. `results/room_probe/summary.json` was rewritten on 2026-11-23 for the rerun; that file is finding 64.

The exception is the other pose. The away-anchored model, read on the facing episode, names 36 of 52 station cells (probability 0.65). The away-uniform model names all 52, at probability 0.85. Neither paints the station onto a floor cell.

Imagination, one rollout from the first hold frame, logged actions, mode of `z_prior`. On the pose the run trained, step 15 still matches: the two in-view models name the station cell, and the two away models keep a field with no station cell (accuracy 1, false station 0). Cross-pose is not that stable. The away-anchored model names the facing station at step 1 and misses it at step 5. The in-view uniform model’s away rollout is accuracy 0.44 from step 1.

The teacher file is alternating. Each run stored 4 real demonstrations (length 52 or 54, reward 1) and 4 timeouts (length 160, reward 0, the agent still on the start tile). `teacher.reset` ran before `env.reset`, so the plan was made from the previous episode’s ending pose and then played from the start. A uniform window of that pool contains `use` in 4 slices out of about 600. An anchored window still ends on the 4 real uses. Actor successes are longer than the demonstration (means about 68 to 98 steps, against 52 or 54), and the last quarter does not climb (2/14 and 1/14 on the two anchored runs). Entropy at 8k is 0.22 to 0.36.

## Failed alternatives

Reading 14/56 against 5/53, or 12/56 against 6/54, as the loss window transferring `use`. The eval bags are 3/8, 2/8, 2/8, and 2/8. Half the teacher rows are no-ops.

Reading those eval rates as a solved room.

Reading the logged `field_station_recall` of 1.0 as the probe. It was computed on the same broken teacher rollout. The reset-correct probe is the number above, and on the trained pose it agrees.

## Paper spin

The second setting’s latent names a tile that fills one encoder cell. The Crafter field head did not name the station. The action half of this 2×2 waits on a rerun whose demonstrations all leave the start.
