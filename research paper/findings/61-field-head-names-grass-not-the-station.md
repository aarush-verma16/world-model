# The field head names grass, and puts almost no probability on the station or the ore

**Claim.** On the same eight teacher lives, the mode of `z_posterior` names the 9×7 material at 72% of cells, against a 54% grass baseline. Grass is named at probability 0.93. A table cell gets probability 0.13–0.16 of being a table. Coal, iron, and furnace sit near 0.03, 0.01, and 0.01–0.06. The three 100k checkpoints agree. By imagination step 15 the field has fallen back to the grass baseline.

## Why this is not the paper

This is the read of the latent on one benchmark's pixels. The method section still has to stand without that game's items, and the room is not built. Real steps in these runs still hand the actor the environment's field. Imagination is the path that reads this head.

## Evidence

`scripts/probe_field_head.py`, `results/field_probe/summary.json`, 2026-11-16. Eight lives from one process, four with the station placed in the crop and four walked home, cap 400. 116,235 cells. 62,533 are grass (53.8%), 725 are table, 167 are furnace, 1,609 are coal, 796 are iron. The read is the mode of the 32 categoricals. `h` does not see the current frame. `z_posterior` does. The same lives are scored with all three `ckpt_step_100000.pt` files.

| model | all cells | grass | stone | path | coal | iron | table | furnace |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M22 | 72.1% | 0.93 | 0.52 | 0.55 | 0.037 | 0.003 | 0.145 | 0.061 |
| M23 | 72.4% | 0.92 | 0.51 | 0.58 | 0.036 | 0.004 | 0.129 | 0.014 |
| M24 | 71.7% | 0.92 | 0.52 | 0.53 | 0.034 | 0.012 | 0.159 | 0.045 |

The number under each material is the mean probability the head assigns to the true class, which matches the mode recall. When the mode is wrong, the true class has almost no mass. The head is confident.

Objects are 114,205 empty cells out of 116,235. Object accuracy 97.6% is that empty class.

A mode rollout with the logged actions, 3,465 cells at each horizon: material accuracy is about 70% at step 1, about 65% at step 5, and 56%, 57%, and 60% at step 15. The grass baseline on the whole set is 54%. Table cells in those rollouts are 25 at step 15, and the mode names 2 of them in every model.

Open-loop strips, context 8, are `research paper/figures/field-head-open-loop-m22_field_craft.png` and the M23 and M24 twins. Context L1 on that window is 0.025, 0.024, and 0.029. Imagined frames have mean about 0.23.

The logged `wm_local` at env 100,000 was 0.014, 0.030, and 0.016. That average is mostly grass.

## Failed alternatives

Reading 72% cell accuracy as the field containing the station. Reading object accuracy 97.6% as the head seeing creatures. Reading `wm_local` near 0.02 as the station being named. Ranking M22, M23, and M24 on table recall. The three probabilities are 0.15, 0.13, and 0.16. Treating the step-15 table count, 2 of 25, as a rate that separates the checkpoints.

## Paper spin

The latent carries the terrain. It does not carry the station or the ore, and the average field loss hides that because most cells are grass. Imagination trains the actor on this field. The environment map used on real steps is a different input.
