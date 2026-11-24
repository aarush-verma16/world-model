# Evidence the draft may quote

Rounded numbers below are the ones to print. Exact floats are in `data/probes/`. Episode logs are in `data/benchmark/` and `data/room/`. If a rounded number and a file disagree, the file wins and this page gets corrected before the tex does.

One seed per run. Say so.

## What the latent contains on the benchmark

Source: `data/probes/field_probe_summary.json`, eight teacher lives, 116,235 cells, checkpoints at 100k. The mode of the 32 categoricals, not one sample. Mean probability of the true class matches the recall, so a miss is a confident miss.

Grass is 62,533 of the cells (53.8%). That is the baseline a head achieves by naming grass everywhere.

| run | cell accuracy | grass | stone | path | coal | iron | table | furnace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| station inside the crop, window on | 72.1% | 0.927 | 0.516 | 0.550 | 0.037 | 0.003 | 0.145 | 0.061 |
| station inside the crop, window off | 72.4% | 0.924 | 0.507 | 0.575 | 0.036 | 0.004 | 0.129 | 0.014 |
| station walked outside, window on | 71.7% | 0.923 | 0.517 | 0.531 | 0.034 | 0.012 | 0.159 | 0.045 |

Cell counts for those classes: grass 62,533, stone 13,702, path 17,852, coal 1,609, iron 796, table 725, furnace 167.

Object accuracy is 97.6%. Almost every cell is empty, so that number is the empty class.

A mode rollout of `z_prior` is about 70% accurate at step 1 and 56–60% at step 15, against the 54% grass baseline. By step 15 the field has fallen back to grass. The logged field loss near 0.02 is the average over cells, and most cells are grass.

Figures: `figures/field-open-loop-station-inside.png`, `figures/field-open-loop-window-off.png`, `figures/field-open-loop-station-outside.png`. Context reconstruction is sharp. Imagined frames are dark. The caption is the field-head numbers, not a claim that the decoder failed.

## The two switches on the benchmark

Lives are from `data/benchmark/*/collect_episodes.jsonl`. A life with no assisted flag is the actor's. A life the teacher finished after the handoff is the teacher's. Held-out is 10 lives in `eval_metrics.json`.

The actor's rare action, unaided, is two lives in every cell. Held-out execution of that action is 0 of 10 at every checkpoint.

| run | teacher finishes, given the handoff | actor's own rare actions | held-out rare action |
| --- | --- | --- | --- |
| station inside the crop, window on | 23/60 | 2 | 0/10 |
| station inside the crop, window off | 44/121 | 2 | 0/10 |
| station walked outside, window on | 23/114 | 2 | 0/10 |

The two inside-crop lives that also reach further are env 25,024 (length 196) and env 75,024 (length 300) with the window on, and env 25,136 (length 268) and env 75,024 (length 48) with the window off. The two outside-crop lives are env 50,016 (length 100) and env 75,024 (length 74). They stop at the rare action.

Do not print the combined geometric mean. It counts achievements the teacher earned after the handoff. The actor-only last-200 geometric mean sits near 13 to 14 and is mostly earlier achievements.

## The room

Source: `data/room/*/episodes.jsonl` and `data/room/*/train_metrics.json`, rerun of 2026-11-23. Teacher episodes are 8/8 in every cell, length 52 facing the station and 54 facing away. `in_view` at `use` matches the switch on every teacher episode.

| run | actor `use` | eval at about 2k, 4k, 6k, 8k | median success length |
| --- | ---: | --- | ---: |
| station in the crop, window on `use` | 124/147 | 7/8, 7/8, 8/8, 8/8 | 36 |
| station in the crop, ordinary window | 85/110 | 5/8, 4/8, 6/8, 8/8 | 42 |
| station behind the agent, window on `use` | 185/190 | 8/8, 8/8, 8/8, 8/8 | 38 |
| station behind the agent, ordinary window | 130/138 | 7/8, 8/8, 8/8, 8/8 | 46 |

Every success has the pose that cell requires. Failures are timeouts at 160 steps: 23, 25, 5, and 8. The teacher uses at step 52 or 54 after a 48-step hold. Anchored successes cluster just under 40 steps. Ordinary-window successes spread from the teens past the teacher's length. About 1–3% of successes are within two steps of the shortest legal path.

Field head, `data/probes/room_probe_summary.json`, one correct teacher episode per pose, mode of `z_posterior`. Material accuracy 1 and station recall 1 on both poses, for all four models. False station rate 0. Station probability on the trained pose is 1.000. On the facing episode the two away-trained models are 0.986 and 0.981, and the mode is still the station. A 15-step mode rollout of `z_prior` from the first hold frame matches the trained pose. The away-trained models name the facing station at imagination step 1 and miss it at step 5.

Figures: `figures/room-crop-facing-station.png`, `figures/room-crop-facing-away.png`. These are the crops at the south neighbor, scaled with nearest-neighbor from 64×64. They are geometry, not a trained reconstruction.

The 2026-11-17 room logs are in `results/room_2026-11-17/` and are not this table. Half of those teacher episodes never left the start.

## Identifiability notes the method can use

A reconstruction path that skips `z_posterior` drives pixel loss down while the latent stays empty. At the benchmark encoder resolution, one latent cell covers more than the tile the rare action uses. Both notes are in the archive findings named in `sources.md`. They are not new numbers to recompute.
