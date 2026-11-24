# Where each section gets its material

Read `method.md` for how to say it and `evidence.md` for the numbers. This page is the map back to the lab.

## Archive findings that feed the draft

| Finding | Section |
| --- | --- |
| `research paper/findings/01-rssm-bypass.md` | Method. A decoder that skips the posterior empties the latent. |
| `research paper/findings/04-crafter-pixel-geometry.md` | Method. Why a field head exists: one latent cell is larger than a benchmark tile. |
| `research paper/findings/56-m22-100k-handoff-rate-rose.md` | Experiments. Station inside the crop, window on. |
| `research paper/findings/57-m23-uniform-window-same-two-lives.md` | Experiments. Station inside the crop, window off. |
| `research paper/findings/60-m24-100k-teacher-finish-halved.md` | Experiments. Station walked outside, window on. |
| `research paper/findings/61-field-head-names-grass-not-the-station.md` | What the latent contains, benchmark. |
| `research paper/findings/62-room-crop-is-one-tile.md` | Method, second setting. Geometry only. |
| `research paper/findings/64-room-use-learned-in-every-cell.md` | Experiments, second setting. |

Finding 53 is the mechanism by which a random window misses a rare action inside a long episode. The 100k result is finding 57: over a full run, anchoring the window left the unaided count unchanged. Use 53 as a sentence, not as a second result.

Finding 63 is the 2026-11-17 room run. Do not cite it as the room result.

## Logs copied into this folder

| Draft path | Original |
| --- | --- |
| `data/probes/field_probe_summary.json` | `results/field_probe/summary.json` |
| `data/probes/room_probe_summary.json` | `results/room_probe/summary.json` |
| `data/benchmark/m22_field_craft/` | `results/m22_field_craft/collect_episodes.jsonl`, `eval_metrics.json` |
| `data/benchmark/m23_uniform_window/` | `results/m23_uniform_window/collect_episodes.jsonl`, `eval_metrics.json` |
| `data/benchmark/m24_walk_home/` | `results/m24_walk_home/collect_episodes.jsonl`, `eval_metrics.json` |
| `data/room/room_in_anchored/` | `results/room_in_anchored/episodes.jsonl`, `train_metrics.json` |
| `data/room/room_in_uniform/` | `results/room_in_uniform/episodes.jsonl`, `train_metrics.json` |
| `data/room/room_out_anchored/` | `results/room_out_anchored/episodes.jsonl`, `train_metrics.json` |
| `data/room/room_out_uniform/` | `results/room_out_uniform/episodes.jsonl`, `train_metrics.json` |

The episode tensors behind the benchmark field probe stay at `results/field_probe/episodes.npz`. They are not needed to print the table.

## Checkpoints, not copied

Each file is about 3.4 GB for the benchmark and about 250 MB for the room.

- `checkpoints/m22_field_craft/ckpt_step_100000.pt`
- `checkpoints/m23_uniform_window/ckpt_step_100000.pt`
- `checkpoints/m24_walk_home/ckpt_step_100000.pt`
- `checkpoints/room_in_anchored/ckpt_final.pt`
- `checkpoints/room_in_uniform/ckpt_final.pt`
- `checkpoints/room_out_anchored/ckpt_final.pt`
- `checkpoints/room_out_uniform/ckpt_final.pt`

Room replay buffers are about 110 MB each under `results/room_*/replay.pt`. They are not paper data.

## Configs

- Benchmark: `configs/m22_field_craft.yaml`, `configs/m23_uniform_window.yaml`, `configs/m24_walk_home.yaml`, size `configs/sizes/dreamer_xl_paper_local.yaml`
- Room: `configs/room_in_anchored.yaml`, `configs/room_in_uniform.yaml`, `configs/room_out_anchored.yaml`, `configs/room_out_uniform.yaml`, size `configs/sizes/dreamer_s_room.yaml`

## Code the method describes

- World model: `src/models/world_model.py`, `src/models/rssm.py`, `src/models/heads.py`
- Actor input, true field versus predicted field: `src/training/actor_input.py`
- Benchmark teacher and the assisted-life flag: `src/training/crafter_teacher.py`
- Room: `src/envs/room_env.py`, `src/envs/room_teacher.py`, `src/training/room_loop.py`
- Field probes: `scripts/probe_field_head.py`, `scripts/probe_room_field.py`

## Citation clusters to look up while writing

Do not invent pages or DOIs. Take them from the papers.

- World models: Ha and Schmidhuber; PlaNet; Dreamer; DreamerV2; DreamerV3; Dreamer 4.
- The benchmark and later agents on it.
- Behavior cloning, DAgger, and learning from demonstration inside an on-policy loop.
- Privileged information used as a training target.
- Local maps and rare events.

## Leave in the archive

Findings 02 through 55, except 04, are how the trainer was made to run or earlier runs that are not the comparison. Dashboards, VRAM plots, score-gap figures, and the November 17 room counts do not go in the paper. The list of sentences that must not be written is at the bottom of `README.md`.
