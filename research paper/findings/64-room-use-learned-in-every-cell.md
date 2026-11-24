# In the room, use is learned with the station behind the agent and with the window off

**Claim.** With every teacher episode a real demonstration, `use` is learned in all four cells. The final eval is 8/8 whether the station is in the crop or behind the agent, and whether the loss window ends on `use` or is an ordinary slice. The field head names the station cell from `z_posterior` in every cell. The window changes how long the successful episode is, not whether `use` happens.

## Why this is not the paper

This is the second setting, one seed each, 8,000 env steps. Real eval steps still read the environment's field. An 8/8 here is not the benchmark's held-out rare action, which stayed 0/10.

## Evidence

Rerun on 2026-11-23. The 2026-11-17 checkpoints are archived under `checkpoints/room_2026-11-17` and `results/room_2026-11-17`. Teacher episodes are 8/8 in every run, length 52 when the station is in view and 54 when it is behind the agent. `in_view` at `use` is 1 on every in-view demonstration and 0 on every away demonstration.

| run | actor `use` | eval at ~2k, 4k, 6k, 8k | success length, median |
| --- | --- | --- | --- |
| station in view, window on `use` | 124/147 | 7/8, 7/8, 8/8, 8/8 | 36 |
| station in view, ordinary window | 85/110 | 5/8, 4/8, 6/8, 8/8 | 42 |
| station behind the agent, window on `use` | 185/190 | 8/8, 8/8, 8/8, 8/8 | 38 |
| station behind the agent, ordinary window | 130/138 | 7/8, 8/8, 8/8, 8/8 | 46 |

Every success has the pose the env requires: `in_view` 1.00 on the facing runs and 0.00 on the away runs. Failures are timeouts at 160 steps: 23, 25, 5, and 8. The away runs miss less often. The first success is at env 332, 188, 355, and 411.

The teacher uses at step 52 or 54, after a 48-step hold. Anchored successes sit in a narrow band just under 40 steps, about the walk plus the 32-step window (in-view p10/p50/p90 = 31/36/38; away 34/38/46). Uniform successes spread out (in-view 19/42/83, away 28/46/84). About 1–3% of successes are within two steps of the shortest legal path, so the policy is not skipping the hold entirely.

`scripts/probe_room_field.py` on these four `ckpt_final.pt` files, one correct episode per pose. Posterior mode: material accuracy 1 and station recall 1 on both poses, for all four models. False station rate is 0. Station probability on the trained pose is 1.000. On the facing episode the two away models are 0.986 and 0.981, and the mode is still the station. A 15-step mode rollout of `z_prior` from the first hold frame matches the trained pose, including the station cell when that pose shows it. The away models name that facing station at imagination step 1 and miss it at step 5.

Logged `wm_local` falls from about 0.08–0.12 on the first cycle to about 0.0001–0.0008 at 8k. `results/room_probe/summary.json` is this rerun. The 2026-11-17 probe numbers are in finding 63, not in that file.

## Failed alternatives

Reading the 2026-11-17 counts, 14/56 and 5/53 and 12/56 and 6/54, as this comparison. Those teacher files were half no-ops.

Reading 8/8 as the benchmark skill. The benchmark's held-out rare action was 0/10 in every cell, and that field head did not name the station.

Reading the away successes as imagination acting without the station. Eval is real steps, and real steps are given the environment's field. On an away success that field does not contain the station. The imagined action was not measured.

Treating the anchored length cluster as the full demonstration. The teacher holds for 48 steps. The anchored policy uses about one window earlier.

## Paper spin

Once the station fills an encoder cell and the episode is short, `use` is learned with the station behind the agent and with an ordinary loss window. What the window changes is the length of the successful episode. The benchmark sentence stays the one in findings 57, 60, and 61: the latent names grass, and the unaided count stays at two.
