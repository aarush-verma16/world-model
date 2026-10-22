# cuDNN autotune, not the replay, slowed the XL step

**Claim.** On this 5080 the XL batch-8 × seq-64 cycle is 1.17 env/s with cuDNN's heuristic and about 0.77 env/s with autotune. The interrupted M22 run opened at 1.14, logged 0.77 at step 256, and locked at 0.406 from step 4096 through 20,224. At that stop the actor's 94 lives are geometric mean 17.75 and iron pickaxe 0/94. Replay sampling of the saved buffer is 2 ms.

## Why this is not the paper

17.75 is the loaded M20 policy on 94 lives, not a finished M22 run, and not a number to put next to 14.5. The rate fix is a workstation flag.

## Evidence

`results/m22_field_craft`, interrupted 2026-10-22. Train log: step 256 was 0.771 env/s, 2048 was 0.621, 3072 was 0.429, then 0.405–0.406 from 4096 through 19,456. `vram_alloc` stayed 13.78 GiB, the same peak M20 logged while holding 1.16 env/s through 20k. 94 actor lives, last env 20,416, mean length 217, geometric mean 17.75. Wood 100%, table 95.7%, wood pickaxe 78.7%, stone 57.4%, stone pickaxe 18.1%, coal 17.0%, furnace 11.7%, iron 1/94, iron pickaxe 0, diamond 0, zombie 84%. Held-out is still the step-0 eval only.

A fresh process on `checkpoints/m22_field_craft/ckpt_latest.pt` and `data/m22_field_craft_replay.pt` (143 episodes, 34,622 steps), batch 8, seq 64, 16 world-model and 16 actor updates:

| piece | time |
| --- | --- |
| replay `sample` | 2 ms (teacher-window rebuild 0.023 s once) |
| collect 16 | 0.11–0.15 s |
| autotune on, one cycle | ~21 s, 0.77 env/s |
| autotune off, cycle 0 | 14.18 s, 1.13 env/s |
| autotune off, cycle 1 | 13.68 s, 1.17 env/s (world model 8.1 s, actor 5.5 s) |

`configure_runtime` now leaves `cudnn.benchmark` off. One process that had already reserved ~14.7 GiB and was pulling ~108 W at full clocks ran the same cycle at 0.32 env/s. That is the shape of the 0.41 lock. Two cycles with autotune off stayed at 1.13 then 1.17. That is not a 20k-step proof.

## Failed alternatives

Rewriting the replay. Sampling the 34k-step buffer is 2 ms, and the teacher episodes are 32 lives. `empty_cache` before a cycle left the autotuned rate at 0.76 env/s. Loading the checkpoint on CPU was already in and did not restore 1.16. The dashboard is every 5,000 steps; the lock was already in by step 4,096.

## Paper spin

Compute appendix: at train_ratio 512 this box is ~1.2 env/s when cuDNN autotune is off, and the replay is not the wall. Do not grind the 0.41 process. Iron pickaxe had not transferred in 94 lives.
