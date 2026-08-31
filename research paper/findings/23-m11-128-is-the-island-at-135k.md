# Finding 23 — Ratio 128 from scratch is the sleep island by 135k

**Status:** measured on M11-XL `m11_xl_r128` at 137104 / 1M (786 lives), 2026-08-30  
**Kind:** negative result on “start over at 4× replay and the length line will move”; not a dead actor-critic  
**Evidence:** `results/m11_xl_r128/collect_episodes.jsonl`, `eval_metrics.json`; M9 jsonl restricted to env ≤ 135k

## Claim

At **~135k** env steps, M11 (from scratch, `train_ratio` 128, ~8 env/s) is already the same **sleep + die at ~180** island as M9. Last-200 mean length **177**, **76%** dead before 200, **0%** reach hunger (338). `wake_up` **96.5%**, stone **0**, wood pickaxe **0**, zombie **1%**. Held-out **1.24** at 100k (sapling 100, wake 100, wood 40, everything else ~0). Last-200 gmean **~2.0** is those easy unlocks, not a walk toward 14.5.

`ac_H` ~0.40–0.44 (unimix floor ~0.08). The trainer is not collapsed. The **length line will not drift up** if this runs to 400k — M9 already did that experiment at ratio 32.

M9 at the **same** env horizon had the same length (**177**) but **more** combat: zombie **12.5%**, wood **70%**. Four times the updates per env step did not buy earlier survival. It locked `wake_up` faster (85% in the first 25k).

Do not grind M11 to 1M. Do not resume it into a 512 run.

## Windows (collect)

| env | n | mean len | sapling | wood | plant | wake | zombie | stone | wood pick |
|---|---|---|---|---|---|---|---|---|---|
| 0–25k | 142 | 175 | 54 | 23 | 35 | 85 | 1.4 | 0 | 0 |
| 25–50k | 146 | 171 | 61 | 26 | 34 | 90 | 0 | 0 | 0 |
| 50–75k | 144 | 175 | 57 | 27 | 42 | 94 | 0 | 0 | 0 |
| 75–100k | 144 | 174 | 86 | 27 | 55 | 94 | 0.7 | 0 | 0 |
| 100–125k | 139 | 179 | 90 | 35 | 64 | 97 | 0.7 | 0 | 0.7 |
| 125–137k | 71 | 172 | 82 | 32 | 69 | 94 | 1.4 | 0 | 0 |

Held-out: 1.10 @ 0, **1.35 @ 50k**, **1.24 @ 100k**. n=10 lottery (finding 13), same three skills.

## Failed alternatives

- Waiting for mean length to “inch off” 180. Combat deaths do not trend; they jump when the policy stops sleeping in the open (finding 19).
- Reading held-out 1.35 → 1.24 as actor collapse. `ac_H` is ~0.4.
- Grinding because last-200 ticked 1.6 → 2.0. That is sapling/plant/wake saturating.
- Calling this the paper 512 1-day probe. ~8 env/s is ratio 128. `results/m12_xl_r512` was never written.

## Paper spin

Results: a 4× `train_ratio` from scratch can *advance* the `wake_up` local optimum rather than delay it. Survival (length, `defeat_zombie`, stone) is the Crafter score; extra replay on the first +1s is not.
