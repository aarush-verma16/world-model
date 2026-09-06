# Finding 29 — M16 462k: 350k dip recovered; table 4% is not a tree; stone still 0

**Status:** measured on M16-XL `m16_xl_r512_acwarmup` at 461680 / 1M (2737 lives), 2026-09-06  
**Kind:** follow-up to finding 28, not a 14.5 trajectory  
**Evidence:** `results/m16_xl_r512_acwarmup/collect_episodes.jsonl`, `eval_metrics.json`, `train_metrics.json`

## Claim

The 350k scare reversed. Last-200 gmean **1.49 at 350k → 2.08 at 425k → 1.90 live**. Wood did **not** collapse to M15’s 4%: last-200 wood is **22%** (peak **35%** at 425k). Drink held ~23–34%. Table left the 1% floor to **4%** (8 of 200 lives). Stone is still **0**. Wood pickaxe **0.5%**. Mean length **178**.

That is a healthier mix of the same cheap +1s, not a tech-tree climb. Geometric mean can sit near 2.0 on wake/sapling/plant/wood/drink plus a rare table. It cannot walk toward 14.5 while stone is 0.

Held-out after 350k: 1.28 → 1.71 → **2.07 (425k, wood 50 / drink 50 / table 20)** → **1.87 (450k, wood 60 / table 10)**. Same n=10 lottery as finding 28. The 2.07 is not a new high-water mark.

`ac_entropy` 400–450k mean **0.53** (min 0.32). `wm_recon_l1` **0.004**. ~1.60 env/s. Trainer is not collapsed. Planned halfway look is **500k** (~6 hours at this rate). If table is still single-digit and stone is 0 there, warmup is a wood-hold. Do not grind to 1M from 462k hoping 1.9 slopes up.

## Last-200 ticks

| env | gmean | wood | drink | table | plant | zombie | stone | pick |
|---|---|---|---|---|---|---|---|---|
| 350k (finding 28) | 1.49 | 25.5 | 23.5 | 1.0 | 59 | 0 | 0 | 0 |
| 375k | 1.52 | 16.0 | 29.0 | 1.5 | 69 | 0 | 0 | 0 |
| 400k | 1.85 | 24.5 | 26.0 | 2.0 | 70 | 1.0 | 0 | 0.5 |
| 425k | **2.08** | **35.0** | 34.5 | 4.0 | 70 | 1.5 | 0 | 1.0 |
| 450k | 1.95 | 22.5 | 24.5 | 4.0 | 72 | 2.0 | 0 | 0.5 |
| 462k live | 1.90 | 22.0 | 23.0 | **4.0** | 69 | 1.0 | **0** | 0.5 |

Cumulative online gmean **1.70**.

## Failed alternatives

- Reading teal 1.49 → 2.08 as leaving the island. Stone 0, length 178.
- Captioning held-out 2.07 next to 14.5.
- Killing at 462k because table is “only” 4%. The 500k look is the planned control.
- Adding a table bonus so 4% becomes 40%. Findings 01, 05, 19.

## Paper spin

Results: the 350k last-N dip on a paper-ratio warmup run can reverse without a new achievement class. Table at 4% is detectable and still lottery-scale. Stone remaining 0 at ~half the 1M budget is the binding constraint, not orange-line variance.
