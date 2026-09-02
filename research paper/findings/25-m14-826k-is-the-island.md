# Finding 25 — Ratio 32 + blocks=2 is still the sleep island at 826k

**Status:** measured on M14-XL `m14_xl_r32_b2` at 827664 / 1M (4619 lives), 2026-09-02  
**Kind:** negative result on “deeper CNN at the finishable replay rate”; not a dead trainer  
**Evidence:** `results/m14_xl_r32_b2/collect_episodes.jsonl`, `eval_metrics.json`, `train_metrics.json`; M9 last-200 at 405k

## Claim

M14 (from scratch, `train_ratio` 32, encoder/decoder **blocks=2**, ~18 env/s) at **~826k** is the same **wake / sapling / die at ~180** island as M9 (`blocks=1`, ratio 32, stopped at 405k). Cumulative online gmean **2.24** looks “higher than 100k” because those two achievements saturated. Last-200 is **1.41** and has been **falling** since the 180k look (**2.45 → 2.26 → 2.06 → 1.59**). Stone is **0**. Mean length **180**. `ac_H` last **0.79** (min 0.10). The trainer is not collapsed. Table B.1 residuals did not buy survival on this schedule.

Do not caption this next to 14.5. Do not grind the last 170k hoping last-200 walks back up. That is finding 21 again with a deeper CNN.

## Windows (collect)

| env | mean len | wake | sapling | plant | wood | zombie | stone | last-200 gmean (nearest log) |
|---|---|---|---|---|---|---|---|---|
| 0–180k | 176 | 89 | 81 | 42 | 45 | 4.3 | 0 | **2.45** @ 180k |
| 180–400k | 177 | 97 | 97 | 66 | 53 | 8.0 | 0.1 | **2.26** @ 400k |
| 400–600k | 183 | 97 | 98 | 50 | 38 | 6.6 | 0 | **2.06** @ 500k |
| 600–828k | 181 | 97 | 99 | 61 | 22 | 4.5 | 0 | **1.59** @ 800k |
| last-200 | 180 | 96 | 99 | **12** | **13** | 2.5 | 0 | **1.41** live |

M9 last-200 at 405k: length 177, wake 93, sapling 97, plant 74, wood 50, zombie **12**, stone 0, last-200 gmean **2.49**. M14 at the same env was already on that island; by 826k plant/wood have **dropped**, not climbed.

Held-out (n=10) sawtooth **0.53 → 3.27 → 0.95** (800k). Finding 13: that is lottery, not collapse.

## Failed alternatives

- Reading online 2.24 vs M12’s 1.4 at 100k as “blocks=2 worked.” Different replay rate and a saturated cumulative line.
- Waiting for length to leave 180. Combat deaths do not inch (finding 19).
- Finishing 1M so the number can sit next to 14.5. This is M9’s experiment to a longer horizon.

## Paper spin

Results: residual `blocks=2` at workstation `train_ratio` 32 is not a substitute for the paper replay rate or for a policy that mines. Last-N gmean can fall after the island locks even while `ac_H` stays healthy.
