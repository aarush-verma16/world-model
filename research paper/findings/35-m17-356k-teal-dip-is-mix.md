# Finding 35 — M17 356k: last-200 2.12→1.67 is the 318k mix leaving the window; wood 28%, not collapse

**Status:** measured on M17-XL `m17_xl_paper` at 355888 / 1M (2066 lives), 2026-09-11  
**Kind:** follow-up to finding 34; last-N identifiability, not a dead trainer  
**Evidence:** `results/m17_xl_paper/collect_episodes.jsonl`, `eval_metrics.json`, `train_metrics.json`

## Claim

The “score dropping like crazy” is **last-200 occupancy mix**, not actor collapse. Teal **2.12 at 323k → 1.67 at 356k** because the 318k extras left the window: drink **15% → 8%**, table **7.5% → 2.5%**, cow **6% → 2.5%**, wood **39% → 28%**. Wake/sapling/plant stayed saturated (**95 / 94 / 86**). Last-50 table is **0**. Stone is still **0**. Length **170**. Cumulative online **1.79** barely moved.

Held-out **0.876 at 350k** is the n=10 lottery at its worst: wood **0**, table **0**, plant **100**. The previous bag was **1.51** (wood 20). Do not kill for orange 0.88.

`ac_H` last **0.17** (min **0.16**). Unimix floor ~0.08. `recon_l1` **0.0037**. Wood **28%** is still a hold vs M15’s **4%** at 136k. Last-50 wood **22%** is the number to watch at 400k — if it goes to ~4%, that is finding 26. A 2.1 → 1.7 last-N dip with wood in the 20s is finding 28.

## Last-200 ticks

| env | gmean | wood | drink | table | plant | cow | stone |
|---|---|---|---|---|---|---|---|
| 318k (finding 34) | **2.09** | 38.5 | 14.5 | 7.0 | 85 | 6.0 | 0 |
| 330k | 1.97 | 33.0 | 15.5 | 6.0 | 84 | 4.5 | 0 |
| 350k | 1.71 | 29.5 | 10.0 | 3.5 | 83 | 2.0 | 0 |
| 356k live | **1.67** | **28.0** | 8.0 | **2.5** | 86 | 2.5 | 0 |
| last-50 | 1.54 | **22.0** | 8.0 | **0** | 88 | 4.0 | 0 |

## Failed alternatives

- Reading orange 0.88 / teal 2.1→1.7 as M7 collapse. `ac_H` 0.17, wood 28%.
- Adding a wood/table bonus so last-N cannot dip (findings 01, 05, 19).
- Grinding to 1M because online 1.79 is “still fine.” Cumulative hides the mix leaving.

## Paper spin

Evaluation: last-N gmean on a four-to-six achievement island is a **window statistic**. A 0.4 drop in 30k env steps can be three rare +1s rolling out, not a new algorithm failure. Caption wood% next to teal.
