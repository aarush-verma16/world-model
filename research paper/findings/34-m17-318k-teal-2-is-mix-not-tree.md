# Finding 34 — M17 318k: last-200 2.10 is drink/table/cow mix; stone still 0

**Status:** measured on M17-XL `m17_xl_paper` at 318560 / 1M (1848 lives), 2026-09-10  
**Kind:** follow-up to finding 33, not a 14.5 trajectory  
**Evidence:** `results/m17_xl_paper/collect_episodes.jsonl`, `eval_metrics.json`, `train_metrics.json`

## Claim

Teal **1.82 at 200k → 2.10 at 318k** is more of the **same cheap +1s**, not a tech-tree climb. Last-200: wood **39%**, drink **15%**, table **7.5%**, eat_cow **6%**, plant **85%**, pickaxe **0.5%**, stone **0**. Length **178**. Online **1.80**. `ac_H` last flushed **0.18** (min **0.16** at 262k). `recon_l1` **0.0047**. ~1.31 env/s.

Table leaving 4% for **7.5%** is detectable and still lottery-scale (finding 29’s 4% at 462k). Geometric mean can sit near 2.1 on wake/sapling/plant/wood/drink plus rare table/cow. It cannot walk toward 14.5 while stone is 0.

Held-out **1.17 (200k) → 1.34 (300k, wood 20 / table 0 / stone 0)** is the n=10 lottery. Do not read orange 1.34 vs teal 2.10 as two policies.

## Last-200 ticks

| env | gmean | wood | drink | table | plant | cow | stone | pick |
|---|---|---|---|---|---|---|---|---|
| 200k (finding 33) | 1.82 | 33.5 | 9.5 | 4.0 | 78 | 1.5 | 0 | 0 |
| 250k | 1.64 | 27.0 | 6.0 | 2.0 | 78 | — | 0 | 0 |
| 300k | 1.90 | 39.0 | 10.5 | 5.0 | 89 | — | 0 | 0.5 |
| 318k live | **2.10** | 38.5 | **15.0** | **7.5** | 85 | **6.0** | **0** | 0.5 |

## Failed alternatives

- Reading teal 1.8 → 2.1 as leaving the island. Stone 0, length 178.
- Captioning 2.10 next to 14.5.
- Killing because `ac_H` is 0.18. Floor is ~0.08; wood held.
- Grinding to 1M hoping 2.1 slopes up.

## Paper spin

Results: a last-N gmean tick from extra drink/table/cow on a paper-knob 512 run is the same occupancy mix as finding 29. Stone remaining 0 past 300k is still the binding constraint.
