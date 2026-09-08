# Finding 32 — M17 100k: `ac_H` 0.21 is not collapse; wood held, stone still 0

**Status:** measured on M17-XL `m17_xl_paper` at 106496 / 1M (649 lives), 2026-09-08  
**Kind:** planned 100k look; entropy identifiability, not a 14.5 trajectory  
**Evidence:** `results/m17_xl_paper/collect_episodes.jsonl`, `eval_metrics.json`, `train_metrics.json`

## Claim

`ac_H` **0.21** (flushed min **0.19** at 95k) is the lowest *healthy* XL band we have logged. It is **not** dead. Collapse (findings 14, 16) is a **flat 0.08–0.10** with wake-only lives. Last-200 wood is **39%**, sapling **93%**, plant **79%**, table **4.5%**, stone **0**, pickaxe **0.5%**. Last-200 gmean **1.85**. Length **178**. `recon_l1` **0.0048**.

That is M16’s wood-hold with paper seq/GRU/replay, not M15’s 4% wood unlearn at 136k (finding 26). `ac_H` drifted 0.32 (25–50k) → 0.25 (50–75k) → **0.22** (75–100k). Real-env `collect_entropy` is **0.079** — greedy in collect, still mixing in imagination. Kill if `ac_H` **sits under 0.12** *and* wood falls toward 4%. Either alone is not a kill.

Held-out 100k **1.96** (wood 70 / table 40 / pick 10 / stone **0**) is n=10. 25k was 2.75. Do not caption 1.96 next to 14.5.

## Last-200 vs M15 / M16

| run | env | last-200 wood | table | stone | `ac_H` |
|---|---|---|---|---|---|
| M15 (finding 26) | 136k | **4%** | ~0 | 0 | 0.25–0.30 |
| M16 (finding 27) | 186k | **24.5%** | 0 | 0 | ~0.30–0.35 |
| M17 live | 106k | **39%** | 4.5 | **0** | **0.21** |

## Failed alternatives

- Reading 0.21 as the unimix floor. Floor is ~0.08; last twelve logs wiggle 0.20–0.24.
- Killing M17 at 100k because entropy is lower than M16’s 0.43. Wood is higher than M16 at the same horizon.
- Reading the unlock bars as “stone 40.” Last-200 `collect_stone` is **0**. The tall bars are sapling / plant / wood / wake.
- Waiting for teal 1.85 to walk to 14.5. Stone is still 0.

## Paper spin

Results: Table B.1 knobs at batch 8 can hold wood at 100k with imagination entropy in the 0.20s. That entropy is a **greedy island**, not a dead actor. The 14.5 missing piece is still stone, not a 0.21 `ac_H`.
