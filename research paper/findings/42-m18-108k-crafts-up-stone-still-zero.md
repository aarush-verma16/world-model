# Finding 42 — M18 108k: table 22% and pickaxe 7.5%; gmean 2.69 still has stone at 0

**Status:** measured on `m18_masked_inventory` at env 108448 (699 lives), train flush 99840, 2026-09-27  
**Kind:** 100k craft look; not a 500k forecast and not 14.5  
**Evidence:** `results/m18_masked_inventory/collect_episodes.jsonl`, `eval_metrics.json`, `train_metrics.json`. M17 at 106k (finding 32): table 4.5%, pickaxe 0.5%, stone 0, gmean 1.85.

## Claim

Last-200 gmean **2.69** is a richer **early tree**, not a late tree. Wake 95 / sapling 81 / plant 65 / wood **48%** / drink 23 / table **22.5%** / wood pickaxe **7.5%** / wood sword 1.5 / zombie 1.5 / stone **0** / stone pickaxe **0** / coal 0 / iron 0 / furnace 0. Length **175**. `ac_H` **0.15**. `recon_l1` **0.005**. ~1.32 env/s.

From 25k → 108k, table went **5% → 22%** and pickaxe **1% → 7.5%**. That is the M18 difference versus M17, which at 100k was still table 4.5% and pickaxe 0.5%. Geometric mean can sit in the high 2s on that set. It cannot reach 5, let alone 14.5, while stone is 0. Lives that craft a wood pickaxe are still dying on the combat wall before a stone tile.

Held-out n=10 at 100k is **1.70** (table 10, pick 0, stone 0). The 75k bag was 2.34. Lottery.

## 500k, not 1M

Keep the **500k** stop. Extend to 1M only if last-200 **stone** is non-trivial by ~250k. If pickaxe keeps rising and stone stays 0 through 250–500k, the bottleneck has moved from illegal presses to “pickaxe in hand, never on a stone tile, dead at ~175.” Another 500k of the same policy is finding 30 again.

## Paper spin

Results: inventory conditioning plus a legality mask, on the same outer loop as M17, moves wood-pickaxe achievement rate by 108k. The score above 2 is that craft, not mining. The 500k question is whether pickaxe mass reaches stone.
