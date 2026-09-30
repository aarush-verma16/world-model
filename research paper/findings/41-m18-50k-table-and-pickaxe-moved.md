# Finding 41 — M18 at ~50k: table 15% and pickaxe 2.5% in last-200; stone is one life

**Status:** measured on `m18_masked_inventory` collect jsonl at env 49472 (364 lives), train flush at 36096, 2026-09-27  
**Kind:** first training look, not a 500k result and not a legality audit  
**Evidence:** `results/m18_masked_inventory/collect_episodes.jsonl`, `train_metrics.json`, `eval_metrics.json`. Compared with M17 last-200 at 106k (finding 32: table 4.5%, pickaxe 0.5%, stone 0).

## Claim

The mask-plus-inventory run is **ahead of M17 on craft achievements at a fifth of the env steps**. Last-200: wood **52%**, drink **32%**, `place_table` **15%**, `make_wood_pickaxe` **2.5%**, stone **0.5%** (1 of 200). Last-50 is stronger on crafts: table **26%**, pickaxe **6%**, stone **0**. Last-200 gmean **2.42**. Length **166**. `recon_l1` **0.005**. ~1.15 env/s. `ac_H` last **0.18** (wiggle 0.15–0.23). One early log hit **0.068** at env 1280, then left the floor.

These are **achievement flags**, not the HUD legality audit. M17's failure was illegal presses that never became achievements. A 15% table rate means the achievement fired, so some presses spent wood. It does not yet prove the illegal rate is near 0. Stone at one life is not a mine.

Held-out n=10: **1.67 at 0** (table 20) → **1.37 at 25k** (table 0, wood 50). Ignore orange.

## Do not

- Caption 2.42 next to 14.5.
- Kill because `ac_H` touched 0.07 once at step 1280.
- Treat last-50 pickaxe 6% as the 500k answer. Next look is **100k** with `diag_craft_legality.py` on the replay.

## Paper spin

Results: the first 50k of a labeled non-Dreamer craft mask moves table and wood-pickaxe achievement rates past what paper-knob M17 had at 100k. The open measurement is still successful-versus-illegal presses, and stone.
