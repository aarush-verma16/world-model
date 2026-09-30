# Finding 43 — M18 156k: table 22%→12% and pickaxe 7.5%→1.5%; stone is 3 lives

**Status:** measured on `m18_masked_inventory` at env 155568 (973 lives), train flush 149760, 2026-09-28  
**Kind:** follow-up to finding 42; the 108k craft climb did not hold  
**Evidence:** `results/m18_masked_inventory/collect_episodes.jsonl`, `eval_metrics.json`, `train_metrics.json`

## Claim

Last-200 gmean **2.69 at 108k → 2.39 at 156k** because the new crafts left the window. Table **22.5% → 12%**, wood pickaxe **7.5% → 1.5%**, wood still **47%**, drink **22%**, length **173**. Last-50 pickaxe is **0**, table **6%**, gmean **1.88**.

Stone exists and is still not a behavior. **3 / 973** lives collected stone. All 3 also made a wood pickaxe. Those pickaxe lives (36 total) last **172** steps, the same wall as the rest. `ac_H` **0.15**. `recon_l1` **0.005**. Held-out **1.03 at 125k** (wood 0) → **1.80 at 150k** (wood 30, table 10, stone 0) is n=10.

The 250k stone check is still the stop-or-extend gate. A 1% last-200 stone rate at 156k is three lucky pickaxe lives, not a reason to move the budget to 1M.

## Paper spin

Results: the inventory mask can open wood-pickaxe achievements by 108k and those rates can slide back by 156k while stone stays a handful of lives that die at ~170. Gmean in the mid-2s tracks that craft mix.
