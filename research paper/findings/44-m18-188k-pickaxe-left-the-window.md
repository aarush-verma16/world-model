# Finding 44 — M18 188k: table came back to 18%; last-200 pickaxe and stone are 0

**Status:** measured on `m18_masked_inventory` at env 187632 (1157 lives), train flush 174848, 2026-09-29  
**Kind:** follow-up to finding 43; pickaxe did not return with table  
**Evidence:** `results/m18_masked_inventory/collect_episodes.jsonl`, `eval_metrics.json`, `train_metrics.json`

## Claim

Last-200 gmean is **2.41**. Wood **50%**, table **18.5%** (back from 12% at 156k), drink **18%**, length **174**. Wood pickaxe in the last 200 lives is **0**. Stone in that window is **0**. Cumulative pickaxe lives are still **36** and stone lives still **3** — no new one since finding 43. Those pickaxe lives still average **172** steps. `ac_H` **0.14**. `recon_l1` **0.0048**.

Table can sit in the high teens while the tool that stone requires is gone from the recent window. Held-out **2.01 at 175k** (wood 30, table 10, pick 0, stone 0) is n=10. Do not extend this run to 1M because table recovered.

## Paper spin

Results: on the inventory-mask run, `place_table` and `make_wood_pickaxe` are not one chain that locks in. Table can return while pickaxe stays at the 36 lives already counted, and stone stays those 3.
