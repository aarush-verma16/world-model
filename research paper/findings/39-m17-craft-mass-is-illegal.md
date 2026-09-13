# Finding 39 — M17 craft mass is almost all illegal: 98% of `place_table` has wood < 2

**Status:** measured on `data/m17_xl_paper_replay.pt` (2922 lives, 515124 steps), HUD inventory decoded with Crafter's own `ItemView` templates, 2026-09-13  
**Kind:** Crafter execution / identifiability; not a 14.5 recipe  
**Evidence:** `scripts/diag_craft_legality.py` (decoder checked on `crafter.Env` reset: health/food/drink/energy=9, wood=0)

## Claim

Finding 38's `P(place_table)=0.068` does **not** mean the actor crafts. Replay actions are `place_table` **3.68%** of steps (**18820** presses). **98.46%** of those presses happen with **wood < 2**, so Crafter no-ops. Only **289** presses had the items; **90** actually spent 2 wood (**0.48%** of table actions; **89/2922** lives, last-200 **4/200 = 2%**).

`make_wood_pickaxe` is **7583** presses (**1.48%** of steps). **81%** have wood < 1. Of **1408** with wood, **2** succeed (`wood -= 1` and pickaxe += 1). Pickaxe is in inventory on **0.02%** of steps. Last-200 pickaxe success **0**.

Stone cannot enter replay: you need a pickaxe in hand, and that happened twice in half a million steps. More env steps on this policy press craft on empty inventory. That is why 500k more of M17 will not invent mining.

## Numbers

| | place_table | make_wood_pickaxe |
|---|---|---|
| presses | 18820 (3.68% of steps) | 7583 (1.48%) |
| illegal (no items) | 18531 (**98.5%**) | 6175 (**81.4%**) |
| had items | 289 (1.54%) | 1408 (18.6%) |
| Crafter actually resolved | **90 (0.48%)** | **2 (0.03%)** |
| had items, still no-op | 199 | 1406 |

Inventory occupancy: `wood >= 2` on **2.51%** of steps; `wood >= 1` on **12.94%**; a wood pickaxe on **0.02%**. Only **8.3%** of lives ever held 2 wood. Last-200 collect_wood ~27% (finding 37) is “picked one log,” not “had a table kit.”

HUD decoder: live `crafter.Env` reset matches the templates. Success is `wood_t - wood_{t+1} = 2` (table) or wood −1 and pickaxe +1 (craft), not the achievement jsonl.

## Kill / do not train

This is not a graph bug to fix with another 5-day run. Raising entropy so they press `place_table` more would add more **illegal** presses. A stone/hunger bonus (findings 01, 05, 19) does not give them 2 wood on the facing tile.

A change is only worth a 25k probe if `scripts/diag_craft_legality.py` on the **same** dump (or the same policy on that dump) moves **successful** table/pickaxe counts, not `P(place_table)`.

## Failed alternatives

- Reading finding 38 table mass as “they know how to craft.” Mass is a button mash.
- Masking illegal actions in collect to force 14.5. That is not DreamerV3; do not ship it as the paper recipe.
- Grinding M17 because 90 tables “already started the tree.” 2 pickaxes in 2922 lives is the tree.

## Paper spin

Methods: discrete Crafter crafts are almost always illegal under a peaked island policy, so REINFORCE on `place_table` trains a no-op. Evaluation should report **legal** craft rate (HUD inventory × action), not actor softmax mass. Negative result: Table B.1 knobs produce craft spam, not tool use.
