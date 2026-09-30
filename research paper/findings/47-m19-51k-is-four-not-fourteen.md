# M19 at 51k is geometric mean 4 because stone pickaxe never entered the tree

**Claim.** M19 last-200 geometric mean 4.19 at 51,408 env steps is wood + table + a little pickaxe. Stone is 3% of lives, stone pickaxe is 0, length is still 175. Facing names one 7px tile. The map teacher stopped four steps after the first stone, so later mining never existed in replay. The live notebook also never passed `teacher_fraction` into `outer_cycle`, so even those short teacher lives were sampled uniformly after pretrain.

## Why this is not the paper

4.2 is not 14.5. Do not caption them together. This is the measurement that says “keep training M19” will not open coal, iron, or furnace.

## Evidence

`results/m19_facing`, 292 lives, last env 51,408 (2026-09-30). Last 200: gmean 4.19, wood 75%, table 52%, drink 40%, wood pickaxe 11.5%, wood sword 11.5%, stone 3%, zombie 3.5%, stone pickaxe 0, coal 0.5%, iron 0, furnace 0, length 174.9. Last 50: pickaxe 6%, stone 2%, gmean 3.69. 33 pickaxe lives, 8 stone lives, all 8 stone lives also had a pickaxe. Pickaxe mean length 197.5. Train 49,920: `ac_H` 0.19, `recon_l1` 0.0046, inventory 0.0017, spatial 0.021. Held-out 50k is 2.33 (n=10, pickaxe 0, stone 0).

v1 teacher on this run: 64 lives, 61 mined one stone in ~30 steps, then stopped. CrafterTeacherV2 on seeds 0–7, cap 800: mean length 245, 8/8 stone pickaxe and defeat_zombie, 7/8 furnace and coal, 5/8 iron, 0/8 iron pickaxe or diamond.

## Failed alternatives

More M19 steps. A second facing head. Extending M19 to 1M while stone pickaxe is 0.

## Paper spin

Results: an inventory-and-facing mask plus a teacher that stops at first stone produces a high-2s to low-4s island. Methods: later-tree transitions have to be in the buffer, and the actor needs the 9x7 view as symbols, not only the faced tile.
