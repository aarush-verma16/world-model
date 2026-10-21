# Placing the station beside the iron crafts; walking home did not show up in M21

**Claim.** M21 at 22k still has iron pickaxe in 0 of 109 lives, because teacher v3 walks back to a table outside the 9×7 crop. Teacher v4 carries wood for a second table, mines the furnace stone at the ore, and places both on the cave floor. Across three draws of the same eight episode ids, v4 makes the iron pickaxe 5, 6, and 6 times, against v3's 3, 5, and 4, and finds diamond 3, 5, and 6 times against 0, 2, and 1.

## Why this is not the paper

These are teacher unlocks and one stopped actor run, not a Crafter score for M22. The actor has not been trained on the field-craft lives yet.

## Evidence

`results/m21_iron_craft`, 2026-10-20, stopped on purpose at env 22,048 (109 actor lives, last train 22,016). All 109: gmean 14.88, length 208, wood pickaxe 66%, stone 44%, stone pickaxe 11.9%, coal 11.9%, furnace 4.6%, iron 0, iron pickaxe 0, diamond 0. Tail rate about 0.36 env/s. Held-out exists only at step 0 (gmean 14.08). No `ckpt_step_*` was written (`checkpoint_every` is 25,000).

`CrafterTeacherV2`, eight episode ids, cap 800, idle stop 200 (2026-10-20). Crafter names each map with `hash((seed, episode))`, which is a different draw in each process. Three draws, v3 then v4 in the same process. Hash seed 0: iron pickaxe 3/8 versus 5/8, diamond 0/8 versus 3/8. Hash seed 1: 5/8 versus 6/8, diamond 2/8 versus 5/8. One unpinned process: 4/8 versus 6/8, diamond 1/8 versus 6/8. That is the rollout `seed_teacher_episodes` uses: eight resets on one env. Eight fresh worlds, each a new env so the episode counter stays 1, hash seed 0: v3 4/8 and v4 3/8. The wood trip before the cave loses some of those lives. In a traced life, v4 places the table on the path it just mined, turns into a side stone, mines that into a furnace pad, places the furnace, and crafts on the next step. Lives that never reach iron stay at 0 under both versions.

## Failed alternatives

Packing 4 stone before the cave, and topping wood up again after the table is placed. The first detour killed lives that already craft by walking a short way home. The second spends the table's wood and walks off, so the furnace is no longer next to the table. Stone is mined beside the ore. Wood is gathered once, before coal, and the coal branch does not run again once the ore is in hand.

Grinding M21 to 100k. At 22k the last-200 rates are the M20 plateau and iron is still 0.

## Paper spin

The demonstration has to place the table and furnace inside the crop the actor sees. A craft that happens at the home station is a long walk with no table in view, and 22k of cloning that walk did not produce an iron pickaxe.
