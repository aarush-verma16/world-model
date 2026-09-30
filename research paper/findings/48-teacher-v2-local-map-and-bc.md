# Whole-life teacher plus a 9x7 map and cloning those actions

**Claim.** The missing transitions after first stone are playable by a map teacher: 8/8 lives reached `make_stone_pickaxe` and `defeat_zombie`, 7/8 placed a furnace, 5/8 collected iron, in ~245 steps. M20 writes those lives into replay, feeds the actor the 9x7 local grid, clones teacher actions on those steps, and keeps injecting 8 lives every 25k actor steps. Eval never calls the teacher. No M20 score yet.

## Why this is not the paper

No actor score. Teacher length 245 is still the combat wall; iron pickaxe was 0/8. This does not guarantee 14.5.

## Evidence

`CrafterTeacherV2.act` on `CrafterReward-v1`, seeds 0–7, cap 800 (2026-09-30). Unlocks per 8 lives: table/wood tools/stone/stone pickaxe/zombie 8, coal/furnace 7, iron 5, iron pickaxe 0, diamond 0. Mean length 245.

M19 at 51k never saw `make_stone_pickaxe` in 292 actor lives (finding 47). Its notebook called `outer_cycle` without `teacher_fraction`, so yaml `0.5` was idle after pretrain. M20’s notebook passes `teacher_fraction` and `bc_scale`.

The local map is the same 9x7 `engine.LocalView` cells already in `CrafterEnv` info: 63 material classes and 63 object classes, supervised from `[h, z]`, concatenated onto the actor. Imagination uses the head. Real collect still uses ground truth, same as M19 facing.

## What this does not fix

The actor still cannot see beyond 9x7. Iron tools need longer lives than 245. Cloning a global-BFS teacher can fail to transfer if the actor only has the local crop. If last-200 stone pickaxe is still ~0 at 100k, the next piece is not “train longer.”

## Paper spin

Methods: change the training distribution to include the later tree, upweight those windows, and clone the teacher on those steps while giving the actor the view as symbols. Results: teacher unlock table is the ceiling of the data, not of the score.
