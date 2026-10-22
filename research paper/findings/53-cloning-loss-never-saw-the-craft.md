# The cloning loss was already zero, and it was not looking at the craft

**Claim.** At M22's stop the behavior-cloning loss was ~0 and iron pickaxe was still 0/94, because a random 64-step window of a teacher life almost never contains the craft. 22 of the 32 saved teacher lives do place the furnace and make the iron pickaxe, around step 170–245 of a life that is often 200–800 steps. The actor had memorized the slices it was shown, which are wood, the home table, and walking.

## Why this is not the paper

94 lives is not a finished run, and an assisted life is the teacher finishing a stone-pickaxe episode. That count is not the actor's Crafter score.

## Evidence

`results/m22_field_craft` train log, 2026-10-22, env 20,224: `ac_bc` was 0.247 at step 256, 0.018 at 1024, 0.001 at 12,288, and 7e-5 at 20,224, with `ac_bc_frac` still 0.38–1.0. `data/m22_field_craft_replay.pt`: 32 teacher episodes, 22 with `make_iron_pickaxe` (inventory rises once, at the action). The other craft buttons in those lives are `place_furnace` a few steps earlier and sometimes `make_iron_sword`. Actor collect: 94 lives, geometric mean 17.75, iron 1/94, iron pickaxe 0/94.

A 64-step window that ends on the craft is about 64 starts out of ~300–800, so a uniform teacher draw hits it a few percent of the time and the mean loss ignores it.

## What changed

Teacher draws now use those craft windows when any exist, and the craft button weighs 8 in the cloning loss. On collect, once a life has a stone pickaxe and no iron pickaxe, teacher v4 plays until the pickaxe exists. The episode is stored for cloning and tagged `assisted` in the jsonl. The logged score and `actor_px` skip those lives. `assist_px` is the teacher finishing a life the actor already got to a stone pickaxe.

## Failed alternatives

More of the same uniform cloning. The loss was already ~0. Raising `bc_scale` on windows that do not contain the craft teaches wood again.

## Paper spin

The demonstration only transfers if the gradient sees the craft state. A mean over the whole life reports success while the one button that matters is missing.
