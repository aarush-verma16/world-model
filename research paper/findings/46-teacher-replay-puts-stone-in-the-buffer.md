# A map teacher puts stone in the replay; uniform sampling would throw it away

**Claim.** The world model cannot learn a transition that is absent from replay. M18 at 203,648 steps had stone in 3 of 1,245 lives. A facing head does not create those frames. A map-reading teacher, used only to write data, finishes wood → table → wood sword → wood pickaxe → one stone in about 30 steps, 3/3 episodes in a direct check. Those episodes are pinned in the buffer and half of each training batch is drawn from windows that overlap them. Eval never calls the teacher.

## Why this is not the paper

No M19 score exists. This does not promise a geometric mean of 14.5. It promises the mining transition is in the training distribution on purpose. The actor still has to do it on a fresh map with no teacher.

## Evidence

`seed_teacher_episodes` on `CrafterReward-v1`, seeds 0–2, cap 1500: lengths 33, 23, 25. Each life unlocked `place_table`, `make_wood_pickaxe`, and `collect_stone`. The teacher reads `crafter.Env._world` and walks to the nearest tree, a legal table tile, the table it just placed, and the nearest stone. Adjacent zombies are faced and hit first. `do` is used on a tree, a hostile, or stone, which is the same button the mask deliberately leaves legal.

Uniform replay would bury 64 such lives inside 500k agent steps (about 0.1% of draws). `ReplayBuffer.sample(..., teacher_fraction=0.5)` draws half the batch from windows that overlap a `teacher=True` episode. FIFO eviction skips those episodes, so a 1e6 cap cannot delete the only stone data. Configs that omit `teacher_episodes` stay at zero; M18 is unchanged.

## What this does not fix

The teacher sees the whole grid. The actor sees pixels plus inventory, facing, and nearby materials. Copying “walk twenty tiles toward stone” is not a function of those features. What the oversampled lives can teach is the local part: faced a tree, press `do`; wood in hand and grass in front, place the table; pickaxe in hand and stone in front, press `do`. Combat past a wood sword, iron, and diamonds are not in the teacher’s chain.

## Paper spin

Methods: when a predicate never occurs, change the training distribution, and upweight the new episodes or the gradient never sees them. Results: the before-number is M18’s 3 stone lives. The after-number is the actor’s own last-200 stone rate on M19, which is not measured yet.
