# Fieldcraft — problem, method, and what counts

This is the claim the draft has to be able to say. It is not the draft.

## Problem

Crafter's 64×64 frame is a 9×7 grid of 7-pixel tiles plus a HUD. A Dreamer CNN stores that view in 4×4 cells, and one cell is larger than a tile, a tree, or the player (finding 04). The action `do` only hits the tile in front of the player. A policy that cannot tell that tile from grass cannot mine it, and a world model cannot learn a transition that is missing from replay.

On this machine the Dreamer baseline never left that state. M17, paper knobs, 500k env steps: last-200 geometric mean 1.61, stone in 0 of 2951 lives (finding 37). M19 added the facing tile and a teacher that stopped after the first stone: last-200 4.19, stone pickaxe in 0 of 292 lives (finding 47).

The problem is that gap. A craft the actor cannot see, or a craft that is not in the replay, does not get learned. Matching DreamerV3's published 14.5 is a different experiment (finding 20).

## Method

Fieldcraft keeps the DreamerV3 world model: discrete RSSM, imagination actor-critic, horizon 15, `train_ratio` 512. The RSSM is not the change.

Three additions, all on the policy side:

1. **Local map.** A head predicts the 9×7 grid (63 cells, material and object) from `[h, z_posterior]`. On real collect the actor reads the true grid. In imagination it reads the head.
2. **Whole-life teacher.** A script writes episodes into replay. Version 2 (M20) plays until it dies and stockpiles ore for two iron tools before crafting either. Version 3 (M21) crafts the pickaxe when 1 wood, 1 coal, and 1 iron are in hand, then walks home to the table. Version 4 (M22) places the table and the furnace beside the ore, inside the 9×7 crop.
3. **Behavior cloning.** Action cross-entropy on replay steps flagged `teacher`, with those episodes oversampled (`teacher_fraction` 0.5). Evaluation never calls the teacher. Teacher unlocks are not the score.

M20 trained from scratch and stopped at 200k. M21 loaded those weights into a new replay and stopped at 22k. M22 is the same load with teacher v4, and it has no actor score yet.

## What counts as a result

The result is the actor, from `collect_episodes.jsonl`. Teacher episodes are not in that file.

- **Primary number:** geometric mean of the official 22 achievements on the last 200 actor lives, plus the rates for stone, stone pickaxe, coal, furnace, iron, and iron pickaxe.
- **One seed.** A second seed does not exist. The draft says so.
- **Held-out is 10 lives.** It is logged in `eval_metrics.json`. It is not the result. M20's held-out score went from 20.7 at 100k to 6.6 at 200k with stone at 0 on that last bag.

## Numbers rechecked against the logs (2026-10-21)

M20, `results/m20_local_map/collect_episodes.jsonl`, 944 lives, last env step 199,936.

| Window | Lives | Geometric mean | Stone pickaxe | Coal | Iron | Iron pickaxe |
|---|---:|---:|---:|---:|---:|---:|
| Last 200 | 200 | 15.66 | 13.5% | 18.5% | 0.5% | 0 |
| 150k–200k | 227 | 15.52 | 13.2% | 18.5% | 0.4% | 0 |
| Final 50 | 50 | 11.78 | 6% | 12% | 0 | 0 |
| All lives | 944 | 15.86 | 12.5% | 17.5% | 1.4% | 0.3% (3 lives) |

The three iron-pickaxe lives are at env steps 25,024, 50,016, and 100,064. Quote last-200 (15.66) with the 150k–200k window (15.52) beside it. The final 50 lives are a slide, not a second score.

M21, `results/m21_iron_craft/collect_episodes.jsonl`, 109 lives, last env step 22,048. Geometric mean 14.88. Stone pickaxe 11.9%, coal 11.9%, iron 0, iron pickaxe 0. There are fewer than 200 lives, so that figure is the whole run.

Held-out M20 (`eval_metrics.json`, 10 lives): 2.8, 14.1, 13.2, 19.8, 20.7, 10.5, 13.7, 15.7, 6.6. Iron pickaxe is 0 at every point. These match the file.

## What is not a result

Teacher counts are eight lives, and Crafter names each map with `hash((seed, episode))`, which is a different world in each process. Finding 50's 2/8 to 4/8 and finding 51's 5, 6, 6 versus 3, 5, 4 are separate processes. One fresh-world draw in finding 51 has teacher v4 at 3/8 and v3 at 4/8. Those counts are why M22 exists. They are not a Fieldcraft score.

M22 has no `collect_episodes.jsonl`. There is nothing to report until that run finishes.

Do not print 15.66 next to DreamerV3's 14.5. The protocols differ: a symbolic map, a scripted teacher in the replay, behavior cloning, a 200k budget, one seed, and a held-out line that does not sit on the online number.
