# M20 at 200k held stone and coal; iron pickaxe left the window

**Claim.** At the 200k stop the actor's last-200 geometric mean is 15.7, with stone pickaxe 13.5% and coal 18.5%. Iron pickaxe is 0 in that window. The three iron-pickaxe lives in the whole run all happened before 166k. Teacher v2 dies holding both ores because it stockpiles a second tool's materials before the first craft; crafting as soon as the pickaxe recipe is in hand raises the same 8 seeds from 2/8 to 4/8.

## Why this is not the paper

200k with a scripted teacher in the replay is the measurement of this recipe, not a second protocol. Held-out at the stop is 6.6 on 10 lives (stone 0, coal 0) after 20.7 at 100k. That orange line is not the result.

## Evidence

`results/m20_local_map`, 944 actor lives, last env 199,936, checkpoint `ckpt_step_200000.pt` (2026-10-16). Recomputed from `collect_episodes.jsonl` on 2026-10-21. Last 200: gmean 15.66, length 220, wood 97.5%, table 91%, wood pickaxe 69.5%, stone 45.5%, stone pickaxe 13.5%, coal 18.5%, furnace 4%, iron 0.5%, iron pickaxe 0, diamond 0, zombie 81%. Windows of gmean / stone pickaxe / coal / furnace / iron: 0–50k 13.0 / 5.6 / 11.7 / 2.0 / 1.2; 50–100k 17.1 / 16.5 / 18.2 / 6.4 / 2.5; 100–150k 17.3 / 15.0 / 21.9 / 6.4 / 1.3; 150–200k 15.5 / 13.2 / 18.5 / 4.0 / 0.4. The final 50 lives are gmean 11.78, stone pickaxe 6%, coal 12%, furnace 0. Quote last-200 with the 150k–200k window, not the final 50 as a second score. Lifetime iron-pickaxe lives: 3/944, at env steps 25,024, 50,016, and 100,064. Held-out n=10: 2.8, 14.1, 13.2, 19.8, 20.7, 10.5, 13.7, 15.7, 6.6. Last train row: `wm_recon_l1` 0.009, `ac_entropy` 0.09, 0.72 env/s.

`CrafterTeacherV2` seeds 0–7, cap 800, one process (2026-10-19). Version 2: iron pickaxe 2/8. Seeds 2 and 3 died with coal=2 and iron=2, goal still the craft, because coal and iron are gathered for both the pickaxe and the sword before either is made. Version 3 crafts when the next tool's own recipe is in hand and skips the hunt: iron pickaxe 4/8 in that same process, including seed 2. Seeds 3 and 6 still die on the walk back with wood=1, coal=1, iron=1. Crafter's map id is `hash((seed, episode))`, so this 2/8 and 4/8 is not the same eight worlds as finding 51. Eight lives in one process are a reason to change the teacher, not a rate to publish.

## Failed alternatives

Grinding M20 past 200k. Stone pickaxe and iron both left their mid-run peaks in the last window. Another 200k of the same teacher does not put the craft in the buffer.

## Paper spin

Results: the local map plus a whole-life teacher moves last-200 through stone pickaxe and coal by 200k, and the iron-pickaxe rate does not stay. Methods: the demonstration has to craft the pickaxe when its own recipe is in hand. Stockpiling the sword's ore spends the life in the cave.
