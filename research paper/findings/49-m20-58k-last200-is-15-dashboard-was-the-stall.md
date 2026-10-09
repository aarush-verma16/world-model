# M20 at 58k last-200 is 15.7, and 0.07 env/s was the dashboard

**Claim.** At 58,336 env steps the actor's last-200 geometric mean is 15.7: wood pickaxe 59%, stone 36%, stone pickaxe 9%, coal 16%, furnace 4%, iron 2%. The world-model step is about 1.1 env/s. The stretch at 0.07 env/s was a matplotlib figure on every 256-step log, which blocked the notebook channel. The UI froze on step 50,688 while the loop was still advancing.

## Why this is not the paper

Held-out at 50k is 13.2 on 10 lives. Last-200 online is not that protocol, and 58k is not 500k. Do not caption 15.7 next to DreamerV3's 14.5.

## Evidence

`results/m20_local_map`, 286 actor lives, last env 58,336 (2026-10-09). Last 200: gmean 15.69, wood 95%, table 82.5%, wood pickaxe 59%, wood sword 34.5%, stone 36%, place_stone 18.5%, stone pickaxe 9%, stone sword 4%, coal 15.5%, furnace 4%, iron 2%, iron pickaxe 1%, diamond 0.5%, zombie 71%, length still ~200. Held-out eval: step 0 was 2.76, 25,008 was 14.06, 50,000 was 13.20 (mean length 194). Train logs from 256 through 57,344 sit at 0.9–1.16 env/s, then 57,600–58,368 sit at 0.06–0.08. `dashboard_every` had been set equal to `log_every` (256), and the text line was skipped whenever the figure ran, so a blocked `display()` looked like a stuck counter. Replay sampling of the 74k-step dump is 2 ms; the score pass over 286 lives is 30 ms. Neither is the 15× drop. Same class of stall as finding 08.

Teacher lives are not in `collect_episodes.jsonl`. These percents are the actor.

## Failed alternatives

A figure every log so the dashboard "keeps up." That is what finding 08 already ruled out. Text every 256 steps, a figure every 5000, and the status line printed before `display()`.

## Paper spin

Results: a 9×7 map plus a whole-life teacher in the buffer moves last-200 off the wood/table island by 58k (stone pickaxe 9%, coal 16%). Methods: the step rate on this box at train_ratio 512 stays ~1 env/s; a live figure on the log tick is not part of the recipe.
