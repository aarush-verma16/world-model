# At 51k the takeover crafts iron; the actor has done it once

**Claim.** At the 2026-10-23 interrupt (env 51,312) teacher v4, taking over after a stone pickaxe, made an iron pickaxe in 7 of 18 assisted lives. One unassisted collect life, env 25,024, length 196, made an iron pickaxe, an iron sword, and a diamond. The 50k held-out bag of 10 did not.

## Why this is not the paper

Seven of eighteen is the teacher finishing a life the actor already got to a stone pickaxe. One collect life is not a rate, and the held-out score at 50k is one bag of ten.

## Evidence

`results/m22_field_craft`, interrupt saved `ckpt_final.pt` at 03:20. Train log ends at env 51,200. After the benchmark-off restart the step settled around 0.85–0.90 env/s (0.883 at 51,200), not the earlier 0.41 lock.

Actor-only collect: 212 lives, geometric mean 17.62, length 216. Since the resume at env 20,448: 118 actor lives, geometric mean 16.30, stone pickaxe 2/118. That drop is the accounting. The 18 assisted lives all have a stone pickaxe, so they left the actor pool. Combined since resume, stone pickaxe is 20/136 (14.7%), the same band as the 20k window.

The one actor iron-pickaxe life is env 25,024. It is not tagged `assisted`. It also has `make_iron_sword` and `collect_diamond`. `actor_px` stayed 1 from step 25,088 through 51,200, so no second actor life did it. Held-out at 25,008: score 20.79, iron 1/10, iron pickaxe 0/10. Held-out at 50,000: score 12.84, coal 0, furnace 0, iron 0, stone pickaxe 2/10. Ten lives swing that much; it is not a collapse.

Assisted lives: 18, length 305, geometric mean 43. Coal 12/18, furnace 13/18, iron pickaxe 7/18. The seventh finished at env 51,312, after the last log line (that line still said 6). Eleven assisted lives reached a stone pickaxe and still did not craft.

## Failed alternatives

Reading the actor-only geometric mean 16.3, or stone pickaxe 2/118, as the policy getting worse. Those lives were moved into the assisted bucket. Reading held-out 12.8 as the run dying. Reading 7/18 as the actor learning the craft.

## Paper spin

The station can be finished from a stone-pickaxe state the policy already reaches. The policy's own iron pickaxe, so far, is one life, and the 50k held-out bag did not contain it.
