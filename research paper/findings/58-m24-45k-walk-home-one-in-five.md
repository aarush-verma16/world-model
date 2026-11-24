# At 45k the walk-home teacher finishes iron on about 1 handoff in 5

**Claim.** M24 is paused at env 44,800 on 2026-11-07. The station is walked home, outside the 9×7 crop, and the craft window stays on. Stone pickaxe is 62/231 lives (27%). The teacher then made an iron pickaxe on 11 of 60 of those handoffs. The actor's own iron pickaxe is 0/171. Held-out iron pickaxe is 0/10 at 25k.

## Why this is not the paper

This is halfway to the 100k stop. Eleven of sixty is the teacher, and the field runs' full-budget conversion was about four in ten. The actor has not had a collect life to set next to M22 and M23's two.

## Evidence

`results/m24_walk_home`, `ckpt_latest.pt` written 18:37. The GPU was idle after that save. 231 lives, 171 actor, 60 assisted. Every assisted life has a stone pickaxe.

| env | lives | gmean | stone pickaxe | iron pickaxe given a stone pickaxe |
| --- | --- | --- | --- | --- |
| 0–25k | 107 | 23.7 | 30/107 (28%) | 3/30 |
| 25–45k | 123 | 24.6 | 31/123 (25%) | 8/31 |

Actor lives: geometric mean 12.81, length 207, wood pickaxe 128/171, wake 95%, zombie 81%. Two actor lives still show a stone pickaxe, env 34,960 (length 259) and env 35,616 (length 239). Neither has coal, a furnace, or iron. Actor-only stone pickaxe 2/171 is the assist tag.

Held-out, 10 lives: score 14.3 at env 0 and 21.3 at env 25,008. At 25k, stone pickaxe 4/10, coal 1/10, furnace 2/10, iron 1/10, iron pickaxe 0/10.

Step rate median about 0.88 env/s through 30k and about 0.92 from 30k to 45k. Last logged actor entropy is 0.13. The last notebook line said assist iron pickaxe 6/50. The episode file has 11/60.

## Failed alternatives

Reading actor-only geometric mean 12.8, or stone pickaxe 2/171, as the policy missing the handoff. Combined stone pickaxe is 62/231. Reading 11 assisted iron pickaxes as the actor's craft. Stopping at 45k and treating 11/60 as the finished walk-home rate. The field runs' first 50k were in this band too. M22's first half was about one iron pickaxe per five stone-pickaxe lives.

## Paper spin

Walking the station home has not stopped the handoff or the teacher's craft. At the halfway save the finish rate is lower than the field teacher's full run and close to that run's first half. The actor has no unaided iron pickaxe yet.
