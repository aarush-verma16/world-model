# By 70k the walk-home actor has one unaided iron pickaxe

**Claim.** M24 is at env 69,888 on 2026-11-13 and still stepping. The station is walked home. One untagged life made an iron pickaxe, at env 50,016 (length 100, return 7.0, with coal, a furnace, and iron). Held-out iron pickaxe is 0/10 at env 0, 25,008, and 50,000. In the latest window the teacher finished iron on 6 of 16 handoffs.

**Correction (2026-11-24).** The untagged iron-pickaxe lives in this finding span a teacher injection and inherited the demonstrator's world; the actor's own count is 0 (finding 65).

## Why this is not the paper

One collect life is the same count M22 had at this budget, and it is shorter. The held-out bag has not made an iron pickaxe. The run has about 30k steps left.

## Evidence

`results/m24_walk_home`, log line env 69,888 at about 0.98 env/s. The card was near 15.7 GiB, 180 W, and 2900 MHz while that line was written. 334 lives, 255 actor, 79 assisted. Every assisted life has a stone pickaxe.

| env | lives | gmean | stone pickaxe | iron pickaxe given a stone pickaxe |
| --- | --- | --- | --- | --- |
| 0–25k | 107 | 23.7 | 30/107 (28%) | 3/30 |
| 25–50k | 142 | 24.6 | 35/142 (25%) | 8/35 |
| 50–70k | 83 | 24.5 | 17/83 (20%) | 7/17 |

The 7 in the last window are 6 teacher lives plus the untagged life at env 50,016. Teacher overall is 17/79. The first 25k was 3/30. From 50k the takeover finish rate is 6/16.

Actor lives: geometric mean 13.28, length 211, wood pickaxe 185/255. The other two untagged stone-pickaxe lives are env 34,960 (length 259) and env 35,616 (length 239). Neither has coal or a furnace. Actor entropy on the 69,888 line is 0.073.

Held-out, 10 lives: 14.3 at env 0, 21.3 at env 25,008, 15.4 at env 50,000. At 50k, stone pickaxe 2/10, coal 2/10, furnace 0/10, iron 1/10, iron pickaxe 0/10.

Step rate median about 0.88 through 30k, about 0.94 from 30k to 50k, and about 1.08 from 50k to 70k.

## Failed alternatives

Reading actor-only geometric mean 13.3 as the handoff dying. Combined stone pickaxe is 82/334. Reading 17 assisted iron pickaxes as the actor's craft. Treating the length-100 life as a held-out skill. Reading the 50k held-out score 15.4, after 21.3 at 25k, as a collapse. That bag is 10 lives. M22's 50k bag was 12.8 and the later bags were higher.

## Paper spin

Walking the station home has not blocked the handoff, and it has not blocked a single unaided craft. At 70k that craft is one short life, and the held-out iron pickaxe is still 0/10. The recent teacher finish rate matches the field teacher's full-run rate. The cumulative rate is lower because the first 25k finished 3 of 30.
