# Walking the station home halved the teacher's finish rate and left two short actor crafts

**Claim.** M24 finished env 100,000 on 2026-11-14. The station is walked home, outside the 9×7 crop, and the craft window stays on. The teacher made an iron pickaxe on 23 of 114 stone-pickaxe handoffs (20%). M22 was 23/60 and M23 was 44/121, both about 37%. The actor's own iron pickaxe is two lives, env 50,016 (length 100, return 7.0) and env 75,024 (length 74, return 6.0). Held-out iron pickaxe is 0/10 at every checkpoint. Last quarter stone pickaxe is 30/107.

**Correction (2026-11-24).** "Outside the 9×7 crop" holds for the walk home, not for the craft. Crafter's `make_*` checks `world.nearby(pos, 1)`, the 3×3 around the player, and that is always inside the 9×7. In M24 the station is out of the field while the teacher walks back from the ore and in the field for the last steps before the craft. Only the room (finding 64) puts the station outside the image at the action.

## Why this is not the paper

Two unaided lives is the same count as both field-station runs, and the held-out iron pickaxe is zero in all three. The crop switch separated the teacher's finish rate. It did not separate the actor's iron pickaxe into a zero.

## Evidence

`results/m24_walk_home`, `ckpt_step_100000.pt` written 2026-11-14 17:53. 463 lives, 349 actor, 114 assisted. Every assisted life has a stone pickaxe.

| env | lives | gmean | stone pickaxe | teacher iron pickaxe given a stone pickaxe |
| --- | --- | --- | --- | --- |
| 0–25k | 107 | 23.7 | 30/107 (28%) | 3/30 |
| 25–50k | 142 | 24.6 | 35/142 (25%) | 8/33 |
| 50–75k | 107 | 23.5 | 23/107 (22%) | 6/22 |
| 75–100k | 107 | 25.5 | 30/107 (28%) | 6/29 |

The last quarter's seventh iron pickaxe is the untagged life at env 75,024. Actor-only last-200 geometric mean is 13.93, length 217, wood pickaxe 144/200, wake 97%, zombie 89%. The two untagged stone-pickaxe lives inside that window are the two iron-pickaxe lives. Two earlier untagged stone-pickaxe lives, env 34,960 (length 259) and env 35,616 (length 239), have no coal and no furnace.

Both untagged iron-pickaxe lives have coal, a furnace, and iron. Neither has an iron sword or a diamond. Across all three 100k runs the teacher made zero iron swords and collected zero diamonds. The sword and the diamond appear once each, on the long untagged lives in M22 (env 25,024, length 196) and M23 (env 25,136, length 268).

Held-out, 10 lives: 14.3, 21.3, 15.4, 19.5, 19.7 at env 0, 25k, 50k, 75k, and 100k. Iron pickaxe is 0/10 every time. At 100k, stone pickaxe 2/10, coal 2/10, furnace 2/10, iron 0/10.

Step-rate median about 0.88 through 40k and about 1.05 from 40k to 100k. Last actor entropy is 0.092.

## Failed alternatives

Reading actor-only geometric mean 13.9, or stone pickaxe 4/349, as the handoff dying. Combined last quarter is 25.5 and stone pickaxe is 30/107. Reading 23 assisted iron pickaxes, or the combined geometric mean, as the actor's score. Treating the two short lives as a held-out skill. Reading "station outside the crop" as an actor iron-pickaxe count of zero. The matched count is two, two, and two. Grinding M24 past 100k.

## Paper spin

The Crafter table is three 100k runs. Anchoring the craft button left the unaided count at two. Walking the station home cut the teacher's finish rate from about two handoffs in five to about one in five, and the two actor lives stop at the pickaxe. Held-out iron pickaxe is 0/10 in every cell. The sentence "the actor cannot craft when the station is outside the crop" is stronger than this table.
