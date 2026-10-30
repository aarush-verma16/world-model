# At 100k the actor reaches the stone pickaxe more often, and the field teacher finishes about 4 in 10

**Claim.** M22 finished at env 100,000 on 2026-10-28. In the last quarter, 24 of 112 lives crafted a stone pickaxe (21%), up from 20 of 112 in the first quarter and from 13.5% in M20's last 200. The teacher, taking over only after that pickaxe, made an iron pickaxe on 10 of those 23 handoffs. One more iron pickaxe, at env 75,024, has no `assisted` flag. Held-out iron pickaxe is 0/10 at 100k.

## Why this is not the paper

The combined geometric mean of 24 counts achievements the teacher earned after the handoff. The two unaided iron-pickaxe lives are 2 of 360 actor lives, and none of the four held-out bags of 10 contain one. Actor-only last-200 of 13.1 is the tag, not a second M20.

## Evidence

`results/m22_field_craft/collect_episodes.jsonl`, 444 lives, closed with `ckpt_step_100000.pt` at 18:17 on 2026-10-28. `assisted` is written only when the collector set the flag. Stone pickaxe is crafted before that flag can turn on, so the stone-pickaxe count on an assisted life is still the actor's action.

All lives, by quarter:

| env | lives | gmean | stone pickaxe | iron pickaxe | of which the teacher, given a stone pickaxe |
| --- | --- | --- | --- | --- | --- |
| 0–25k | 112 | 18.7 | 20/112 (18%) | 1/112 | 1/3 |
| 25–50k | 112 | 23.4 | 15/112 (13%) | 6/112 | 5/14 |
| 50–75k | 108 | 22.4 | 21/108 (19%) | 7/108 | 7/20 |
| 75–100k | 112 | 24.2 | 24/112 (21%) | 11/112 | 10/23 |

Last 200 lives of any kind (env 55,168–99,840): geometric mean 23.0, stone pickaxe 41/200, iron pickaxe 16/200. M20's last 200 was 15.66, stone pickaxe 13.5%, coal 18.5%, furnace 4%, iron pickaxe 0.

Actor-only last 200: geometric mean 13.14, stone pickaxe 2/200, iron pickaxe 1/200. Across the whole run the actor bucket is 360 lives, geometric mean 16.02, iron pickaxe 2/360. The log's `actor_px` was 2 at env 79,872, while both unaided lives were inside the window, and 1 at env 100,000 because env 25,024 had aged out.

Unaided iron-pickaxe lives, `assisted` absent:

- env 25,024, length 196, return 7.8: iron pickaxe, iron sword, diamond.
- env 75,024, length 300, return 13.6: coal, furnace, iron, iron pickaxe. No sword, no diamond.

After env 25,000 only three actor lives still show a stone pickaxe. The third is env 50,016, length 82, coal and furnace, no iron. The other 60 stone-pickaxe lives from the handoff onward are the assisted bucket, 23/60 iron pickaxe. Ten of those 23 finished after env 83,248, the last at env 97,488.

Held-out, 10 lives, teacher never called: score 20.3, 20.8, 12.8, 18.8, 18.0 at env 0, 25k, 50k, 75k, 100k. Iron pickaxe 0/10 every time. At 100k, stone pickaxe 4/10 and coal 4/10. Furnace was 2/10 at 75k and 0/10 at 100k.

The 24 assisted lives with no stone pickaxe are one block, env 61,472–66,768, mean length 197. That is the health takeover. The last quarter has none. Final-quarter step rate median is about 0.65 env/s. Actor entropy at env 100,000 is 0.13.

## Failed alternatives

Reading actor-only last-200 13.1, or stone pickaxe 2/200, as the policy falling back through M20. Those lives left the bucket when the teacher took over. Reading combined 24, or 23 assisted iron pickaxes, as the actor's Crafter score. Reading two unaided crafts as a held-out skill.

## Paper spin

The policy's own measured step is reaching a stone pickaxe, and that rate was higher in the last quarter than in M20's finished window. Field placement then finishes the iron pickaxe on about four handoffs in ten, through the end of the budget. The unaided craft is two collect lives, and the held-out bags do not contain it.
