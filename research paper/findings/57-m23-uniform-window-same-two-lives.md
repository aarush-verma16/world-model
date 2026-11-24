# With the craft window off, the unaided iron craft is still two lives

**Claim.** M23 finished at env 100,000 on 2026-11-03. It is the M22 recipe with `craft_window_fraction` 0: teacher rows are ordinary slices, and the field teacher is unchanged. Last quarter, 37 of 111 lives crafted a stone pickaxe, and 14 of those 37 made an iron pickaxe. Two of the iron pickaxes are untagged, at env 25,136 and env 75,024. Held-out iron pickaxe is 0/10 at 100k. M22, with the craft button anchored, also ended at two unaided lives and 0/10.

**Correction (2026-11-24).** The untagged iron-pickaxe lives in this finding span a teacher injection and inherited the demonstrator's world; the actor's own count is 0 (finding 65).

## Why this is not the paper

One switch, one seed. Combined geometric mean 26 counts the teacher's crafts after the handoff. Two collect lives are the same count M22 already had, and the held-out bags still do not contain the craft. The walk-home switch is not in this run.

## Evidence

`results/m23_uniform_window`, `ckpt_step_100000.pt` at 03:31 on 2026-11-03. 443 lives, 322 actor, 121 assisted. Every assisted life has a stone pickaxe. No health-takeover block.

All lives, by quarter. Stone pickaxe is the actor's craft; the teacher starts only after it is in the inventory.

| env | lives | gmean | stone pickaxe | iron pickaxe |
| --- | --- | --- | --- | --- |
| 0–25k | 111 | 22.3 | 15/111 (14%) | 7/111 |
| 25–50k | 109 | 28.9 | 37/109 (34%) | 10/109 |
| 50–75k | 112 | 26.9 | 35/112 (31%) | 15/112 |
| 75–100k | 111 | 26.7 | 37/111 (33%) | 14/111 |

M22's last quarter was geometric mean 24.2, stone pickaxe 24/112, iron pickaxe 11/112. Teacher conversion of a stone-pickaxe handoff is 44/121 here and was 23/60 on M22.

Unaided iron-pickaxe lives, `assisted` absent:

- env 25,136, length 268, return 9.1: iron pickaxe, iron sword, diamond.
- env 75,024, length 48, return 3.8: coal, furnace, iron, iron pickaxe. No sword, no diamond.

The only other actor life with a stone pickaxe is env 50,016, length 55, no iron. Actor-only last 200: geometric mean 14.03, stone pickaxe 2/200, iron pickaxe 1/200. That 1 is the length-48 life. Actor-only 0/322 would be the wrong read of the tag.

Held-out, 10 lives, teacher never called: score 9.5, 19.7, 17.1, 20.5, 21.1 at env 0, 25k, 50k, 75k, 100k. Iron pickaxe 0/10 every time. At 100k, stone pickaxe 6/10, coal 3/10, iron 2/10, furnace 0/10. M22's 100k bag was 18.0 with stone pickaxe 4/10 and iron 0/10.

Step rate after the wedged first line: about 1.16 env/s through 10k, about 1.0 through 30k, then about 0.74 from 40k to 100k. Last logged actor entropy is 0.057.

## Failed alternatives

Reading actor-only stone pickaxe 3/322 as the policy failing to reach the handoff. Those lives are the assisted bucket; combined last quarter is 37/111. Reading 44 assisted iron pickaxes, or combined geometric mean 26, as the actor's score. Treating two unaided crafts as the craft window being unnecessary for a held-out skill. The held-out craft is 0/10 with the window and 0/10 without it. Grinding past 100k because the last entropy is 0.057.

## Paper spin

Anchoring the craft button and sampling ordinary slices of the same field-teacher lives produced the same unaided outcome: two collect lives, held-out iron pickaxe 0/10. The lever this pair does not isolate is where the station is built.
