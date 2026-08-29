# Finding 21 — The 400k flat gmean is a skill island, not a dead trainer

**Status:** measured on M9-XL `m9_xl` at 404992 / 1M, 2026-08-29  
**Kind:** eval identifiability + Crafter local optimum; not a reason to roll back the RSSM  
**Evidence:** `results/m9_xl/collect_episodes.jsonl` (2265 lives), `eval_metrics.json` (9 ticks), `train_metrics.json`; last-200 vs cumulative gmean

## Claim

The blue online line sitting at **~2.56 from 150k to 405k** is a **saturated early-game island**, not a broken architecture. Last-200 lives still score **2.49**. Almost every life already does `wake_up` / sapling / plant. The remaining 19 Crafter achievements stay near zero, so the geometric mean cannot move. Held-out **2.82 at 400k** is the same n=10 lottery as finding 13 (350k was **1.39**). Rolling back to extra recon heads or a hunger bonus will not mine stone.

Do not grind another 600k at `train_ratio` 32 hoping the cumulative line climbs. FIFO 500k has **not** evicted yet — rare pickaxe/zombie windows are still in replay. The next lever is more replay per env step on these weights (M10 `train_ratio` 128), not a new decoder.

## What the 405k jsonl actually is

Mean/median length **179 / 178**, max **466**, 0.5% ≥ 400. Same combat wall as finding 19. `ac_H` mean 0.55–0.84 by 50k window (floor ~0.08). `recon_l1` last-10k **0.012**, `kl_rep_raw` **15.4** — the world model is still learning on-policy coverage, not frozen.

| env window | n | sapling | wood | plant | wake | zombie | table | wood pick | stone |
|---|---|---|---|---|---|---|---|---|---|
| 50k | 275 | 43 | 33 | 16 | 74 | 6.5 | 4.4 | 0.4 | 0 |
| 100k | 276 | 92 | 61 | 56 | 93 | 14 | 16 | 3.6 | 0 |
| 150k | 279 | 94 | 65 | 80 | 93 | 10 | 7.2 | 2.9 | **0.7** |
| 250k | 290 | 94 | 46 | 74 | 90 | **21** | 3.1 | 0 | 0 |
| 400k | 281 | 96 | 50 | 63 | 95 | 11 | 3.6 | 1.1 | 0 |

Zombie kills **peaked at 21% then fell**. Stone appeared once at 150k and vanished. Wood pickaxe 3.6% → ~1%. The policy shifted toward plant + sleep, which is +1 `wake_up` and then a 7-damage sleeping zombie (finding 19).

Cumulative gmean **2.559** vs last-200 **2.494** vs env>350k **2.494**. The dashboard cumulative line is *designed* to go flat once a few achievements saturate. Plot last-200 or it will be read as “training stopped.”

Held-out ticks: 1.06, 0.96, 2.06, 1.95, 1.34, 1.51, 1.84, **1.39**, **2.82**. Last eval: sapling/plant/wake 100, wood 80, drink 30, cow 20, table 10, zombie 10, **stone 0**. That 2.82 is four easy unlocks in ten lives, not a tech-tree climb.

M6 at 400k held-out was **1.12** (finding 13). M9 is a better island, not a worse trainer. M6 at 1M was still 1.64 with stone 0.02%. Another 600k of ratio 32 is that experiment again.

Imagination at 400k is a 15-step `z_prior` of standing still by water. That is what this policy *does*. A static imagination of a static policy is not a decoder bypass (finding 01).

## Failed alternatives

- Roll back the RSSM / extra recon heads / crop losses. Findings 01, 05.
- Add a drink/hunger/sleep penalty. Finding 19.
- Treat held-out 2.82 as a new high-water mark and 350k 1.39 as collapse.
- Resume `m8_xl_acfix2` or restart XL from scratch to “fix the architecture.”
- Grind M9 to 1M at ratio 32 because the cumulative line might move. Last-200 is already the policy.
- Wait until replay FIFO drops the remaining pickaxe rows (~500k env) and then turn up `train_ratio`.

## What we changed

`configs/m10_xl_r128.yaml`: same XL graph, `train_ratio` **128** (4 WM+AC / 16 env), loads `ckpt_step_400000.pt` into new dirs. Paper 512 is still 16× this. Notebook 10 plots last-200 gmean so the next plateau is visible.

## Paper spin

Results / evaluation: Crafter geometric mean over a growing episode pool saturates once a few achievements hit ~90%. A second curve (last-N lives) is required to see whether the *policy* is stuck. Negative: 12% `defeat_zombie` does not move mean length off ~180 if `wake_up` stays ~93%.
