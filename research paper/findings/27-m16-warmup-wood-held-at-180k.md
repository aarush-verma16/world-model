# Finding 27 — Actor warmup held wood at 180k; table still 0 so gmean cannot leave ~1.5

**Status:** measured on M16-XL `m16_xl_r512_acwarmup` at 186496 / 1M (1189 lives), 2026-09-04  
**Kind:** first positive skill delta vs M15, not a 14.5 trajectory  
**Evidence:** `results/m16_xl_r512_acwarmup/collect_episodes.jsonl`, `eval_metrics.json`, `train_metrics.json`; M15 last-200 at 141k

## Claim

Delaying the actor 25k (`ac_warmup_env`) **did** stop the M15 wood-unlearn. Last-200 **wood is 24.5%** at 186k (M15 was **3.5%** at 141k and falling). Wood **climbed** after AC-on: 9.6% (25–50k) → 13% (50–100k) → 17% (100–150k) → **19%** (150–180k). That is not the 512 island of finding 26.

It is also **not** a score spike. Last-200 gmean **1.78 at 150k → 1.42 at 186k** because **table is 0%** in the last 200 lives (it was 1–4% early). Geometric mean of wake/sapling/plant/wood saturates near **1.4–1.8**. Stone, pickaxe, and table must move for the teal line to leave that band. `defeat_zombie` is **1%**. Mean length **179**. `ac_H` **~0.30–0.35**. `recon_l1` **~0.004**. ~1.6 env/s.

Do not read the 180k dip as collapse. Do not caption 1.42 next to 14.5. Next look is whether **table** leaves zero while wood stays above ~15%. If wood falls back to ~4%, warmup only postponed finding 26.

## Windows (collect)

| env | n | mean len | wake | sapling | plant | wood | table | zombie | stone |
|---|---|---|---|---|---|---|---|---|---|
| 0–25k (AC off) | 151 | 165 | 93 | 56 | 51 | **28.5** | 4.0 | 0 | 0 |
| 25–50k | 146 | 170 | 95 | 79 | 61 | 9.6 | 1.4 | 0 | 0 |
| 50–100k | 303 | 165 | 93 | 81 | 70 | 12.9 | 1.7 | 1.7 | 0 |
| 100–150k | 383 | 174 | 97 | 89 | 84 | 17.2 | 1.0 | 1.6 | 0 |
| 150–180k | 170 | 177 | 97 | 87 | 78 | **19.4** | **0** | 1.2 | 0 |
| last-200 | 200 | 179 | 98 | 89 | 79 | **24.5** | **0** | 1.0 | **0** |

Held-out n=10: 1.74 → 0.96 → 0.88 → 1.55 → 1.29 → **1.80** → 1.03 → 1.30. Lottery (finding 13). The 1.80 at 125k was three extra wood/zombie lives, not a tech-tree climb.

## Failed alternatives

- Waiting for last-200 gmean to “spike” at 180k. The metric cannot spike without a new achievement rate.
- Calling warmup a failure because teal fell 1.78 → 1.42. Wood rose in that window.
- Adding a table/stone bonus. Same extra-loss regime as findings 01, 05, 19.
- Stopping M16 at 180k because it is “not 14.5.” This is the first 512 run where wood is still climbing.

## Paper spin

Results: a 25k actor delay changes *which* early +1s survive into the 512 on-policy loop. Wood can hold; crafting (table) can still die. Last-N gmean dips when a rare fourth achievement leaves the window even while a third (wood) is rising.
