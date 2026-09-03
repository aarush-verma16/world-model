# Finding 26 — Paper-ratio 512 + blocks=2 locked the island by 136k

**Status:** measured on M15-XL `m15_xl_r512_b2` at 135920 / 1M (807 lives), 2026-09-03  
**Kind:** negative result on “512 from scratch breaks the sleep island”; not a dead trainer  
**Evidence:** `results/m15_xl_r512_b2/collect_episodes.jsonl`, `eval_metrics.json`, `train_metrics.json`; M9 / M14 windows

## Claim

M15 (from scratch, `train_ratio` **512**, encoder/decoder **blocks=2**, ~1.66 env/s) at **~136k** is the same **wake / sapling / plant / die at ~180** island as M12/M13 (512, stopped ~100k) and M14 (ratio 32, 826k). The world model is **fine**: last flushed `recon_l1` **0.0030**, reward MAE **0.0008**. Last-200 gmean **~1.17** cannot leave ~1.2 because wood **fell** (13% in the first 50k → **4%**) and stone/table/pickaxe stay ~0. `ac_H` **0.25–0.30** (not the 0.10 floor).

The 512 actor overwrote early diversity. Held-out at **env 0** (fresh actor) had wood **20%** and table **20%**. Last-200 at 136k has wood **4%** and table **0.5%**. Random play already found the next +1s; 16 AC updates per 16 env steps then burned them. M9 at ratio 32 had wood **61%** at 100k. Do not grind M15 to 1M hoping 1.2 slopes up. Do not add a stone/hunger loss. Next run is actor warmup (`configs/m16_xl_r512_acwarmup.yaml`), not a resume of these weights.

The 180k skill look was the planned control. At 136k the island is already locked (wood worse than at 50k). Leave the kernel only if you want that extra 40k on disk; do not treat 180k as a delivery date.

## Windows (collect)

| env | n | mean len | wake | sapling | plant | wood | table | zombie | stone |
|---|---|---|---|---|---|---|---|---|---|
| 0–25k | 150 | 167 | 90 | 48 | 42 | **13.3** | 0.7 | 0 | 0 |
| 25–50k | 150 | 166 | 94 | 54 | 49 | **12.7** | 0 | 0 | 0 |
| 50–75k | 149 | 168 | 95 | 86 | 79 | 4.7 | 0 | 0.7 | 0 |
| 75–100k | 150 | 166 | 94 | 83 | 72 | 3.3 | 0 | 0.7 | 0 |
| 100–125k | 145 | 173 | 95 | 74 | 60 | 3.4 | 0.7 | 0 | 0 |
| 125–136k | 63 | 175 | 97 | 86 | 57 | 4.8 | 0 | 0 | 0 |
| last-200 | 200 | 173 | 96 | 78 | 60 | **4.0** | 0.5 | 0 | **0** |

Held-out n=10: **1.29 → 0.76 → 0.87 → 0.79 → 0.99 → 1.05**. Lottery (finding 13). The 125k eval bag is wake/sapling/plant plus one zombie; that is the dashboard bar chart, not last-200.

Flushed train at 124928: `recon_l1` 0.0030, `ac_H` 0.299, last-200 gmean 1.229, **1.66 env/s**.

## Failed alternatives

- Reading a flat 1.2 last-200 as “achievements are broken.” Wake/sapling/plant are 60–96%.
- Adding a stone / hunger / HUD bonus. Findings 01, 05, 19.
- Raising `entropy_scale` above 3e-4 as the first knob. Finding 17.
- Loading `checkpoints/m15_xl_r512_b2` into M16 and calling it warmup-from-scratch.
- Grinding M15 to 1M because recon is 0.003. A good WM of the island stabilizes the island.

## What we changed

`configs/m16_xl_r512_acwarmup.yaml`: same XL `blocks=2` + ratio 512, **`prefill_steps` 25000**, **`ac_warmup_env` 25000** (0 AC updates until then). New dirs. Notebook 10 last-200 bars come from collect jsonl.

## Paper spin

Results: paper replay rate on this graph locks the cheap +1 loop *faster* than ratio 32 (M9 wood 61% at 100k vs M15 wood 4% at 136k) while recon is already paper-good. The missing piece is not a decoder head; it is when the actor is allowed to train on the first easy rewards.
