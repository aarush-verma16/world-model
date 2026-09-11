# Finding 36 — M17 400k: wood is sliding 39%→21% (last-50 12%); teal will not walk to 14.5

**Status:** measured on M17-XL `m17_xl_paper` at 401472 / 1M (2328 lives), 2026-09-11  
**Kind:** follow-up to finding 35; wood fade, not just mix leaving  
**Evidence:** `results/m17_xl_paper/collect_episodes.jsonl`, `eval_metrics.json`, `train_metrics.json`

## Claim

Finding 35’s 2.12→1.67 dip was mix leaving. By **400k** wood itself is fading: last-200 **39% (318k) → 21%**, last-50 **12%**, table **1%** (last-50 **0**). Last-200 gmean **1.60**. Wake/sapling/plant still **96 / 96 / 77**. Stone **0**. Length **173**. `ac_H` **0.19**. `recon_l1` **0.004**. Cumulative online **1.78**.

This is not M7 collapse (entropy on 0.08, wake only). It is closer to M15’s wood-unlearn (13%→4% by 136k) on a slower clock. Paper knobs delayed that fade; they did not stop it. Teal does **not** “start increasing” toward 14.5 from here. It can bounce a few tenths if drink/table return to the window (finding 29). Geometric mean cannot leave ~2 while stone is 0.

Held-out recovered **0.88 (350k, wood 0) → 1.57 (400k, wood 30)**. That bag is not a comeback.

## Last-200 ticks

| env | gmean | wood | last-50 wood | table | drink |
|---|---|---|---|---|---|
| 318k | 2.09 | 38.5 | — | 7.0 | 14.5 |
| 350k | 1.71 | 29.5 | — | 3.5 | 10.0 |
| 400k | **1.60** | **21.0** | **12.0** | **1.0** | 7.0 |

## Failed alternatives

- Telling the user teal will start climbing after 400k. It cannot without stone.
- Killing because orange was 0.88. 400k held-out is 1.57; `ac_H` 0.19.
- Adding a wood bonus. Findings 01, 05, 19.

## Paper spin

Results: Table B.1 knobs held wood through 200–300k and then **the same unlearn as M15**, stretched in env steps. Last-N gmean drops when that third +1 fades, even while the WM is excellent (`recon_l1` 0.004).
