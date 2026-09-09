# Finding 33 — M17 200k: teal 1.83 is the same island; paper knobs did not open stone

**Status:** measured on M17-XL `m17_xl_paper` at 199936 / 1M (1183 lives), 2026-09-09  
**Kind:** planned ~180k skill look after finding 32; negative on “Table B.1 seq/GRU/replay walks to 14.5”  
**Evidence:** `results/m17_xl_paper/collect_episodes.jsonl`, `eval_metrics.json`, `train_metrics.json`; `ckpt_step_200000.pt`

## Claim

The Crafter score is **flat because the achievement set is flat**, not because the trainer died. Last-200 at 200k is wake **99 / sapling 92 / plant 78 / wood 34 / table 4 / drink 10 / stone 0 / pickaxe 0**. Last-200 gmean **1.82** (100k was **1.85**). Online **1.76**. Length **177**. `recon_l1` **0.005**. `ac_H` **0.19** (min **0.17** at 174k) — still above the ~0.08 unimix floor; last logs wiggle 0.18–0.22. Collect entropy **0.14**.

100–200k wood **36%** vs 0–100k **33%**. Paper seq 64 / GRU 4096 / replay 1e6 / actor-from-0 **held wood** past the M15 136k unlearn (finding 26) and did **not** open stone. Geometric mean of this set saturates in **1.7–1.9**. It will not climb toward 14.5 at 500k or 1M unless stone becomes common.

Held-out sawtooth **2.75 (25k) → 1.96 (100k) → 1.17 (200k, wood 30 / table 0 / stone 0)**. Finding 13 lottery. Do not read 1.17 as collapse.

## Windows (collect)

| env | n | mean len | wood | table | stone | pick | last-200 gmean @ end |
|---|---|---|---|---|---|---|---|
| 0–100k | 614 | 177 | 32.9 | 3.9 | 0 | 0.2 | **1.85** @ 106k (finding 32) |
| 100–200k | 569 | 178 | **35.7** | 3.7 | **0** | 0 | **1.82** @ 200k |
| last-200 | 200 | 177 | 33.5 | 4.0 | **0** | 0 | 1.82 |

## Failed alternatives

- Waiting for teal to leave 1.8 because “the paper climbs after 200k.” Without stone there are no remaining gmean degrees of freedom (finding 30).
- Killing because `ac_H` is 0.19. Floor is ~0.08; wood is still 34%.
- Captioning held-out 1.17 as the policy dying. 100k was 1.96 with table 40 in ten lives.
- Grinding M17 to 1M so 1.8 can sit next to 14.5.

## Paper spin

Results: matching leftover Table B.1 knobs on 16 GiB (seq 64, GRU 4096, replay 1e6, actor on from 0, batch 8) **reproduces the wood-hold**, not the 14.5 tree. The 200k plateau is occupancy of four cheap +1s. The missing mechanism is still how stone becomes common.
