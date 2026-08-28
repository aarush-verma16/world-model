# Finding 15 — Paper `train_ratio` 512 on this box is ~2 env/s

**Status:** measured on M7 v2 (`m7_xl_paper`) at ~40k / 1M, 2026-08-27  
**Kind:** workstation constraint; not an XL bug and not the M5 dashboard decay  
**Evidence:** notebook status `~2.14 env/s` at env 39936; M6 size-S 16/1/1 was **~37 env/s** (finding 13); `loop_updates` math in `src/training/outer_loop.py`

## Claim

Chasing DreamerV3 Crafter at `train_ratio` 512 on this workstation is a **multi-day** 1M, not an overnight 1M. That is the outer-loop schedule, not a hung Jupyter kernel and not “XL is 10× slower.”

M6 did **1 WM + 1 AC** update per 16 env steps (size-S) at **~37 env/s**. This run does **16 WM + 16 AC** per 16 env steps (XL, seq 32). 37 / 16 ≈ **2.3**, and the live status is **2.14 env/s**. The 16× GPU work is the whole story. XL vs size-S is a rounding error on top of that.

`log_every: 256` / `dashboard_every: 1000` are already the finding-08 cadence. This is **not** the M5 18→1.1 env/s matplotlib bug.

## Numbers (2026-08-27, still running)

| | M6 1M | M7 v2 at 40k |
|---|---|---|
| size | S (~19M WM) | XL (~198M WM) |
| updates / 16 env | 1 + 1 | 16 + 16 |
| env/s | ~37 | **~2.14** |
| 1M wall clock | hours | **~130 h / 5.4 days** at this rate |
| 40k wall clock | — | **~5.2 h** (notebook timer) |

Seq 32 vs paper-torch seq 64 also **doubles optimizer steps** relative to `NM512/dreamerv3-torch` Crafter. Their `Every(batch·length / train_ratio)` with 16×64 / 512 trains **once per 2 env steps** (8 WM+AC per 16 env). We keep the same *transitions* per env step, so 16×32 / 512 trains **once per env step** (16 WM+AC per 16 env). Dropping seq to fit 16 GiB made the 1M slower, not faster.

Published “Crafter in 4–24 hours” numbers are JAX / cluster DreamerV3, not this torch XL notebook.

## Collect at 40k is not the v1 collapse

230 lives to env 39968. Mean length still **~170** (M6 died at 190; 10k cap idle). Entropy on the screenshot **0.56**, not the unimix floor (0.08). Online gmean **0.69** is a running mix, but the jsonl unlocked real items after 25k:

| env window | mean len | non-wake lives | notable unlock rates |
|---|---|---|---|
| 0–25k | 161–185 | 0–3 / ~28 | wake_up 67–96%; drink/wood rare |
| 25–30k | 186 | 6/27 | drink 15%, wood 7% |
| 30–35k | 174 | **18/29** | wood **48%**, drink 24%, sapling 10%, one table, one zombie |
| 35–40k | 166 | 13/30 | drink 33%, wood 13%, sapling 13% |
| last 40 lives | 171 | — | drink 38%, wood 28%, sapling 15%, wake 68% |

The orange **1.387** bar chart is still **eval at step 0**. Do not read it as a new held-out.

## Cancelled (same day)

The 1M was stopped at **~40k**. Five hours for 30–40k env steps is unusable on this
desk, even though it is the intended 16× schedule. Next live config is
`configs/m7_xl_workstation.yaml`: **same seed recipe**, `train_ratio` **32**
(1+1 per 16 env, M6's update count), new dirs. Do not resume
`checkpoints/m7_xl_paper`. Caption that run as ratio-32, not paper 512.

The notebook title “episode length (skill gate)” plus a 190 line was a **lying
caption**. M6's *finished* 1M mean length is 190 (finding 13). Dipping under
that line at 40k is the same hunger death, not a unique crash. Watch `ac_H`
and unlocks. At 40k this cancelled run had entropy **0.56** and first wood/drink
rates; v1 had entropy **0.08** and wake_up only.

## Failed alternatives (do not do these because it is “too slow”)

- Kill the run at 40k because 1M is days, then judge it solely by length vs 190.
  Cancel for wall-clock is fine; calling it uniquely “shit” vs M6 is not — M6
  also died at 190 after 1M. The 40k jsonl had wood/drink; v1 did not.
- Drop `train_ratio` mid-run and keep the same checkpoint dir. That is a new recipe; caption it or start a new dir.
- Silently cut to M6’s 16/1/1 and still call it the paper outer loop. M6 already showed size-S + that ratio dies at 190.
- Blame XL params for the 5-day clock. The 16× update count is enough.

## Paper spin

Hardware appendix next to finding 08: seq 32 was a VRAM choice that **also** doubled Crafter gradient steps vs the torch reference at seq 64. A 1M at `train_ratio` 512 on 16 GiB torch is a 5-day experiment; this box cancelled it. The follow-up is ratio 32 in a new dir, captioned as such. Do not put a 14.5 next to a 40k screenshot.
