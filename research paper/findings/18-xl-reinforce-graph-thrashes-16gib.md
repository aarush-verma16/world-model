# Finding 18 — XL at `train_ratio` 32 was paging 16 GiB, not “the AC fix failed”

**Kind:** workstation constraint / implementation identifiability  
**Evidence:** `nvidia-smi` during M8-XL ~22k: **15750 / 16303 MiB**, GPU util 100%, **power 88 W**, ~5 env/s. M7 XL same ratio was **~26 env/s**. M8-S 200k was **~43 env/s**. Live `ac_H` **0.50–0.57** (unimix floor is ~0.08).

## Claim

The 1 hour / 20k-step crawl is **WDDM paging**, not a broken actor-critic. The card is full. 88 W at 100% util is copies, not matmuls. M7 XL smoke peaked **12.9 / 13.7 GiB** and ran at 26 env/s. Crossing ~15.5 GiB on a 16 GiB 5080 that already shares the desktop is a 5× wall-clock hit.

The charts at 22k looking “unfixed” are the usual early-Crafter dashboard: jagged hunger deaths, n=10 held-out, late-tree bars at 0. They are **not** the M7 collapse (entropy glued to 0.08, held-out falling, wake-up only). Online gmean climbing through 1.2 with wood/sapling/table unlocks at 22k is the size-S AC-fix result showing up on XL from scratch, slowly.

## What was allocating the extra graph

Imagination **detaches** the replay posterior, then still ran XL `encode` / `observe` and 15 `img_step`s **with autograd enabled**. Crafter uses `imag_gradient=reinforce`, so that RSSM graph is never backward’d. The STE action still `requires_grad`, so `img_step` saved activations for a path the loss then discarded. Plus the encoder backward for starts that are immediately `.detach()`’d.

Fix: observe under `no_grad`; skip the RSSM graph unless `imag_gradient` is `dynamics` or `both` with mix > 0.

## Failed alternatives (do not do these)

- Rolling back to M7 / M6. M8-S already held entropy; this XL 22k run is not on the floor.
- Reading length vs 190 or empty late-tree bars at 22k as “the fix did not take.”
- Grinding this 5 env/s job to 1M (~55 h) while nvidia-smi says 15.7 GiB / 88 W.

## Paper spin

Hardware appendix next to findings 08 and 15: a 16 GiB torch reimplementation can look “5× slower than last week” when the unused `dynamics` graph plus WDDM clients push the process over the paging line. Caption remains XL, seq 32, `train_ratio` 32, `imag_gradient=reinforce`.
