# Finding 31 — Table B.1 GRU 4096 + seq 64 fits 16 GiB only at batch 8

**Status:** smoked 2026-09-07 on RTX 5080 16 GiB after finding 18’s unused-graph cut  
**Kind:** workstation constraint that changed the M17 recipe; not a 14.5 result  
**Evidence:** `scripts/count_params.py --smoke --size xl_paper`; `configs/sizes/dreamer_xl_paper.yaml`

## Claim

Matching the leftover DreamerV3 knobs (seq **64**, `dyn_deter` **4096**, `dyn_hidden` **1024**, `blocks=2`) is possible on this card **after** the reinforce-graph fix (finding 18). It is **not** possible at the paper batch. `start_mode=all` × batch **16** × seq **64** pages: 15.9 GiB reserved, ~74 W, no step completes. Batch **8** × seq 64 bf16 is **11.40 / 11.80 GiB** and finishes a WM+AC step in ~11 s. fp32 at the same batch is **13.28 / 14.22 GiB** — smoke-legal, too tight for eval + dashboard + the Windows compositor (finding 08).

`train_ratio` 512 is preserved: `loop_updates` gives **16** WM+AC per 16 env at batch 8 × seq 64 (same transition count as M16’s 16 × 16 × 32). WM is **227.4M** vs the paper’s ~200M count (our LN-GRU + 2-layer prior is still heavier; hidden 1024 instead of 2560 avoided the 336M 4096 variant from finding 20).

Do not set batch 16 on `m17_xl_paper`. Do not treat bf16 as the thing that will block 14.5; jax Crafter is also bf16. Do not load M16 into this size.

## Smoke

| recipe | result |
|---|---|
| 4096 / hidden 1024 / blocks=2 / seq 64 / batch 16 / bf16 / `all` | page: 15.9 GiB, ~74 W, hung |
| same / **batch 8** / bf16 | **PASS 11.40 / 11.80 GiB**, 10.7 s |
| same / batch 8 / **fp32** | PASS 13.28 / 14.22 GiB, 8.1 s |

## Failed alternatives

- Jumping 4096 + seq 64 + batch 16 + fp32 in one yaml and debugging the OOM (finding 20).
- Calling 227M “not XL” and shrinking the 32×32 latent.
- Resuming `checkpoints/m16_xl_r512_acwarmup` into the 4096 graph.

## Paper spin

Compute appendix next to findings 08 / 15 / 18 / 24: the remaining Table B.1 knobs fit at seq 64 if batch drops 16→8. The live M17 run is that recipe, captioned, not a silent 14.5 clone.
