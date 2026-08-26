# Research paper archive

This folder is **not the paper**. It is a living dump of *our* findings — mechanisms, negative results, and workstation-scale facts that are **not** restatements of DreamerV3 — so a later arXiv draft can be spun from evidence instead of memory.

Open [`index.html`](index.html) in a browser for the interactive hub (filterable findings, hoverable 50k/700k curves, figure galleries). The Markdown files under [`findings/`](findings/) are the cite-able writeups.

## What belongs here

- Training-graph identifiability failures (bypass paths, frozen weights, loss *reductions*).
- Crafter-specific pixel geometry (7px tiles, 14px HUD, 4×4 latent cells).
- KL floors applied to the wrong term, or treated as a setpoint.
- Negative results: crop/blob/HUD losses, extra decoders, “just train longer” vs coverage.
- Numbers from *this* box (RTX 5080 16 GiB, frozen random replay, size-S reset).

## What does not belong here

- “RSSM has `h` and `z`.” “We use KL balancing.” “Crafter is 64×64.”
- Anything already in Hafner et al. 2023 unless we *measured a disagreement* (e.g. Table 4 vs `NM512/dreamerv3-torch` `configs.yaml`).

## How to use this for the paper later

| Archive finding | Likely paper home |
|---|---|
| RSSM bypass / frozen decoder | Pitfalls / implementation appendix; “why extra recon heads are not free” |
| Mean vs sum image loss | Method: loss reduction; or a short identifiability note |
| `free_nats` on `kl_dyn` | Ablation or KL-balancing appendix |
| Nearest upsample vs sub-pixel | Architecture / Crafter geometry |
| MSE ghosts of rare movers | Limitations + why we did not add sprite losses |
| 50k KL peak then 700k plateau | Results figure; “floor ≠ setpoint”; frozen-replay compute |
| seq 64 thrash / host-RAM dashboard OOM | Compute / hardware appendix |

## Index of findings

1. [The RSSM bypass](findings/01-rssm-bypass.md) — extra decoder + frozen `[h,z]` path.
2. [Reconstruction reduction inverts the loss](findings/02-recon-reduction.md) — mean L1 vs sum-MSE.
3. [Free-nats on the dynamics term](findings/03-free-nats-dyn.md) — prior freeze.
4. [Crafter pixel geometry](findings/04-crafter-pixel-geometry.md) — 7px tiles, nearest upsample, HUD paste, blob-mean.
5. [MSE ghosts vs crop losses](findings/05-mse-ghosts.md) — residue is not a missing head; 700k still coverage-limited.
6. [Paper vs reference KL scales](findings/06-kl-scale-mismatch.md) — Table 4 vs size-S YAML; 50k peak, 700k plateau ~2 nats.
7. [Diagnostics that lie](findings/07-lying-diagnostics.md) — resume plots, log-continue, embed-vs-`[h,z]`.
8. [Workstation recipe](findings/08-workstation-recipe.md) — seq 64, replay size, VRAM, Jupyter kernels, host-RAM strip OOM.
9. [Frozen-replay plateau](findings/09-frozen-replay-plateau.md) — 500k→700k is noise; DreamerV3’s 1M is env steps.

Catalog (machine-readable): [`catalog.json`](catalog.json).

## Status of the runs cited here

- **Pre-reset** (`configs/m3_world_model.yaml`): two weeks of extra losses; 12k-step run, `recon_l1 ≈ 0.032`, no sprites on `[h,z]`.
- **M3 reset** (`configs/m3_dreamer_s.yaml`): size-S, one live decoder, paper four-term loss.
  - **50k steps, 2026-08-24.** `recon_l1` 0.326 → 0.0095; `kl_rep_raw` peaked 6.14 at step 16650, finished 3.49; reward correlation **r = 0.91**.
  - **700k steps, 2026-08-25.** Last-10k mean `recon_l1` **0.0045**, `kl_rep_raw` **1.96**; reward **r = 0.98**; open-loop std ratio **0.98**. Host RAM OOM at 628150 (matplotlib strip); resumed from 620k. Plateau after ~500k — do not push this buffer to 1M gradient steps (finding 09).

## Conventions

- Every claim needs a number or a still in `figures/`.
- Name latents `z_prior` / `z_posterior`. Never “the latent `z`.”
- Failed approaches stay in the archive. They are the paper’s negative results.
