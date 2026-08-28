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
| Imagined λ-return vs reward on frozen WM | Evaluation: do not plot M4 return as Crafter skill |
| 4×400 CrafterReward return vs achievements | Evaluation: M5 0.10 is one unlock then die; geo-mean is M6 |
| 400-step collect cap vs Crafter 10k | Eval protocol: continue follows `discount`; do not train at 400 and score at 10k |
| Size-S 1M gmean 1.6 with death at 190 | Results: captioned baseline; 10k cap still idle; n=10 eval is variance |
| XL-from-scratch actor on unimix floor at 20k | Negative: 200M ≠ 14.5; prefill-after-one-episode is not the paper seed |
| Paper train_ratio 512 is ~2 env/s / 5-day 1M | Compute appendix: 16× M6 updates, not dashboard decay; seq 32 doubles torch seq-64 steps |
| XL + workstation ratio 32 still collapses | Negative with 14/15: finishable XL-from-scratch dies; paper ratio is 5 days |

## Index of findings

1. [The RSSM bypass](findings/01-rssm-bypass.md) — extra decoder + frozen `[h,z]` path.
2. [Reconstruction reduction inverts the loss](findings/02-recon-reduction.md) — mean L1 vs sum-MSE.
3. [Free-nats on the dynamics term](findings/03-free-nats-dyn.md) — prior freeze.
4. [Crafter pixel geometry](findings/04-crafter-pixel-geometry.md) — 7px tiles, nearest upsample, HUD paste, blob-mean.
5. [MSE ghosts vs crop losses](findings/05-mse-ghosts.md) — residue is not a missing head; 700k still coverage-limited.
6. [Paper vs reference KL scales](findings/06-kl-scale-mismatch.md) — Table 4 vs size-S YAML; 50k peak, 700k plateau ~2 nats.
7. [Diagnostics that lie](findings/07-lying-diagnostics.md) — resume plots, log-continue, embed-vs-`[h,z]`.
8. [Workstation recipe](findings/08-workstation-recipe.md) — seq 64, replay size, VRAM, Jupyter kernels, host-RAM strip OOM, M5 dashboard 18→1.1 env/s.
9. [Frozen-replay plateau](findings/09-frozen-replay-plateau.md) — 500k→700k is noise; DreamerV3’s 1M is env steps.
10. [Imagined λ-return is not skill](findings/10-imagined-return-not-skill.md) — M4 return ~2.6 is critic bootstrap; reward stays ~0.017/step.
11. [4×400 eval return is not a Crafter score](findings/11-eval-return-is-not-crafter-score.md) — M5 100k last 0.10 is one achievement then die.
12. [400-step cap is not Crafter](findings/12-episode-cap-is-not-crafter.md) — M5 lengths were under 400; timeout must not look like death.
13. [M6 1M still dies at 190](findings/13-m6-1m-still-dies-at-190.md) — held-out 1.64 / online 1.94; 10k cap idle; not DreamerV3 14.5.
14. [XL from scratch collapsed the actor](findings/14-xl-from-scratch-actor-collapse.md) — entropy on unimix floor by 20k; held-out 1.30→0.23, wake_up only.
15. [Paper train_ratio 512 is ~2 env/s](findings/15-xl-paper-ratio-is-five-days.md) — 16 WM+16 AC / 16 env; 1M is ~5.4 days on this box; not the M5 dashboard bug.
16. [XL + M6 update count still collapses the actor](findings/16-xl-workstation-actor-collapse.md) — ratio 32 is 26 env/s; ac_H ~0.09 from 20k; held-out 1.69→0.50. Do not grind to 1M.

Catalog (machine-readable): [`catalog.json`](catalog.json).

## Status of the runs cited here

- **Pre-reset** (`configs/m3_world_model.yaml`): two weeks of extra losses; 12k-step run, `recon_l1 ≈ 0.032`, no sprites on `[h,z]`.
- **M3 reset** (`configs/m3_dreamer_s.yaml`): size-S, one live decoder, paper four-term loss.
  - **50k steps, 2026-08-24.** `recon_l1` 0.326 → 0.0095; `kl_rep_raw` peaked 6.14 at step 16650, finished 3.49; reward correlation **r = 0.91**.
  - **700k steps, 2026-08-25.** Last-10k mean `recon_l1` **0.0045**, `kl_rep_raw` **1.96**; reward **r = 0.98**; open-loop std ratio **0.98**. Host RAM OOM at 628150 (matplotlib strip); resumed from 620k. Plateau after ~500k — do not push this buffer to 1M gradient steps (finding 09).
- **M4 actor-critic** (`configs/m4_actor_critic.yaml`): 20k steps on the frozen 700k WM, 2026-08-26. Notebook exit **PASS**. Last-2k imagined reward **0.017**/step vs λ-return **2.61** (finding 10). GIF: `results/m4_actor_critic/imagine_final.gif`.
- **M5 outer loop** (`configs/m5_outer_loop.yaml`): 100k env steps, 2026-08-26. Notebook plumbing **PASS**. Last eval return **0.10** = one achievement then death (finding 11), not geo-mean. Dashboard 18→1.1 env/s then **~38** after the log-cadence fix (finding 08). Live: `notebooks/08_train_outer_loop.ipynb`.
- **M6 baseline** (`configs/m6_baseline.yaml`): **1M env steps**, 2026-08-26. Protocol **PASS**. Held-out gmean **0.627 → 1.639**, online **1.941**. Collect mean length **190**, max **441** — 10k cap never binds (finding 13). Live: `notebooks/09_train_baseline.ipynb`.
- **M7 paper-online** (`configs/m7_paper_online.yaml`): XL ~198M from scratch. v1 at **100k** (2026-08-27) held-out **1.298 → 0.233**, entropy on unimix floor by 20k, wake_up only (finding 14). v2 (`checkpoints/m7_xl_paper`) cancelled at **~40k**: **~2.14 env/s**, entropy still alive (finding 15). v3 (`configs/m7_xl_workstation.yaml`, ratio 32) at **100k**: **26 env/s**, held-out **1.694 → 0.499**, entropy on the floor from ~20k (finding 16). Do not grind v3 to 1M. `notebooks/10_train_paper_online.ipynb`.

## Conventions

- Every claim needs a number or a still in `figures/`.
- Name latents `z_prior` / `z_posterior`. Never “the latent `z`.”
- Failed approaches stay in the archive. They are the paper’s negative results.
