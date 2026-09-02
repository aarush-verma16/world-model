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
| XL + workstation ratio 32 still collapses | Negative with 14/15: finishable XL-from-scratch dies; paper ratio is 5 days — **superseded by 17** |
| Six silent actor-critic divergences → one symptom | Implementation identifiability: off-by-one advantage is exactly 0-error at init; `imag_gradient_mix: 0.0` is load-bearing |
| ~200-step death is combat, not hunger | Results / Crafter mechanics: starvation clock is 338; modal 170 is zombies + sleep |
| 14.5 gap is recipe + sampler, not a missing head | Compute/protocol: ratio 32 vs 512 is 16× less replay; remaining code is `is_first` and CNN `blocks=1` |
| XL `blocks=2` fits 16 GiB at seq 32 | Compute: +18M weights, smoke peak 11.3 GiB; M13 is that knob at ratio 512 |
| 400k flat gmean is a skill island | Eval: last-N vs cumulative; do not roll back the decoder |
| Ratio 128 on the 400k actor did not buy survival | Negative: more replay on a sleep policy ≠ 512 from scratch |

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
16. [XL + M6 update count still collapses the actor](findings/16-xl-workstation-actor-collapse.md) — ratio 32 is 26 env/s; ac_H ~0.09 from 20k; held-out 1.69→0.50. Do not grind to 1M. **Its conclusion is superseded by 17.**
17. [The collapse was six actor-critic bugs](findings/17-actor-critic-alignment-bugs.md) — `V(s_t+1)` advantage baseline, off-by-one critic target, no slow critic, no discount weights, critic loss leaking into the actor, and a notebook that never forwarded `imag_gradient` so both XL runs summed a dynamics term DreamerV3 weights at 0.
18. [XL 5 env/s is 16 GiB paging](findings/18-xl-reinforce-graph-thrashes-16gib.md) — 15.7 GiB / 88 W, not a dead actor. `ac_H` 0.50 at 22k. Unused reinforce RSSM graph.
19. [~200-step death is combat, not hunger](findings/19-length-is-combat-not-hunger.md) — starvation clock 338; 66% of M8 lives die at 150–220 with drink still left.
20. [The 14.5 gap is a different experiment](findings/20-score-gap-is-recipe-not-a-missing-head.md) — ratio 32 vs 512, seq 32 vs 64, episode-bounded replay; do not put 1.36 next to 14.5.
21. [The 400k flat gmean is a skill island](findings/21-m9-400k-is-a-skill-island.md) — last-200 is 2.49; stone 0; held-out 2.82 is n=10. Continuing that actor is finding 22.
22. [Continuing the sleep-island actor does not buy survival](findings/22-m10-continue-did-not-buy-survival.md) — M10 +120k at ratio 128, length still 183, wake_up 97%; `RESUME=auto` loaded 500k.
23. [Ratio 128 from scratch is the island by 135k](findings/23-m11-128-is-the-island-at-135k.md) — last-200 ~2.0, length 177, wake 96.5%, zombie 1% vs M9’s 12.5% at the same env.
24. [XL `blocks=2` fits 16 GiB](findings/24-xl-blocks2-fits-16gib.md) — 215.6M WM, smoke peak 11.3 GiB at 16×32; M13 is ratio 512 + Table B.1 residuals. Do not load M12.
25. [Ratio 32 + blocks=2 is still the island at 826k](findings/25-m14-826k-is-the-island.md) — last-200 2.45→1.41, stone 0, length 180. Do not grind to 1M for 14.5.

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
- **M8 actor-critic fix** (finding 17): `m8_s_acfix` **200k DONE**. Held-out **1.386 → 2.068**, last `ac_H` **0.89** (min 0.74). XL `m8_xl_acfix2` stopped ~80k (online ~1.46, held-out 1.36 @ 50k, combat wall, V2 sampler). **Do not resume it.**
- **M9 streaming replay** (`configs/m9_xl.yaml`): XL from scratch to **~405k**. Cumulative gmean **2.56**, last-200 **2.49**, wake_up ~93%, stone ~0 (finding 21). Keep `ckpt_step_400000.pt`. **Do not resume it.**
- **M10 ratio 128** (`configs/m10_xl_r128.yaml`): continued those weights to **~523k**. Mean length still **~183**, held-out **2.25 → 1.76**, `wake_up` **97%** (finding 22). **Do not resume it.**
- **M11 from scratch** (`configs/m11_xl_r128.yaml`): ratio 128 to **~137k**. Last-200 **~2.0**, length **177**, `wake_up` **96.5%**, zombie **1%** (finding 23). **Stop — do not grind to 1M.**
- **M12 paper ratio** (`configs/m12_xl_r512.yaml`): from scratch, `train_ratio` **512**, `blocks=1`. ~2 env/s. Island-shaped at ~100–116k (last-200 ~1.4, wake saturated, stone 0). **Do not resume it into M13.**
- **M13 blocks=2** (`configs/m13_xl_r512_b2.yaml`): same 512 schedule, encoder/decoder **blocks=2**. Smoke **11.3 GiB** (finding 24). Interrupted around **106k** (last-200 ~1.5, stone 0) to leave the 5-day clock. **Do not resume it.**
- **M14 workstation** (`configs/m14_xl_r32_b2.yaml`): from scratch, `train_ratio` **32**, `blocks=2`. At **~826k**: online **2.24**, last-200 **1.41**, length **180**, stone **0** (finding 25). Same island as M9. **Do not put this next to 14.5.**

## Conventions

- Every claim needs a number or a still in `figures/`.
- Name latents `z_prior` / `z_posterior`. Never “the latent `z`.”
- Failed approaches stay in the archive. They are the paper’s negative results.
