# World Model Project — Full Milestone Plan (Scratch to Finish)

This is the master plan. The 3-week day-by-day guide covers *daily* mechanics; this document
covers the project at the *milestone* level — what must be true before you're allowed to move
forward, what can go wrong at each stage, and how the whole thing chains together with
dependencies. Treat each milestone as a gate: you do not start the next one until the current
one's exit criteria are genuinely met, not "basically working."

---

## 0. How This Plan Is Structured

Each milestone has the same six fields:

- **Goal** — the one-sentence purpose of this stage
- **Depends on** — which prior milestone(s) must be fully complete first
- **Tasks** — the concrete build work
- **Exit criteria** — the specific, checkable conditions that mean "done" (not vibes)
- **Common failure modes** — what actually goes wrong at this stage, and the fix
- **Artifact** — what you commit/tag/produce as proof this milestone is complete

Ten milestones total, M0 through M9, roughly mapping to the 3-week timeline but described here
as dependency-gated stages rather than fixed days — some will take longer than a day, some
less, and that's fine as long as exit criteria are met before moving on.

---

## Milestone Dependency Graph

```
M0 (Setup)
 └─> M1 (Perception: Encoder/Decoder)
      └─> M2 (Dynamics: RSSM core)
           └─> M3 (Full World Model Loss + Replay Buffer)
                └─> M4 (Imagination + Actor-Critic)
                     └─> M5 (Full Outer Loop Integration)
                          └─> M6 (Baseline Result on Crafter)
                               ├─> M7 (Ablation Study)
                               └─> M8 (Analysis + Paper Writing)
                                    └─> M9 (Docs, Repo Polish, Release)
```

Nothing here is parallelizable in a meaningful way for a solo builder except M7 and M8, which
can overlap once M6 is complete (you can start drafting the paper's intro/related-work while
ablation runs are training in the background).

---

## Milestone 0 — Environment & Repo Foundation

**Goal:** a working, verified development environment with nothing left to chance before real
code gets written.

**Depends on:** nothing (starting point).

**Tasks:**
- Conda environment, PyTorch install, CUDA verification
- Gymnasium + Crafter + MiniGrid install and smoke test
- Repo skeleton, `.cursor/rules`, `PROJECT_BRIEF.md` committed
- TensorBoard working locally (log one dummy scalar and confirm it renders)

**Exit criteria:**
- `torch.cuda.is_available()` returns `True` and `nvidia-smi` sees the GPU
- `CrafterReward-v1` resets and steps, returns `(64, 64, 3)` observations
- A dummy TensorBoard scalar log is viewable in the browser
- Repo pushed to GitHub with a real README stub (not the default)

**Common failure modes:**
- `pip install torch` from PyPI on Windows installs a **CPU-only** wheel, so
  `cuda.is_available()` is False even though `nvidia-smi` works. Mitigation:
  install from `https://download.pytorch.org/whl/cu128` (see README /
  `scripts/setup_windows.ps1`).
- RTX 50-series (Blackwell, sm_120) on a CUDA 12.4/12.1 build raises
  "no kernel image is available". Mitigation: CUDA 12.8+ PyTorch, recent driver.
- Crafter install conflicts with a system-level OpenGL/rendering dependency —
  mitigation: install inside the conda env only, never system Python.

**Artifact:** tag `v0.0-setup-complete`.

---

## Milestone 1 — Perception (Encoder + Decoder)

**Goal:** prove the compression half of the system works before touching anything recurrent.

**Depends on:** M0.

**Tasks:**
- CNN encoder: 64x64x3 → embedding vector
- CNN decoder: embedding → reconstructed 64x64x3
- Train on pure reconstruction loss only (no RSSM, no reward, no actions yet) using a
  small buffer of random-policy Crafter frames
- Log reconstruction loss curve + periodic sample reconstructions to TensorBoard as images

**Exit criteria:**
- Reconstruction loss visibly decreases over training and plateaus at a low value
- Decoded images are visually recognizable as the corresponding input frame (blurriness is
  fine and expected — objects/positions should be roughly right)
- No shape errors across a full batch/epoch

**Common failure modes:**
- Normalizing pixel values inconsistently between encoder input and decoder output (e.g.
  forgetting to rescale to `[-1, 1]` or `[0, 1]` consistently) — produces reconstructions
  that are uniformly wrong in a specific, diagnosable way (washed out, inverted, clipped).
- Using too small a random-policy dataset — a random Crafter agent rarely explores past the
  first biome, so your encoder overfits to a narrow visual distribution. Collect from several
  hundred random episodes, not a handful.

**Artifact:** tag `v0.1-encoder-decoder-working`, saved sample reconstructions in `results/`.

---

## Milestone 2 — Dynamics Core (RSSM)

**Goal:** the recurrent state-space model can take a real sequence and produce internally
consistent prior/posterior latents without yet being trained end-to-end on the full loss.

**Depends on:** M1.

**Tasks:**
- GRU-based deterministic state `h`
- Discrete categorical stochastic latent `z` (DreamerV3's 32x32; M2's forward-pass
  config may still use 16x16 as a cheap shape check, which is not a training recipe)
- Posterior head: `z_posterior` from `h` + real encoder embedding
- Prior head: `z_prior` from `h` alone
- Straight-through gradient estimator for the discrete sampling step
- Forward-pass-only test: feed a real (obs, action) sequence through, confirm shapes and that
  the recurrence doesn't blow up (NaN/Inf) over 50+ steps

**Exit criteria:**
- A full batch of `[batch, time, ...]` sequences passes through the RSSM with no shape errors
- `z_prior` and `z_posterior` are distinct tensors with correctly separated naming throughout
  the code (no variable named just `z`)
- Running the recurrence for 100+ steps produces no NaNs even before any real training (a
  broken recurrence will blow up even on random weights — catch this now, not after days of
  wasted training)

**Common failure modes:**
- Backpropagating through the discrete sampling step without a straight-through estimator —
  gradients silently vanish and nothing trains, with no error thrown. This is the single most
  common silent-failure bug in RSSM implementations — verify gradients are actually flowing
  into the categorical logits before moving on (check `.grad` is non-zero after a backward pass).
- Confusing which state feeds forward — deterministic `h` must always be updated by the GRU
  using the *previous* `z` and action, not the current one. An off-by-one here trains but
  produces a subtly wrong (and hard to detect) dynamics model.

**Artifact:** tag `v0.2-rssm-forward-pass-verified`.

---

## Milestone 3 — Full World Model (Loss, KL Balancing, Replay Buffer)

**Goal:** the complete world model — encoder, RSSM, decoder, reward head, continue head —
trains end-to-end on real replayed experience and produces a genuinely predictive model.

**Depends on:** M2.

**Tasks:**
- Reward prediction head, continue/discount prediction head
- Full loss: reconstruction + reward + continue + KL(posterior || prior) with KL balancing
  (asymmetric weight between the two KL directions)
- Proper sequential replay buffer (returns contiguous length-L chunks, not i.i.d. samples —
  the RSSM needs temporal continuity to train correctly)
- Collect a larger, more diverse dataset (a few thousand steps across many episodes, still
  random or lightly-scripted policy at this stage)

**Exit criteria:**
- All four loss terms decrease over training (log each separately in TensorBoard, not just
  the sum — a summed loss can look fine while one term has collapsed)
- KL term does not collapse to exactly zero (posterior collapse — model ignores the latent
  entirely and just uses `h`) or explode unboundedly (posterior ignores the prior, imagination
  becomes impossible)
- Reward predictions on held-out real sequences correlate reasonably with actual rewards

**Common failure modes:**
- Posterior collapse (KL → 0): the decoder becomes powerful enough to ignore `z` and
  reconstructs from `h` alone. Fix via KL balancing weight tuning — push more weight onto
  training the prior toward the posterior rather than the reverse direction.
- Reward head trained on too few positive-reward examples (Crafter rewards are sparse under
  a random policy) — reward predictions collapse to always predicting near-zero. This is
  expected at this stage and is fixed later once the policy starts exploring more
  purposefully in M5 — don't over-optimize for this with a random-policy dataset alone.

**Artifact:** tag `v0.3-world-model-trained`, loss curves saved to `results/`.

---

## Milestone 4 — Imagination + Actor-Critic

**Goal:** an actor and critic that train purely on imagined rollouts generated by a frozen
world model, with no real environment interaction during this training phase.

**Depends on:** M3.

**Tasks:**
- Imagination rollout function: start from a real latent state, roll forward N steps
  (default 15) using `z_prior` only, with actions chosen by the current actor at each step
- Actor MLP (policy) and critic MLP (value function) operating on `[h, z]`
- Lambda-return computation over imagined trajectories
- Return normalization (percentile-based scaling, DreamerV3-style) for training stability
- Backprop directly through the differentiable imagined rollout into actor + critic weights

**Exit criteria:**
- A full imagined rollout runs end-to-end and produces a coherent chain of predicted rewards
  (not NaN, not wildly oscillating)
- Actor-critic loss decreases over training on a fixed, frozen world model
- Decoded imagined rollouts (pass the imagined `z` states through the decoder for
  visualization only) look plausible for a handful of steps before degrading — some
  degradation over the horizon is expected and even informative, total incoherence from step
  1 is not

**Common failure modes:**
- Forgetting to detach/freeze the world model's gradients during actor-critic training —
  gradients meant for the actor leak into and corrupt the world model weights.
- Reward scale mismatches between what the world model predicts and what the critic expects —
  without return normalization, a few large-reward Crafter achievements can dominate the loss
  and destabilize training entirely.

**Artifact:** tag `v0.4-imagination-actor-critic-working`, sample imagined-rollout GIF in
`results/`.

---

## Milestone 5 — Full Outer Loop Integration

**Goal:** the complete Dreamer loop runs continuously and produces visible learning progress
over time, not just working components in isolation.

**Depends on:** M4.

**Tasks:**
- Wire the full cycle: act in real env with current policy → append to replay buffer → train
  world model on replay → freeze world model → train actor-critic on imagination → repeat
- Add periodic real-environment evaluation (run the current policy for real, log actual
  achieved reward/Crafter score, not just imagined predictions)
- Add checkpointing (save/restore full model + optimizer state)

**Exit criteria:**
- The loop runs continuously for an extended period (hours) without crashing, OOM-ing, or
  diverging (NaN losses)
- Real-environment evaluation reward trends upward over the course of training, even if
  slowly — this is the first point in the project where you should see actual learning, not
  just stable losses
- Checkpoint can be saved and reloaded and continues training/evaluating correctly

**Common failure modes:**
- Memory creeping upward over hours of training (a leak in the replay buffer or logging code
  that keeps references alive) — profile memory on a short run before committing to an
  overnight run.
- Evaluation reward flat despite decreasing losses — usually means the world model is
  learning to predict well but the actor isn't exploring enough; check that action selection
  during real-environment data collection includes some exploration noise/entropy, not a
  fully greedy policy from the start.

**Artifact:** tag `v0.5-full-loop-integrated`.

---

## Milestone 6 — Baseline Result on Crafter

**Goal:** one clean, complete, honestly-reported baseline training run with a real Crafter
score, directly comparable to published numbers.

**Depends on:** M5.

**Tasks:**
- Hyperparameter pass suited to the 16 GiB VRAM budget (batch size, sequence length;
  keep the 32x32 latent — that is the recipe, not a memory knob)
- Implement the Crafter evaluation harness matching the paper's metric (achievement
  percentages + geometric mean score)
- Run the full baseline training run to completion (likely several hours to overnight)
- Record final score, reward curve, and a handful of qualitative rollout recordings

**Exit criteria:**
- A completed run with a final Crafter score computed via the standard metric
- Reward curve and score are saved, reproducible from the exact logged config
- You can honestly state your result is real and specific, even if far below published
  DreamerV3 numbers (this is expected — you're running at a fraction of the compute)

**Common failure modes:**
- Comparing against published numbers without matching the evaluation protocol
  (e.g. different episode counts, different achievement definitions) — produces a comparison
  that looks better or worse than it actually is. Match the metric definition exactly before
  quoting a comparison in the paper.

**Artifact:** tag `v1.0-baseline-result`. This is the most important tag in the whole project —
everything downstream depends on this being a real, verified, reproducible number.

**Inserted next (2026-08-27):** M6's agent still died at mean length 190 on
size-S (~19M). Before the ablation below, chase the paper *setup* with
`configs/m7_paper_online.yaml` / `notebooks/10_train_paper_online.ipynb`:
XL ~200M from scratch, fresh actor, 10k-step lives. Size-S keep-WM fallback
is `configs/m7_s_reset_actor.yaml`. Do not grind the M6 checkpoint. Ablation
stays this milestone once the question is no longer "they just die."

---

## Milestone 7 — Ablation Study

**Goal:** a controlled, defensible empirical result answering one specific research question,
under a fixed compute budget.

**Depends on:** M6.

**Tasks:**
- Choose the ablation axis based on what the baseline actually revealed (pick one):
  - Discrete latent grid size (e.g. 8x8 vs 16x16 vs 32x32) vs. performance-per-compute
  - Imagination horizon length (5 vs 15 vs 30 steps) vs. compounding prediction error
  - Latent-space (RSSM/Dreamer-style) vs. representation-space (JEPA-style, no decoder)
    prediction — more implementation work, most novel
- Build a config system so runs differ by exactly one variable, all else held fixed
- Launch all runs with identical seeds where possible
- Produce sample-efficiency curves and (if relevant) imagination-error-growth plots

**Exit criteria:**
- All planned ablation runs complete under identical compute budgets
- Plots clearly show the swept variable's effect, with axes and scales that don't visually
  exaggerate small differences
- Raw logged data (not just plots) is saved per run for reproducibility

**Common failure modes:**
- Changing more than one variable between runs "for convenience" (e.g. also adjusting batch
  size when sweeping latent size because it's more efficient) — this invalidates the ablation
  as a clean causal comparison. Hold everything else exactly fixed, even if it costs some
  efficiency.
- Only running one seed per condition and treating the difference as conclusive — at minimum,
  note this explicitly as a limitation; if time allows, run 2-3 seeds for the conditions that
  matter most to your conclusion.

**Artifact:** tag `v2.0-ablation-complete`, all plots and raw data in `results/`.

---

## Milestone 8 — Analysis & Paper

**Goal:** a complete, honestly-scoped written paper documenting the method and the ablation
finding.

**Depends on:** M6 fully, M7 substantially (can start drafting intro/related work once M6 is
done, in parallel with M7's runs executing).

**Tasks:**
- Write Abstract, Introduction, Related Work (using the paper lineage: Ha & Schmidhuber →
  PlaNet → Dreamer → DreamerV2 → DreamerV3, plus JEPA if relevant to your ablation choice)
- Write Method section precise enough to be reproducible from the text alone
- Write Experimental Setup, Results (with your plots), Discussion
- Write Limitations honestly — single machine, likely single/few seeds, single environment,
  hardware-specific findings that may not transfer to cluster-scale compute

**Exit criteria:**
- Every claim in Results is directly backed by a plot or number you actually produced
- Limitations section is specific, not boilerplate ("we only tested one environment" is
  useful; "results may vary" is not)
- A reader unfamiliar with the project could reproduce your setup from the Method section alone

**Common failure modes:**
- Overclaiming novelty or performance relative to DreamerV3 — the credible framing is "a
  controlled small-scale study," not "an improvement on Dreamer." Reviewers and readers trust
  honestly-scoped claims far more than inflated ones.

**Artifact:** finished paper draft in `paper/`, posted to arXiv once complete (does not require
peer review to be a legitimate, citable artifact).

---

## Milestone 9 — Documentation, Repo Polish, Release

**Goal:** the repository and docs site are genuinely presentable to someone landing on them
cold — a reviewer, an admissions reader, a future employer.

**Depends on:** M8.

**Tasks:**
- Finish MkDocs site (`index`, `concepts`, `architecture`, `experiments` log, `results`, link
  to paper)
- Finish README: summary paragraph, imagined-rollout GIF, architecture diagram, results table
  vs. published numbers, setup instructions
- Verify setup instructions actually work from a **clean clone** on a fresh checkout — test
  this yourself, don't assume
- Final tag: `v3.0-release`

**Exit criteria:**
- Docs site deployed and viewable on GitHub Pages
- A clean clone + the README's setup steps alone reproduce a working environment
- All milestone tags present in repo history, forming a clean record of the project's
  progression

**Common failure modes:**
- Setup instructions that work "on my machine" because of leftover global state (a package
  installed outside the conda env, a cached dataset) but fail on a truly clean clone — this is
  exactly why the clean-clone test in this milestone is not optional.

**Artifact:** tag `v3.0-release`. Project complete.

---

## Overall Timeline Reference

| Milestone | Approx. duration | Cumulative |
|---|---|---|
| M0 — Setup | 1 day | Day 1 |
| M1 — Encoder/Decoder | 1 day | Day 2 |
| M2 — RSSM core | 1-2 days | Day 3-4 |
| M3 — Full world model | 2 days | Day 5-6 |
| M4 — Imagination + actor-critic | 2 days | Day 7-9 |
| M5 — Full loop integration | 2 days | Day 10-11 |
| M6 — Baseline result | 2-3 days | Day 12-14 |
| M7 — Ablation study | 3 days | Day 15-17 |
| M8 — Paper (overlaps M7) | 3 days | Day 17-19 |
| M9 — Docs & release | 2 days | Day 20-21 |

This lines up with the earlier 3-week day-by-day guide — that document tells you what to do
each day; this document tells you what must be true before you're allowed to consider a stage
finished, regardless of which day you happen to hit it on.

---

## Global Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CUDA / driver / wrong torch wheel | Medium | High | Verify early (M0) with `nvidia-smi` + `smoke_cuda_step.py`; never install torch from PyPI on Windows |
| Posterior collapse in RSSM | Medium | High | KL balancing, monitor KL term separately in logs, not just summed loss |
| CUDA OOM on 16 GiB during the XL CNN + 3 decode heads | Medium | Medium | Default is batch 16 × seq 32 + bf16; drop batch_size if a smoke step OOMs. Desktop compositor already uses ~1.5 GiB. Replay stays in system RAM. |
| Ablation invalidated by accidentally changing 2+ variables | Medium | High | Config-diff check before launching each ablation run — literally diff the yaml files |
| Running out of time before M8/M9 | Medium | Medium | M6 (baseline) is the non-negotiable milestone — if time runs short, a working baseline with a smaller/simpler ablation is a better outcome than an unfinished larger one |
| Overclaiming results in the paper | Low | High | Limitations section written honestly, framed as a controlled small-scale study throughout |

---

## Definition of Done (Whole Project)

The project is complete when all of the following are simultaneously true:

1. `v3.0-release` is tagged and the repo is public.
2. A clean clone + README instructions produces a working environment with no undocumented steps.
3. The baseline result (M6) and ablation result (M7) are both real, reproducible, and their
   raw data is saved in the repo.
4. The paper (M8) makes no claim unsupported by a plot or number in `results/`.
5. The docs site (M9) is live and a stranger could understand the project from it alone,
   without reading the code.