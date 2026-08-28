# 17 — The M7 actor collapse was six actor-critic implementation bugs, not model size or train_ratio

**Claim.** Two consecutive XL Crafter runs drove actor entropy to the unimix
floor (0.085–0.096 nats, floor ≈ 0.08) from ~20k env steps, and we spent both
runs debating `train_ratio` and model size. The cause was in `ac_step` /
`imagine_ahead`: our advantage used `V(s_{t+1})` as the baseline for the action
taken at `s_t`, the critic head was trained one step out of alignment with its
own target, there was no slow critic, no discount weights, the critic loss
leaked gradient into the actor through the imagined dynamics, and — decisively —
the notebook never forwarded `imag_gradient`, so both runs ran a mode that adds
the straight-through dynamics gradient at **full** weight, which DreamerV3
weights at **zero** for discrete actions.

**Why it is not the paper.** Every item is a re-derivation of
`ImagBehavior._compute_target` / `_compute_actor_loss` from the DreamerV3
reference. The publishable content is not "we implemented DreamerV3 correctly";
it is that all six bugs are invisible in the loss curves and present as a
*single* symptom (entropy → unimix floor) that looks exactly like an
exploration or capacity problem. Two 100k-step runs were spent on the wrong
hypothesis.

## Evidence

### The symptom pointed at the wrong knob

| run | size | `train_ratio` | outcome |
| --- | --- | --- | --- |
| `m7_xl_paper` | XL 198M | 512 | cancelled at 40k, ~2.1 env/s (~5.4 d for 1M), entropy still 0.56 |
| `m7_xl_workstation` | XL 198M | 32 | 100k, ~26 env/s, entropy on the floor from ~20k, held-out gmean 1.694 → 0.499 |

Ratio 512 → 32 fixed the wall clock and made the collapse *earlier*, so the
conclusion at the time was "finishable XL-from-scratch collapses on this box"
(finding 16). That conclusion was wrong: both runs shared the same
actor-critic.

### The six divergences from the reference

Verified against `NM512/dreamerv3-torch` `models.py` / `tools.py` /
`configs.yaml` (the `crafter:` block).

1. **Advantage baseline off by one.** `imagine_ahead` returned reward, cont and
   value decoded from the state *after* each `img_step`, and `ac_step` used
   `returns - value` with both at the same index. Since
   `returns[t] = r(s_{t+1}) + γc·[(1-λ)V(s_{t+1}) + λG_{t+1}]` is the target for
   `V(s_t)`, subtracting `V(s_{t+1})` cancels the bootstrap and leaves roughly
   `reward + (γc - 1)·V(s_{t+1})` — a quantity whose sign barely depends on
   which action was sampled. The reference uses `base = value[:-1]`, i.e.
   `V(s_t)`.
2. **Critic target off by one.** The same shift trained
   `critic(feat(s_{t+1}))` against the λ-return for `V(s_t)`, so the critic
   learned a value function shifted one step, which feeds straight back into 1.
3. **No slow critic.** `critic: {slow_target: True, slow_target_fraction: 0.02,
   slow_target_update: 1}` is a DreamerV3 default. We had no EMA copy, so the
   critic regressed only onto a target containing its own bootstrap.
4. **No discount weights.** The reference weights actor and critic losses by
   `∏_{j<t} γ·cont(s_j)`. We used a plain `.mean()`, so imagined steps after a
   predicted death were trained at full weight — in Crafter, where the agent
   dies constantly, that is a large fraction of every rollout.
5. **Critic loss leaked into the actor.** The critic read non-detached features,
   and actor + critic shared one loss, so critic-loss gradient flowed back
   through `img_step` into the straight-through actions. The reference calls the
   critic on `value_input[:-1].detach()`. The actor's own input was also not
   detached, adding a recurrent actor→actor path the reference cuts with
   `inp = feat.detach()`.
6. **`imag_gradient` never reached the loop, and `both` was wrong anyway.**
   `notebooks/10_train_paper_online.ipynb` computed
   `imag_gradient = str(train.get("imag_gradient", "both"))` and then never
   passed it to `outer_cycle`, so both XL runs used `"both"` despite the configs
   saying `reinforce`. Our `"both"` was `reinforce + entropy + backprop` — a
   **sum**. The reference is a convex mix,
   `mix·dynamics + (1-mix)·reinforce`, with `imag_gradient_mix: 0.0`, so its
   `both` is *exactly* pure reinforce. We were applying, at full strength, a
   biased straight-through gradient through a one-hot action that DreamerV3
   deliberately switches off for every discrete-action task (`crafter`,
   `atari100k`, `minecraft`, `memorymaze` all set
   `imag_gradient: 'reinforce'`). With `entropy: 3e-4` there is nothing to hold
   the policy open against it.

Also confirmed missing: `critic.outscale: 0.0`, which makes the initial value
exactly 0 everywhere. Ours used default PyTorch init, so the first advantages
were an arbitrary random field.

### How much of it the advantage measurement can actually show

`scripts/diag_advantage_alignment.py` recomputes both advantage definitions on
one imagined rollout. `R2_state_only` is the fraction of advantage variance
explained by a per-start-state constant — high means the advantage carries no
action signal.

| checkpoint | definition | mean | std | frac > 0 | `R2_state_only` |
| --- | --- | --- | --- | --- | --- |
| M6 1M (size-S, entropy never < 0.223) | fixed `Q−V(s_t)` | +0.049 | 0.490 | 0.463 | **0.685** |
| M6 1M | old `G−V(s_{t+1})` | +0.057 | 0.449 | 0.485 | **0.730** |
| M7 XL 100k (collapsed) | fixed | +0.005 | 0.146 | 0.546 | 0.516 |
| M7 XL 100k (collapsed) | old | +0.008 | 0.121 | 0.575 | 0.563 |

The old baseline does make the advantage more state-determined and less
action-determined, in the predicted direction, but only by ~0.05 `R²`. **This
measurement does not establish that bug 1 alone caused the collapse**, and it
cannot: by 100k the policy is already near-deterministic, so `V` is nearly flat
along a trajectory and the two baselines almost coincide. The magnitude at the
moment of collapse is unmeasured. Bug 6 is the one with a mechanism large
enough to explain the symptom on its own; bugs 1–5 are correctness fixes that
remove the confound.

On a fresh zero-init critic the two definitions are *identically* equal
(`V ≡ 0`), which is why the bug survived every unit test and every smoke run.
That is the interesting part: an off-by-one in an advantage is unobservable at
init and only becomes wrong once the critic learns something.

## Failed alternatives (do not revive)

- **Blaming `train_ratio`.** 512 is genuinely ~5.4 days on this box
  (finding 15), but dropping to 32 did not cause and did not prevent the
  collapse. Both ratios collapsed.
- **Blaming XL-from-scratch capacity** (finding 16's conclusion). Superseded:
  the same actor-critic was under both runs, and the size-S M7 variant was never
  run to compare.
- **Raising `entropy_scale` above the paper's 3e-4.** Not attempted, and it
  should stay not attempted until the fixed loop is measured — it would have
  masked bug 6 rather than removing it.
- **Reading episode length as skill.** Still not a signal: M6's finished 1M died
  at mean length 190, and the dashboard forward-fills a sparse series, so its
  cliffs are plotting artifacts (finding 13/16).

## Fix

`src/training/returns.py` gained `imagined_targets(reward, cont, value, lam)`,
which is the single place the alignment lives: it takes the state-indexed
`[N, H+1]` rollout over `s_0..s_H` and returns `[N, H]` `(returns, base,
weights)`. `imagine_ahead` now returns `H+1` state-indexed predictions with the
replay posterior at index 0, feeds the actor and critic detached features, and
`ac_step` adds the slow-critic regression term, the discount weights, per-module
grad clipping, and a true convex `imag_gradient_mix`.

Regression tests that would have caught this:
`test_imagined_targets_baseline_is_value_at_the_acting_state` (uses a *varying*
value so the off-by-one is visible),
`test_imagined_targets_weights_zero_after_predicted_death`,
`test_critic_reads_detached_features`,
`test_both_mode_default_mix_is_pure_reinforce`,
`test_critic_starts_at_zero_value`.

Post-fix smoke (`scripts/smoke_outer_loop.py`, XL, batch 16 × seq 32, bf16):
entropy 2.758 of ln(17) = 2.833, `value` and `slow_value` exactly 0.0000,
`adv = +0.1225 ± 0.3110` (spread now larger than the mean), `weight` 0.899,
peak VRAM 12.90/13.68 GiB.

### M8-S 200k done (2026-08-28)

`configs/m8_s_acfix.yaml` **finished its 200k budget.** That was the cheap
validation, not the 1M score run. 1M is `configs/m8_xl_acfix.yaml`.

Kill rule `ac_H < 0.15` at 30k: never happened. Whole-run min `ac_H` **0.737**.
Last log **0.894** (collect entropy **1.995**). Zero logs under 0.15.

Held-out 10×10k: **1.386 → 2.068**. Eval length **227**. Last bars: wake 100,
wood **90**, sapling 70, drink 30, table 20, plant 10, wood sword 10. Online
gmean **1.716**. Collect last-50k: sapling 83%, wood **70%**, plant 43%. Last
40 lives: wood **95%**, sapling 88%. Mean collect length **179**, max **442**.
Status `ep_len=57` is one short life in the last ten (214, 200, 165, 269,
**57**, 235, 190, 437, 183, 166), not the run.

Orange sawtooth is n=10 (75k 0.76, 125k 1.87, 200k 2.07). Same lottery as
M6 (finding 13). This is not M7 collapse. Next: XL 1M, new dirs, do not
resume `m8_s_acfix` into a 1M by editing `env_steps`.

## Paper spin

Feeds an "implementation identifiability" section: a table of six actor-critic
divergences, each individually silent in the loss curves, all mapping to the
same observable (policy entropy → unimix floor). The sharpest single sentence is
that an off-by-one advantage baseline is *exactly* zero-error at initialisation
under DreamerV3's own `critic.outscale: 0.0`, so it cannot be caught by any
shape, gradient, or smoke test — only by a test that supplies a non-constant
value function. Pair it with the `imag_gradient_mix` result: a config default of
`0.0` reads like a tuning knob but is actually load-bearing, and reimplementing
`both` as a sum instead of a mix silently enables a biased straight-through
gradient on discrete actions.

## Still divergent from the reference (caption these; do not retune around them)

- `batch_length` 32 vs the reference's 64 (VRAM; seq 64 filled the card).
- `train_ratio` 32 vs 512 (wall clock; finding 15). Run M8-S first. Do not
  restore 512 until entropy is alive at ratio 32 on a known-good WM.
- `kl_free` handling — we run `free_nats_dyn: 0` because flooring `kl_dyn`
  froze the prior (finding 03); the reference floors both at 1.0.

**Loop match, 2026-08-27:** DreamerV3-torch `pretrain: 100` is 100 *joint*
WM-then-AC `_train` calls on the 2500-step random prefill (`dreamer.py`
`_should_pretrain`). We had (a) WM-only in the CLI and (b) **no pretrain at
all in the notebook**, so every M7/M8 notebook run skipped it. That is now
`pretrain_dreamer` in `scripts/train_agent.py`, called from notebook 10 and
the CLI. `overlay_wm_train` also now runs in the notebook so XL's
`dyn_scale`/`rep_scale` 0.5/0.1 actually apply.

Any Crafter number from this repo must be captioned with seq 32, ratio 32,
and `free_nats_dyn: 0`.
