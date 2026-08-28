# Finding 20 — The 14.5 gap is a different experiment, not a missing loss head

**Status:** comparison of live M8-XL (`m8_xl_acfix2`, ~80k / 1M) against Hafner et al. 2023 Crafter and `NM512/dreamerv3-torch` `crafter:` — 2026-08-28
**Kind:** protocol / recipe mismatch; remaining *code* gaps are two, not “the implementation is still DreamerV2”
**Evidence:** published 14.5 ± 1.6; M6 1M held-out 1.639; M8-S 200k 2.068; M8-XL 50k held-out 1.36; `configs/m8_xl_acfix.yaml`; findings 13, 15, 17, 19; DreamerV3 Figure 6 (train ratio / model size)

## Claim

Putting our Crafter geometric mean next to DreamerV3’s **14.5** is valid as a *distance*, and the distance is large. It is **not** evidence that the live actor-critic graph is still wrong, or that we need another reconstruction head. The published 14.5 is XL **at 1M env steps** with **512 replayed transitions per env step**, sequence **64**, streaming replay, and GRU **4096**. We are XL-shaped **at 80k**, ratio **32**, sequence **32**, finished-episode replay, GRU **2560**, and we still modal-die to zombies around step 170 (finding 19). Those are already-measured disagreements, not folklore.

Do not kill `m8_xl_acfix2` to “start the real implementation.” Finish this 1M as the honest **ratio-32 / seq-32 / 16 GiB** number. Then change one remaining gap at a time in a **new** checkpoint dir.

## Same metric, not the same experiment

| run | env steps | held-out gmean | what it is |
|---|---|---|---|
| Crafter random (paper table) | — | **~1.6** | floor |
| M8-XL `m8_xl_acfix2` | 50k eval / ~80k collect | **1.36** / online ~1.46 | early, `ac_H` alive |
| M8-S AC-fix | 200k | **2.068** | entropy held (min 0.74) |
| M6 size-S | 1M | **1.639** (online 1.941) | finished protocol; stone ~0 |
| DreamerV3 XL published | **1M** | **14.5 ± 1.6** | official Crafter gmean |

1.36 and 1.64 are near *random*, not near 14.5. Captioning the 80k dashboard as “we missed 14.5” is the same error as finding 13’s “do not put 1.64 next to 14.5.”

At the **same** 80k env horizon the optimizer has also seen a different amount of data:

| | this box, `train_ratio` 32, seq 32 | paper-torch Crafter, 512, seq 64 |
|---|---|---|
| when we train | 1 WM + 1 AC per **16** env steps | 1 WM + 1 AC per **2** env steps |
| WM+AC steps at 80k | **~5,000** | **~40,000** |
| WM+AC steps at 1M | **~62,500** | **~500,000** |
| replayed transitions / env step | **32** | **512** |

That is **16×** less replay per env step (the paper’s own `train_ratio` definition) and **8×** fewer optimizer steps (because our sequences are also half as long). DreamerV3 Figure 6 is the paper saying this knob moves Crafter score.

## What already matches (do not “fix” these again)

Diffed against `NM512/dreamerv3-torch` `crafter:` and finding 17. Live yaml: `configs/m8_xl_acfix.yaml`.

- Discrete 32×32 categorical, unimix 0.01
- Imagination horizon 15, `imag_gradient: reinforce`, γ 0.997, λ 0.95, entropy 3e-4
- Slow critic 0.02, critic `outscale` 0, symlog two-hot reward/critic
- Batch 16, prefill 2500, pretrain 100, action repeat 1, 1 env, 1M env-step *budget*
- Eval 10 episodes × 10k (Crafter native cap; our collect cap is also 10000)
- Actor-critic graph: `V(s_t)` baseline, aligned critic target, discount weights, slow critic, detached features, `imag_gradient` actually forwarded

The M7 entropy collapse was these six bugs, not “XL is too big.” M8-S held `ac_H` ≥ 0.74 for 200k. Do not reopen that graph because the *score* is still ~2.

## What still differs, ranked

### A. Recipe — cannot claim 14.5 without changing the experiment

1. **`train_ratio` 32 vs 512.** Largest remaining lever. Finding 15: 512 on this box is ~2.1 env/s ≈ **5.4 days** for 1M. The live run is explicitly captioned as ratio 32 in the yaml header. Raising it mid-run in `checkpoints/m8_xl_acfix2` would mix two recipes in one curve.
2. **`seq_len` 32 vs 64.** Seq 64 thrashed 16 GiB on the old XL graph (finding 08: 15.8 GiB, 33 s/step). Same `train_ratio` with shorter sequences also **doubles** optimizer steps vs torch’s `Every(16×64/512)`.
3. **Replay 5e5 FIFO vs torch 1e6 (jax current 5e6).** Host RAM, not VRAM.
4. **Capacity vs Table B.1 XL.** Paper: `cnn_depth` 96, **`dyn_deter` 4096**, `dyn_hidden` 1024, residual **blocks=2**. Ours: channels `[96,192,384,768]`, **`deter_dim` 2560** (4096 on *this* LN-GRU + 2-layer prior is ~336M; 2560 ≈ 198M), `hidden` 2560, **`encoder/decoder blocks: 1`**. We are “XL budget on our graph,” not Table B.1 XL.
5. **Horizon.** Paper 14.5 is at **1M**. M8-XL is at **80k**. Even a perfect clone is not 14.5 yet.
6. **Precision.** Paper-torch default fp32; we bf16 AMP. (Current `danijar/dreamerv3` jax also trains Crafter in bf16 — do not treat fp32 as the 14.5 secret.)

### B. Remaining *code* — two real gaps, not a missing pixel loss

1. **Replay is still DreamerV2-style.** DreamerV3 (paper text): V2 only replayed **completed** episodes; V3 uniformly samples subsequences **regardless of episode boundaries** and resets the GRU with `is_first`. Ours (`ReplayBuffer.sample`): windows live entirely inside one finished episode; `RSSM.observe` never sees `is_first`. At modal length ~170 this is a ~one-life delay before a trajectory is trainable. It becomes load-bearing the moment lives get long — which is exactly when we would start to look like 14.5. Finding 08 parked this as “not an M5 prerequisite.” It is a remaining 14.5 prerequisite.
2. **CNN/decoder residual `blocks=1` vs reference default 2.** The encoder docstring already says DreamerV3’s extra residual blocks are what keep 8–12 px Crafter sprites from aliasing into grass at the first stride-2. XL yaml set `blocks: 1` to fit 16 GiB, not because 1 is the paper.

Adam vs jax’s later LaProp/AGC optimizer is a *later* jax main revision (`lr` 4e-5, `deter` 8192, `classes` 64). The **2023 14.5** recipe is the NM512 torch / Table B.1 stack, not 2026 jax `size200m`. Do not “match current github main” in one jump.

### C. Behavior on this env — even a correct agent has to not die to zombies

Finding 19: starvation death is step **338**; 66% of M8-XL lives end at 150–220; `defeat_zombie` is **0.6%**; `wake_up` is ~90% and a sleeping zombie hits for 7. Paper 14.5 lives long enough to mine and craft. Length will not look like that until zombie kills are common. That is policy/skill + the recipe above, not a 200-step gym cap (already 10000) and not a hunger bonus.

### D. Eval protocol footnote

Paper Crafter score is typically the **training-episode** achievement geometric mean over the 1M budget. We also plot **10 held-out STE episodes** (high variance; finding 13’s 900k 0.43 vs 1M 1.64). Online gmean is the closer analogue; n=10 held-out is a lottery, not the paper table.

## Failed alternatives (the “much to change” trap)

- Treat 1.36 vs 14.5 as proof the RSSM/decoder is still the pre-reset bypass graph. That graph was removed. Recon L1 on size-S went to ~0.005 (finding 09).
- Add drink/hunger/crop/sprite losses to “close the score gap.” Findings 01, 05, 19.
- Silently set `train_ratio: 512` in the live yaml and keep `m8_xl_acfix2`. New recipe, new dir; 5-day clock (finding 15).
- Jump `deter_dim` to 4096, `blocks` to 2, seq to 64, *and* ratio to 512 in one config. Any OOM or collapse then has four causes.
- Compare against current jax `dreamerv3` defaults (replay 5e6, deter 8192, 64 classes, different encoder). That is not the 14.5 paper.
- Stop the live 1M because “the implementation must change.” Then we have no ratio-32 XL-from-scratch number with a working actor-critic.

## What to change, in order, after this 1M (or in a parallel dir)

1. **Keep `m8_xl_acfix2` running** unless `ac_H` dies. It is the first XL-from-scratch run whose actor-critic is finding-17-correct.
2. **Streaming replay + `is_first` in `RSSM.observe`.** The one remaining graph disagreement that is *in the paper’s method section*, not a workstation compromise.
3. If chasing 14.5 on this box: a **new** dir with a middle `train_ratio` (128 or 256) *or* a dedicated 5-day 512 run — captioned, not mixed. Re-smoke seq 64 / `blocks=2` on the *current* reinforce graph before combining with ratio.
4. Do not invent new losses while `defeat_zombie` is still ~0.

## Paper spin

Compute + protocol section, not a new architecture. The figure is “same 1M env-step x-axis, 16× less replay, 8× fewer updates, episode-bounded sampler.” Negative result: a working DreamerV3 *graph* on 16 GiB at ratio 32 is still a random-adjacent Crafter agent at 80k, and size-S at 1M was 1.64. The 14.5 number is not a missing head; it is a different amount of training plus a policy that survives combat.
