# Experiment Log

Living log of approaches tried, including failures and why they failed.

## M7 paper-style online (2026-08-27)

**v1 collapsed** at ~100k (`runs/m7_paper_online`, finding 14): unimix-floor
entropy, wake_up only. Do not resume that checkpoint.

**v2 cancelled at ~40k / 1M (2026-08-27):** paper `train_ratio` 512 was **~2.14 env/s**
(~5.4 days for 1M, finding 15). Entropy ~0.56; first wood/drink after 25k; mean
length still ~170 (same hunger death as M6). Orange held-out was still step 0.

**v3 (wired):** `configs/m7_xl_workstation.yaml` — same XL seed recipe,
`train_ratio` **32** (1+1 / 16 env). New dirs. Do not resume `m7_xl_paper`.
Live: `notebooks/10_train_paper_online.ipynb` — **restart the kernel**. Watch
`ac_H`; 0.08 is the floor, stop. Length vs 190 is not a 40k gate.

## M6 Crafter baseline (2026-08-26, 1M done)

Continue the M5 **100k** agent to **1M** env steps. Config:
`configs/m6_baseline.yaml`. Live run: `notebooks/09_train_baseline.ipynb`
(user-run; do not launch 1M env steps from the agent). CLI:
`python scripts/train_agent.py --config configs/m6_baseline.yaml`.

- Seed: `checkpoints/m5_outer_loop/ckpt_latest.pt` +
  `data/m5_outer_loop_replay.pt` (`seed_joint_ckpt`). Later resumes use
  `checkpoints/m6_baseline/ckpt_latest.pt`.
- Same 16/1/1, batch 16 × seq 32, 32×32 latent. Collect/eval cap **10000**
  (finding 12). Replay FIFO still **500000**.
- Finished **1M** 2026-08-26. Notebook protocol **PASS** (finite metrics,
  entropy 0.533, jsonl n=4746, 10 held-out evals, joint ckpt).
- **Cited numbers:** held-out gmean **0.627 → 1.639**; online (collect jsonl
  from resume) **1.941**. Caption: size-S, seq 32, FIFO 500k, continued from
  M5 100k. **Not DreamerV3 14.5.**
- Collect mean length **190**, max **441**. Zero episodes hit 1000. The 10k
  cap never binds — they still die (finding 13). Last eval: sapling 100%,
  wood 60%, plant 60%, wake 40%; stone/pickaxe 0.
- Held-out n=10 is a lottery (900k 0.43, 200k 2.22). Online buckets stay
  1.4–2.1. Do not re-run because the orange line dipped.
- Weights every 50k; GIF `results/m6_baseline/eval_step_1000000.gif`.
- Tag `v1.0-baseline-result` when you want the milestone artifact.

## M5 outer loop (2026-08-26)

Online Dreamer cycle on top of the **700k** world model and **20k** actor-critic.
Config: `configs/m5_outer_loop.yaml`. Live run: `notebooks/08_train_outer_loop.ipynb`
(user-run; do not launch 100k env steps from the agent). CLI:
`scripts/train_agent.py`.

- Seed: `checkpoints/m3_dreamer_s/ckpt_step_700000.pt`,
  `checkpoints/m4_actor_critic/ckpt_final.pt`, `data/m3_dreamer_s_replay.pt`.
- Ratio: **16** env steps → **1** WM step → **1** AC step (batch 16 × seq 32).
- Collect and eval cap **400** steps (same as the frozen dump). 10k-step
  protocol + geometric-mean Crafter score are **M6**.
- Collect/eval actions are **STE samples**, not greedy `logits.argmax`.
- Replay stays **episode-bounded** (no `is_first` / stream sampler). FIFO
  `replay_max_steps: 500000`. Online dump is `data/m5_outer_loop_replay.pt`
  (does not overwrite the M3 seed).
- Skill plot is **real eval return**. Imagined λ-return stays an AC diagnostic
  (finding 10).
- Finished **100k** env steps 2026-08-26. Notebook plumbing **PASS** (finite
  losses, entropy 0.943, joint ckpt reloads). Last train-metric row is
  **99840** (`log_every` 256); `ckpt_step_100000.pt` and eval @ 100000 are
  the real end. Dashboard 18→1.1 env/s by 58k, then **~38 env/s** after
  `log_every: 256` / `dashboard_every: 1000` (finding 08). 60k→100k ~18 min.
- Last eval return **0.10** with mean achievements **1.0** and length
  **~180–220**. That is `achievements − 0.9` (one unlock, then die), not a
  Crafter score (finding 11). 75k’s 0.35 is 4-episode noise. Do not grind
  this 100k run for a prettier return plot. Geo-mean is M6.

Smoke: `python scripts/smoke_outer_loop.py` (dozens of env steps). Tag
`v0.5-full-loop-integrated` when you want the plumbing artifact; the eval
curve is not a Crafter baseline.

## M4 actor-critic on frozen 700k world model (2026-08-25)

M3 size-S (`configs/m3_dreamer_s.yaml`) finished **700k** gradient steps on the
frozen 600-episode random-policy replay. Gates passed (`r=0.98`, open-loop std
ratio 0.98, last-10k `kl_rep_raw≈1.96`). Extra WM steps after ~500k were a
plateau — see `research paper/findings/09-frozen-replay-plateau.md`. Do **not**
push this buffer to 1M.

M4 trains actor + critic only, on 15-step `z_prior` imagination from that
checkpoint. Code: `src/agents/actor_critic.py`, `src/training/imagine.py`,
`src/training/returns.py`, `src/training/ac_step.py`. Config:
`configs/m4_actor_critic.yaml`. Live run: `notebooks/07_train_actor_critic.ipynb`
(user-run; do not launch 20k steps from the agent). Dashboard is thinned so a
matplotlib strip cannot host-OOM training again (finding 08, step 628150).

Finished **20k** steps on 2026-08-26. Notebook exit **PASS** (reward finite
0.023, entropy 0.814 > 0.1, critic 5.59 → 1.47, decode std 0.21). Last-2k
window: imagined reward **0.017**/step vs λ-return **2.61** — critic bootstrap,
not Crafter skill (`research paper/findings/10-imagined-return-not-skill.md`).
GIF: `results/m4_actor_critic/imagine_final.gif`. Next is M5 (online collect +
real env eval). Tag `v0.4-imagination-actor-critic-working` when you want the
milestone artifact; do not grind this frozen-WM loop further.

## DreamerV3 M3 reset (2026-08-23)

Two weeks of tuning on `configs/m3_world_model.yaml` (the sub-pixel decoder,
`HzToMap`, per-cell `z`, `free_nats_dyn: 0.0`, blob/avatar/HUD crop losses,
etc. — every entry below this one) never reached M3's exit bar. Stepping back:
none of those fixes were wrong given what they were diagnosing, but the graph
they were diagnosing had drifted from DreamerV3 far enough that the optimizer
was solving a different problem than "train a DreamerV3 world model". Rather
than add another loss term, this reset the training graph to match
`NM512/dreamerv3-torch`'s actual `WorldModel._train` step, and dropped
everything that graph doesn't have.

### What was actually wrong (the graph, not the weights)

1. **A second decoder trained straight off the encoder embedding
   (`recon_embed`), bypassing the RSSM entirely.** This is not in DreamerV3 —
   the only decoder input in the paper is `feat = concat(h, z_posterior)`.
   With a free bypass, the encoder+decoder pair can converge as a plain
   autoencoder (embed recon looked fine) while `[h,z]` carries almost nothing,
   because nothing forces the *information the RSSM sees* to be what gets
   reconstructed. Every earlier "loss weight" tune was really fighting this:
   `[h,z]` misplacing terrain while embed looked fine (the exact split this
   bypass predicts) got diagnosed as "`HzToMap` needs more capacity" or
   "posterior needs to be spatial" instead of "the decoder has an escape hatch
   that doesn't require the RSSM to work at all."
2. **`[h,z]` decoded through a `detach_weights=True` copy of the decoder.**
   So even without the bypass, the one path that *did* go through the RSSM
   could never learn to render — only `feat`, not the renderer, could move.
   Combined with (1), the renderer trained exclusively on the bypass path.
3. **Unweighted-mean pixel L1 vs. a KL summed over categoricals.** DreamerV3's
   image loss is `-logprob` under an (effectively) MSE decoder, i.e. squared
   error **summed over `C*H*W`** then averaged over batch/time — a few hundred
   nats at typical error rates. A per-pixel *mean* (`recon_l1` here) is smaller
   by a factor of `C*H*W` (~12k), so at `recon_scale=5` the actual reconstruction
   pressure was on the order of 0.03-0.15, next to a KL sitting at 1-2 nats —
   backwards from the paper, where reconstruction dominates and KL is
   deliberately kept small by free bits. This is the root cause the "KL welded
   above `free_nats`" entries below were really fighting.
4. Smaller deviations in the same direction: `HzToMap` (no DreamerV3
   equivalent — the paper's decoder reads `feat` through a plain dense layer,
   not a hand-built spatial map from `h`), per-region crop losses
   (`recon_blob`/`recon_avatar`/`recon_hud`/`edge_weight` — none exist in the
   paper; DreamerV3 gets Crafter mobs/HUD from the *encoder's* residual blocks
   and enough training, not per-region loss engineering), a scalar MSE reward
   head (Crafter reward is ~always 0, so MSE learns to always predict 0 —
   DreamerV3 uses symlog two-hot classification specifically to avoid this),
   and `grad_clip=100` vs the paper's `1000`.

### What changed

- `src/models/world_model.py`: removed `HzToMap`, the auxiliary embed
  decoder, and `recon_bottleneck`/`recon_embed`. One decoder, decoding live
  from `feat = concat(h, z_posterior)`. Added `WorldModel.video_predict` (open
  -loop rollout: posterior for a context window, then pure `z_prior`
  imagination for the rest, decoded — DreamerV3's `video_pred` diagnostic).
- `src/models/decoder.py`: dropped `detach_weights` entirely (no path needs
  it now); added optional residual blocks (`blocks>0`, affordable with only
  one decoder) and `output_activation="linear"` (`+0.5`, DreamerV3
  `cnn_sigmoid=False`, pixels in `[0, 1]`) alongside the old `"tanh"`
  (`[-1, 1]`, kept as the default so M1's autoencoder is unaffected).
- `src/models/symlog.py` (new) + `RewardHead` (`src/models/heads.py`):
  symlog two-hot discrete reward regression (255 bins over `symlog([-20,
  20])`), matching DreamerV3's reward/critic head.
- `src/training/losses.py`: dropped every per-region loss
  (`content_weight_map`/`blob_recon_loss`/`tile_blob_loss`/
  `avatar_recon_loss`/`hud_recon_loss`/`weighted_pixel_loss`/
  `gradient_l1_loss`) and the embed/bottleneck recon terms. `world_model_loss`
  is now exactly the paper's four terms: `image_mse_loss` (sum over pixels,
  mean over batch/time), symlog two-hot reward NLL, continue BCE, and
  `kl_balance` (unchanged mechanism — see the KL entries below, still valid).
  `recon_l1` / `reward_mae` kept as unweighted, log-only human-readable
  metrics (never multiplied into `total`).
- `src/training/wm_step.py`: grad-clip norm 100 -> 1000 (paper default; a
  ceiling for genuine blowups, not a routine clamp — it only mattered because
  gradients used to be artificially small).
- `configs/m3_dreamer_s.yaml` (new): DreamerV3 "size S" recipe — encoder/
  decoder depth 32 (channels 32/64/128/256, `blocks=1`), `deter=512`,
  `stoch=32 x classes=32`, `hidden=512`, `prior_layers=1`, `lr=1e-4`,
  `dyn_scale=0.5`/`rep_scale=0.1`/`free_nats=1.0` on **both** KL terms
  (`free_nats_dyn: null` reuses `free_nats`, the paper default — opening
  `free_nats_dyn` is deliberately not the starting point this time, since the
  frozen-decoder bug that motivated it is gone). `configs/m3_world_model.yaml`
  is left on disk for history but is no longer the default anywhere.
- `scripts/collect_replay.py` default config now points at
  `configs/m3_dreamer_s.yaml`, and the new config collects 600 random
  episodes (was 80) — more unique frames per gradient step.
- `scripts/train_world_model.py`, `scripts/smoke_cuda_step.py`,
  `tests/test_world_model_m3.py`, `notebooks/05_train_world_model.ipynb`:
  rewritten for the new API/metrics (`recon`/`recon_l1`/`reward`/
  `reward_mae`/`continue`/`kl`/`kl_dyn_raw`/`kl_rep_raw` — no more
  `recon_embed*`/`recon_blob`/`recon_avatar`/`recon_hud`/`grad`), plus an
  open-loop video-prediction panel/gate alongside the posterior-recon panel.

### Do-not-revive list (each failed for a documented reason above or below)

- A second decoder / any path that can reconstruct without going through
  `z_posterior` (the actual root cause of two weeks of "embed looks fine,
  `[h,z]` doesn't").
- `detach_weights` / a frozen decoder on any path that's supposed to learn to
  render.
- Per-region crop or blob losses (`recon_blob`/`recon_avatar`/`recon_hud`/
  `edge_weight`/`content_weight_map`/`gradient_l1_loss`) — not in DreamerV3;
  they were treating a scale bug (item 3 above) as a missing-detail problem.
- Per-pixel-*mean* reconstruction loss — always use the paper's
  sum-over-pixels reduction (`image_mse_loss`), or KL will dominate again.
- Chasing pixel-perfect / sharp reconstructions as the M3 bar. DreamerV3's own
  Crafter reconstructions are recognizable and blurry; `milestones.md`'s exit
  criterion is reward correlation and healthy losses, not sprite sharpness.

### Honest limitations of this reset

- `is_first` is not threaded into `RSSM.observe`. DreamerV3's real replay is a
  continuous stream where a sampled window can cross an episode boundary
  mid-sequence; `ReplayBuffer.sample` here only ever samples a window fully
  inside one episode (`start + seq_len <= episode_len`). M5 kept that sampler
  (FIFO episode list, not a stream); `is_first` is not an outer-loop prerequisite.
- Still supervised world-model training on a frozen random-policy replay
  buffer, not the online collect/train loop — that's M5, per
  `milestones.md`. M4 is actor-critic on the frozen M3 checkpoint only.
- `prior_layers=1` and the exact `dyn_scale`/`rep_scale` split follow
  `NM512/dreamerv3-torch`'s `configs.yaml` rather than the paper's Table 4
  (which lists `dyn_scale=1.0`); the two published sources disagree slightly
  and this picked the runnable reference.

## Sub-pixel decoder + spatial `h` (2026-08-23)

The 12k run finished with `recon_l1` 0.037 / `recon_embed_l1` 0.027 and still
had: blurry terrain, no trees/zombies/saplings, an unreadable inventory that
did not react when a count changed off its starting 9, an imprecise player, and
`[h,z]` placing terrain in the wrong spots while embed recon looked fine.

Two structural causes, both upstream of every loss knob tried before:

1. **`Upsample(mode="nearest")` in the decoder discarded sub-cell position.**
   Nearest replicates each latent cell into a 2x2 block of *identical* values,
   so the following 3x3 conv has no way to tell where inside the cell it is.
   At the 4x4 latent a cell is 16x16 px — larger than any Crafter sprite and
   larger than the whole 14px inventory strip height. All the sub-cell phase a
   7px tree/zombie/sapling and a ~4px inventory glyph are made of was thrown
   away before a single conv could use it. This is also the true cause of the
   flat 16x16 blocks previously blamed on "live `[h,z]` decoder weights".
   Replaced with sub-pixel convolution (`Conv2d(in, out*4)` + `PixelShuffle(2)`)
   and ICNR init so the four sub-filters start identical (begins as nearest, so
   it cannot checkerboard). Both convs of the final stage now run at 32x32,
   which made the decoder *cheaper*, not more expensive.
2. **`HzToMap` gave `h` no spatial slot.** It was
   `h_proj(h).unsqueeze(-1).unsqueeze(-1)`: a `[B, C, 1, 1]` per-channel bias,
   identical in all 16 cells. So every bit of `[h,z]` layout came from
   `z_proj`, and `z` is 32 categoricals x 32 classes = **160 bits/frame** —
   nowhere near enough to place 63 tiles plus a 9-slot inventory, hence the
   hallucinated placement. `h` now projects to a full 4x4 map, and
   `deter_dim` 512 -> 2048: Crafter layout is mostly persistent (grass, water,
   trees do not move) and `h` is the only state carried across time.

Verified at the representation level before spending a real run, via
`scripts/smoke_decoder_capacity.py` (overfit 8 real frames, 4000 steps,
identical seed/data, sub-pixel vs a nearest control):

| region          | sub-pixel | nearest (old) |
| --------------- | --------- | ------------- |
| full frame L1   | 0.0059    | 0.0170        |
| inventory strip | 0.0076    | 0.0487        |
| player crop     | 0.0074    | 0.0145        |

`results/m3/decoder_capacity_4k.png` shows it directly: the nearest control
paints the inventory bar as smeared dashes (exactly the reported symptom) and
loses sprites, while sub-pixel renders legible digits and visible saplings.
Nearest was a **6.4x** worse HUD at the capacity ceiling, so no loss weight
could ever have fixed it.

### The free-nats floor was on the wrong term (same day)

`kl_raw` sat at ~1.006 for 10k steps. Diagnosis: **capacity was never the
binding constraint.** `z` is 32 cats x 32 classes = 160 bits of *capacity*, but
`kl_rep_raw` is the *rate*, and the floor pinned it at 1 nat = 1.44 bits per
frame. Spread over 32 categoricals that is 0.031 nats each — the posterior was
telling the decoder essentially nothing the prior had not already predicted.

The mechanism: `kl_dyn` detaches the **posterior**, so it can only train the
**prior** and cannot restrict information at all — yet it got the same
`clamp_min(free_nats)`. So once the prior was within a nat, its gradient went to
exactly zero and the dynamics model stopped improving. Since content the prior
predicts costs *zero* rate, freezing the prior permanently charged the
posterior's 1-nat budget for anything that moved, and it was dropped instead.
That closed loop is why the plateau never broke no matter which loss weight was
tuned.

Fix: `free_nats_dyn: 0.0` (floor on the rep term only). Verified by a unit test
that optimizes a prior toward a fixed posterior with plain SGD:

| dyn floor  | final `kl_raw` |
| ---------- | -------------- |
| 1.0 (old)  | 0.998 (stalls) |
| 0.0 (new)  | 0.072          |

14x better prior from an identical budget. `free_nats` stays **1.0** on rep, so
the rate constraint is unchanged; `kl_raw` should now fall *below* 1.0, and a
lower KL with improving recon means more detail is free rather than less.

Supporting changes, both aimed at prior accuracy (the thing that decides how
much is free):

- `rssm.hidden` 512 -> **1024**. It was a narrow waist between a 2048-dim `h`
  and a 1024-dim `z_flat`: prior (2048 -> 512 -> 1024), GRU input, and the
  posterior conv all squeezed through it.
- `rssm.prior_layers: 2` (was a single hidden layer, now configurable).
- 146M params, 9.2 GiB peak, 0.32 s/step.

Explicitly **not** done: raising `stoch`/`classes`. That buys capacity, which was
not the limit, and more categoricals raise the summed KL against the same floor.

### 12k run: KL stayed *above* free_nats after 4.5k (same day)

The dyn-floor fix did what it was supposed to early: steps 2k–4.5k sat under
the line (mean 0.92, 76% of logs < 1). Then the decoder became good enough to
spend leftover nats, and KL climbed and stayed there:

| window     | KL mean | share < 1 | recon_l1 |
| ---------- | ------- | --------- | -------- |
| 2k–4.5k    | 0.92    | 76%       | 0.064    |
| 4.5k–12k   | 1.35    | 3% (4/151)| 0.043    |

Peak 2.19 around 10k, finish 1.13. Those extra 0.35 nats did **not** buy
sprites: `recon_step_12000.png` still has no trees/zombies/saplings, HUD digits
are streaks, `[h,z]` still misplaces terrain. So this was not "using the free
budget on mobs" — it was recon L1 spending bits on grass texture because the
rate term was too weak to push back.

Cause: `recon_scale=5` (5× DreamerV3's rec=1) with `dyn_scale=0.5`,
`rep_scale=0.1` left the paper's dyn/rec and rep/rec ratios 10× and 5× weaker.
Fix: `dyn_scale` 0.5 → **1.0**, `rep_scale` 0.1 → **0.5**. `free_nats` stays
1.0, `free_nats_dyn` stays 0.0. Not raising `free_nats` to paper over it.

Also this pass:

- `recon_avatar_scale` / `recon_hud_scale` 0.0 -> **0.5**. Not the 5.0 that
  flattened recon before (that was 5.0 *plus* a pasted HUD head). Full-frame L1
  gives a pixel 5.0/4096; 0.5 on a 441px avatar crop / 882px HUD crop is
  ~1.9x total weight. Test guards the range at `<= 1.0`.
- `seq_len` 64 re-measured now that the decoder is cheaper: 15.8 GiB and
  **33 s/step** vs 8.8 GiB / 0.29 s/step at 32. It does not OOM, it thrashes.
  Stay at 32; drop `batch_size` if anything ever OOMs.
- 120M params (was 83M) and peak VRAM went *down*, 10.0 -> 8.8 GiB.
- "Loss through the roof" on the 8k->12k resume was a plotting artifact:
  `history` reset on resume so the x-axis started at 8000, and the y-axis
  autoscaled a 0.006 wiggle to fill the panel. y is anchored at 0 now and a
  resume reloads `train_metrics.json`.
- Training appearing stuck at 8500 was `clear_output(wait=True)` blocking the
  kernel on the notebook UI, not CUDA (GPU 1%, VRAM held). `wait=False` and one
  dashboard draw per log tick.

## Windows / CUDA migration (2026-08-21)

Moved the project off the M4 Pro (24 GB unified memory, MPS) onto a Windows
desktop: RTX 5080 (16 GiB dedicated VRAM, Blackwell sm_120), 32 GiB system RAM.

- Device helper is CUDA-first (`src/training/device.py`). bf16 AMP + TF32 +
  cuDNN benchmark are the training defaults.
- `configs/m3_world_model.yaml` scales from the Mac swap workaround
  (`batch_size=4`, `seq_len=32`, fp32) to batch 16 × seq 32 + bf16 (4× the
  frames per step). DreamerV3's seq 64 filled 15.8 / 16.3 GiB on a live smoke
  (desktop compositor already holds ~1.5 GiB) and is left as a later bump.
- Install path: `scripts/setup_windows.ps1` (CUDA 12.8 wheels). Default PyPI
  torch is CPU-only on Windows.
- Halfway through the unweighted-L1 XL run (step ~6k): skip panel was near-
  perfect while `[h,z]` / skip-free embed stayed smeared. That was wiring,
  not cell size: M1's U-Net `stem_to_rgb` was on the world-model graph and
  could copy the frame without putting content in the RSSM embedding.
  World model now uses skip-free `Encoder`. 8x8 spatial experiment reverted.
- Skip-free 4-stride CNN still dropped cows/HUD/trees at step ~4700 (embed
  and `[h,z]` both blob-less except a smeared player). DreamerV3's CNN is
  `ImageEncoderResnet`: stride-2 then `cnn_blocks=2` residual 3x3 pairs at
  each scale. The skinny stack aliased 8-12px sprites into grass. Switched
  encoder to that ResNet (still 4x4 flatten, no skip-to-RGB). Decoder
  residuals were tried and reverted: two XL ResNet decoders used 19 GiB.
- Embed recon showed land/water while `[h,z]` stayed mean grass / one HUD
  blob: posterior was `Linear(12288+512 → 512)`, mixing the 4x4 map. Switched
  to `SpatialPosterior` (conv on the 4x4 embed, then categorical logits).
- `[h,z]` still showed no environment (not even blur) while embed did: two
  separate pixel decoders + a mixing Linear into `[h,z]` pixels. Shared 4x4
  upsample + `HzToMap` + `recon_map` L1 (copy encoder map, detached).
- `[h,z]` environment in the wrong place at step ~8k: mixing Linear(z →
  4x4) plus `[h,z]` pixel loss through the *shared* decoder scrambled
  layout. Per-cell `z` (2 categoricals / cell) + decoder weights detached
  on the `[h,z]` paint. Dashboard vis uses mid-sequence, not t=0.
- Environment + player showed, other sprites never even as blurs: 64x64
  L1 median-erases 7px cows. `recon_blob` is now Crafter-tile (7px) L1 on
  the local view with local-deviation weights (objects on grass).
- Avatar pose/tool stuck: player is always the same camera tile. Added
  `recon_avatar` on the 3×3 tiles around that slot.
- Inventory numbers stuck: amount glyphs are ~4px in a 7px slot, in a
  2×9 bar at rows 49–63. A dedicated HUD head that *pasted* over those
  rows hid whatever the 64×64 decoder learned and left a black bar.
  Removed. Inventory is painted by the same upsample as the world.
  `recon_hud` / `recon_avatar` extra scales are 0.
- Frozen decoder weights on the `[h,z]` paint starved the renderer
  (solid green, no avatar blob). Live weights again; per-cell `z` still
  holds layout.
- KL: `free_nats` is 1. Per-cell `z` (2 cats × 16 cells) parked KL raw at
  1.2–1.6; posterior logits are `Linear` after the 4×4 conv again.
- Solid-color terrain + one left HUD icon at step ~2k: `recon_blob_scale=5`
  is 7px tile-mean L1 (paints each tile one color) and `[h,z]` was training
  the shared upsample (4×4 map → 16×16 solid cells). Blob off; decoder
  weights detached on the `[h,z]` paint again. Resuming a blob-trained
  step-1000 decoder did not grow texture — start from scratch.

## Setup / M0 (2026-07-31)

## Setup / M0 (2026-07-31)

- Created conda env `worldmodel` (Python 3.11) via Miniforge.
- Confirmed `torch.backends.mps.is_available() == True`.
- Confirmed `CrafterReward-v1` resets/steps with observation shape `(64, 64, 3)`.
- Wrote dummy TensorBoard scalars to `runs/m0_dummy/` (`m0/dummy_loss`).
- Saved random-policy visual GIF to `results/m0_random_rollout.gif`.
- Tagged `v0.0-setup-complete`.

### Notes

- Upstream `crafter` only registers with legacy `gym`; we wrap it for Gymnasium in
  `src/envs/crafter_env.py`.
- (Historical, Mac) On zsh, `conda init` (no args) may only patch `.bash_profile`
  — use `conda init zsh` and open a new shell before `conda activate worldmodel`.
  On this Windows box, use `conda init powershell` if `conda activate` fails.
- Craftax migration was tried and reverted; benchmark remains original Python Crafter.

## M1 — Encoder/Decoder

- Baseline (`6afa8f6`): plain encode→decode MSE, embed 256 — loss dropped, but sprites
  smeared into grass (MSE + tight bottleneck). TB also logged near-identical spawn frames.
- Sharpness pass: `PerceptionAutoencoder` with U-Net skips + L1, diverse vis frames,
  `embed_dim=8192`. This is for M1 visual trust; later RSSM still uses the skip-free
  `Encoder` embedding path. Config: `configs/m1_autoencoder.yaml`.
  Logs: `runs/m1_autoencoder_sharp`. Check `results/m1/recon_final.png`.
