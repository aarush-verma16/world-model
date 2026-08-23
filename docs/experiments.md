# Experiment Log

Living log of approaches tried, including failures and why they failed.

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
