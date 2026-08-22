# Experiment Log

Living log of approaches tried, including failures and why they failed.

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
  2×9 bar at rows 49–63. Dedicated slot-grid HUD head off the bottom 4×4
  row + `recon_hud`. Not skip-to-RGB, not 8px world decoder.
- Step ~1000 of that run looked flat/"corrupted" (only the avatar blob
  and HUD icons had any detail, rest of the frame plain grass): set
  `recon_avatar_scale` / `recon_hud_scale` to 5.0, same as full-frame
  `recon_scale`, but they're means over 441px / 882px crops vs 4096px —
  ~9x the per-pixel gradient weight of a generic pixel. Dropped both to
  1.0 (area-fair is ~0.5 / ~1.1).

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
