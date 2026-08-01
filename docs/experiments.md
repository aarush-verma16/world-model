# Experiment Log

Living log of approaches tried, including failures and why they failed.

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
- On zsh, `conda init` (no args) may only patch `.bash_profile` — use `conda init zsh`
  and open a new shell before `conda activate worldmodel`.
- Craftax migration was tried and reverted; benchmark remains original Python Crafter.

## M1 — Encoder/Decoder

- Baseline (`6afa8f6`): plain encode→decode MSE, embed 256 — loss dropped, but sprites
  smeared into grass (MSE + tight bottleneck). TB also logged near-identical spawn frames.
- Sharpness pass: `PerceptionAutoencoder` with U-Net skips + L1, diverse vis frames,
  `embed_dim=8192`. This is for M1 visual trust; later RSSM still uses the skip-free
  `Encoder` embedding path. Config: `configs/m1_autoencoder.yaml`.
  Logs: `runs/m1_autoencoder_sharp`. Check `results/m1/recon_final.png`.
