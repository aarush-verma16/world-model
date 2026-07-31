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
