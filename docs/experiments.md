# Experiment Log

Living log of approaches tried, including failures and why they failed.

## Setup / M0 (2026-07-31)

- Created conda env `worldmodel` (Python 3.11) via Miniforge.
- Confirmed `torch.backends.mps.is_available() == True`.
- Migrated benchmark env from Python `crafter` / Gymnasium to **Craftax-Classic Pixels**
  (`Craftax-Classic-Pixels-v1`) with JAX **CPU-only** (no jax-metal).
- Native Craftax-Classic pixel obs shape confirmed as `(63, 63, 3)` float32 in `[0, 1]`;
  wrapper resizes to `(64, 64, 3)` uint8.
- Wrote dummy TensorBoard scalars to `runs/m0_dummy/` (`m0/dummy_loss`).
- Saved random-policy visual GIF to `results/m0_random_rollout.gif`.
- Tagged `v0.0-setup-complete-craftax`.

### Notes

- On zsh, `conda init` (no args) may only patch `.bash_profile` — use `conda init zsh`
  and open a new shell before `conda activate worldmodel`.
- Craftax comparisons to published DreamerV3 / original Crafter numbers should note that
  Craftax-Classic is a faithful but non-identical codebase.
