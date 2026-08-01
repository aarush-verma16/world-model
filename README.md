# Dreamer-Style World Model on Crafter

A faithful, small-scale implementation of the Dreamer/RSSM recipe trained on
[`CrafterReward-v1`](https://github.com/danijar/crafter). The agent learns a compressed
predictive world model from real experience, then trains its actor-critic almost entirely
inside imagined rollouts from that model.

This is a research / learning project: reproduce the baseline under a fixed laptop compute
budget (Apple M4 Pro, MPS), then run a controlled ablation. Full architecture and workflow
conventions live in [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md).

## Status

**M0 complete** (`v0.0-setup-complete`): environment verified, TensorBoard logging works,
Crafter visual smoke GIF generated. Model code starts at M1.

See [`MILESTONES.md`](MILESTONES.md) for the full gated plan and
[`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) for architecture conventions.

## Setup

Requires [Miniforge](https://github.com/conda-forge/miniforge) (conda) on macOS Apple Silicon.

If `conda activate` says *"Run 'conda init' before 'conda activate"* in Terminal/Cursor,
your shell is almost certainly **zsh** and conda was only initialized for bash. Fix once:

```bash
conda init zsh
# then close this terminal tab and open a new one
conda activate worldmodel
```

Full env setup:

```bash
# Create / update the env (Python 3.11)
conda env create -f environment.yml
# or, if the env already exists:
# conda env update -f environment.yml --prune

conda activate worldmodel
pip install -e .
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Verify all M0 checks (MPS + Crafter + TensorBoard log + visual GIF)
python scripts/verify_m0.py
```

> Note: upstream `crafter` only auto-registers with the legacy `gym` package.
> This repo wraps it for Gymnasium in `src/envs/crafter_env.py` and exposes the
> same IDs (`CrafterReward-v1`, `CrafterNoReward-v1`).

### Day-to-day terminals

1. **Train / scripts** — `conda activate worldmodel`, run Python scripts
2. **TensorBoard** — leave running while developing:

```bash
conda activate worldmodel
tensorboard --logdir runs
# open http://localhost:6006
```

3. **Optional** — git / notes

Expected `verify_m0.py` output includes `MPS available: True`, Crafter obs shape
`(64, 64, 3)`, a TensorBoard event under `runs/m0_dummy/`, and
`results/m0_random_rollout.gif`.

## Repository layout

```
configs/          # YAML configs, one per experiment/ablation
src/
  envs/           # Crafter / MiniGrid wrappers
  models/         # encoder, RSSM, decoder, heads
  agents/         # actor-critic
  training/       # world-model / agent training, replay buffer
experiments/      # per-run configs + logged metrics
docs/             # MkDocs source
paper/            # paper draft + figures
results/          # final plots, tables, rollout GIFs
scripts/          # smoke tests and utilities
```

## Milestone 1 — Encoder / Decoder

```bash
conda activate worldmodel
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/collect_random_frames.py
python scripts/train_autoencoder.py
# Terminal 2:
tensorboard --logdir runs
# open http://localhost:6006  -> m1/recon_l1 + m1/real_vs_recon
# In Images, scrub to the latest step (not step 1 — that one is grey mush).
open results/m1/recon_final.png
pytest -q
```

M1 trains a U-Net-style `PerceptionAutoencoder` so left/right frames stay visually
close. The skip-free `Encoder` / `Decoder` modules remain for later RSSM wiring.

## Next milestones

1. Discrete categorical RSSM with KL balancing
2. Replay buffer + world-model training loop
3. Actor-critic trained on imagined rollouts
4. Baseline result, then ablation

## License

TBD.
