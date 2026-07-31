# Dreamer-Style World Model on Craftax-Classic

A faithful, small-scale implementation of the Dreamer/RSSM recipe trained on
[`Craftax-Classic-Pixels-v1`](https://github.com/MichaelTMatthews/Craftax) — a fast JAX
reimplementation of Crafter with the same task, achievements, and reward structure. The agent
learns a compressed predictive world model from real experience, then trains its actor-critic
almost entirely inside imagined rollouts from that model.

This is a research / learning project: reproduce a baseline under a fixed laptop compute
budget (Apple M4 Pro; PyTorch MPS for learning, JAX CPU for the env), then run a controlled
ablation. Full architecture and workflow conventions live in
[`PROJECT_BRIEF.md`](PROJECT_BRIEF.md).

## Status

**M0 complete** (`v0.0-setup-complete-craftax`): environment verified on Craftax-Classic
Pixels, TensorBoard logging works, visual smoke GIF generated. Model code starts at M1.

See [`MILESTONES.md`](MILESTONES.md) for the full gated plan.

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

# Verify all M0 checks (PyTorch MPS + JAX CPU + Craftax + TensorBoard + visual GIF)
python scripts/verify_m0.py
```

> JAX is installed as **CPU-only** (`jax[cpu]`). Do **not** install `jax-metal` or other
> Metal/GPU JAX plugins — Apple's plugin is unmaintained. PyTorch still uses MPS for model
> training; Craftax env stepping stays on CPU, which is fine under Dreamer's small real-env
> step budget.
>
> The wrapper (`src/envs/craftax_wrapper.py`) resizes Craftax-Classic's native pixel obs
> (printed on first reset; currently `(63, 63, 3)`) to `(64, 64, 3)` uint8 for the model stack.

### Day-to-day terminals

1. **Train / scripts** — `conda activate worldmodel`, run Python scripts
2. **TensorBoard** — leave running while developing:

```bash
conda activate worldmodel
tensorboard --logdir runs
# open http://localhost:6006
```

3. **Optional** — git / notes

Expected `verify_m0.py` output includes PyTorch `MPS available: True`, JAX `default_backend: cpu`,
Craftax wrapped obs shape `(64, 64, 3)`, a TensorBoard event under `runs/m0_dummy/`, and
`results/m0_random_rollout.gif`.

## Repository layout

```
configs/          # YAML configs, one per experiment/ablation
src/
  envs/           # Craftax-Classic wrapper
  models/         # encoder, RSSM, decoder, heads
  agents/         # actor-critic
  training/       # world-model / agent training, replay buffer
experiments/      # per-run configs + logged metrics
docs/             # MkDocs source
paper/            # paper draft + figures
results/          # final plots, tables, rollout GIFs
scripts/          # smoke tests and utilities
```

## Next milestones

1. Encoder + decoder on Craftax-Classic observations
2. Discrete categorical RSSM with KL balancing
3. Replay buffer + world-model training loop
4. Actor-critic trained on imagined rollouts
5. Baseline result, then ablation

## License

TBD.
