# World Model

A Dreamer-style world model for [Crafter](https://github.com/danijar/crafter): learn a compressed predictive model of the environment from pixels, then train a policy inside imagined rollouts instead of on every real interaction.

Built in PyTorch for **Windows + NVIDIA CUDA** (RTX 5080). No cloud APIs in the training or inference path — everything runs locally.

<p align="center">
  <img src="results/m1/recon_final.png" alt="Real vs reconstructed Crafter frames" width="720" />
  <br />
  <em>Real frames (left) vs autoencoder reconstructions (right).</em>
</p>

## How it works

Dreamer separates **world modeling** from **decision making**:

1. An **encoder** compresses each 64×64 RGB observation into an embedding.
2. An **RSSM** (recurrent state-space model) tracks a deterministic memory state `h` and discrete categorical latents. On real data it forms a posterior `z_posterior` from `h` and the embedding; in imagination it samples a prior `z_prior` from `h` alone.
3. A **decoder** and auxiliary heads reconstruct observations and predict reward / continuation, which train the world model.
4. An **actor-critic** acts in latent imagination — rolling the world model forward with `z_prior` only — so most learning happens without stepping the real environment.

This repo follows the DreamerV2/V3 recipe (discrete latents, straight-through gradients, unimix categorical floor, layer-normalized GRU) at the paper's 32×32 categorical size, trained with bf16 AMP at batch 16 × seq 32 on a 16 GiB GPU.

## What’s included

- Gymnasium wrapper for `CrafterReward-v1` / `CrafterNoReward-v1`
- Perception autoencoder with U-Net skips for sharp reconstructions
- Discrete categorical RSSM (`h`, `z_prior`, `z_posterior`, STE)
- Full world-model training: decoder + reward/continue heads, KL balancing, sequential replay
- Training configs, CLI scripts, and interactive notebooks with inline plots
- Local TensorBoard logging and mechanism diagnostics (latent entropy, occupancy, imagination drift)

Architecture notes and naming conventions live in [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md). Experiment history is in [`docs/experiments.md`](docs/experiments.md).

## Requirements

- Windows 10/11, NVIDIA GPU (this machine: RTX 5080, 16 GiB, Blackwell sm_120)
- Recent Game Ready / Studio driver (`nvidia-smi` must work)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Miniforge
- Python 3.11
- **CUDA 12.8+ PyTorch**. RTX 50-series will not run on cu124/cu121 or the default PyPI CPU wheel.

## Install

PowerShell from the repo root (one-shot):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
conda activate worldmodel
```

Or by hand:

```powershell
conda env create -f environment.yml
conda activate worldmodel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
python -m ipykernel install --user --name worldmodel --display-name "Python (worldmodel)"
```

If `conda activate` fails in PowerShell, run `conda init powershell`, open a new terminal, and try again.

Upstream `crafter` registers only with the legacy `gym` package. This repo re-registers the same env IDs for Gymnasium via `src/envs/crafter_env.py`.

Smoke-check the stack:

```powershell
python scripts/verify_m0.py
python scripts/smoke_cuda_step.py
python scripts/run_tests.py
```

`run_tests.py` is the ML unit suite (shapes, gradient flow, invariants). It prints each test plus a grouped pass/fail table. Skip the Crafter env smoke with `--fast`. Same suite via `pytest -v`. These tests do **not** train the model or assert Crafter score — that lives in the notebooks and TensorBoard.

`smoke_cuda_step.py` prints steps/sec and peak VRAM for the current `m3` batch/seq. If it OOMs, drop `train.batch_size` 16 → 8 in `configs/m3_world_model.yaml` and re-run the smoke.

## Usage

**Perception (autoencoder)**

```powershell
python scripts/collect_random_frames.py
python scripts/train_autoencoder.py --config configs/m1_autoencoder.yaml
# optional fine-tune:
# python scripts/train_autoencoder.py --config configs/m1_autoencoder_finetune.yaml `
#   --resume checkpoints/m1_autoencoder_stem_rgb/ckpt_best.pt
```

**World model** — run this from the notebook so you can watch it and stop it:

```powershell
python scripts/collect_replay.py --config configs/m3_world_model.yaml
jupyter notebook notebooks/05_train_world_model.ipynb
tensorboard --logdir runs
```

The CLI (`python scripts/train_world_model.py`) is the same loop without live plots.

**Actor-critic (frozen world model)** — `notebooks/07_train_actor_critic.ipynb`

**Outer loop (M5)** — collect in Crafter, train WM + actor-critic, log real eval return:

```powershell
jupyter notebook notebooks/08_train_outer_loop.ipynb
# CLI (same loop, no live plots):
# python scripts/train_agent.py --config configs/m5_outer_loop.yaml
python scripts/smoke_outer_loop.py
```

Default budget is 100k env steps (hours, stoppable). Geometric-mean Crafter score is M6.

**Crafter baseline (M6)** — continue the 100k agent to 1M with the official geo-mean:

```powershell
jupyter notebook notebooks/09_train_baseline.ipynb
# CLI:
# python scripts/train_agent.py --config configs/m6_baseline.yaml
python scripts/smoke_crafter_score.py
```

Do not quote the M5 0.10 return as this score. The 1M number is a user-run.

**Paper-style online (M7)** — fresh actor, 10k lives, optional XL ~200M:

```powershell
python scripts/count_params.py --smoke --size xl
jupyter notebook notebooks/10_train_paper_online.ipynb
# new dirs: checkpoints/m7_xl_paper  (do not resume m7_paper_online)
```

Stop the collapsed kernel first. Watch entropy, not the orange 10-eval line.

**RSSM diagnostics**

```powershell
python scripts/verify_rssm_forward.py
python scripts/visualize_rssm.py
```

**TensorBoard**

```powershell
tensorboard --logdir runs
# http://localhost:6006
```

## Layout

```
configs/       Experiment YAML (sizes in configs/sizes/)
src/envs/      Crafter / MiniGrid wrappers
src/models/    Encoder, decoder, autoencoder, RSSM
src/training/  Device helpers, rollouts, diagnostics, AMP train step
src/agents/    Actor-critic
notebooks/     Interactive exploration (inline figures)
scripts/       CLI entry points (including setup_windows.ps1)
results/       Figures and rollouts
docs/          MkDocs source
paper/         Paper draft and figures
```

## References

- Hafner et al., [*Mastering Diverse Domains through World Models*](https://arxiv.org/abs/2301.04104) (DreamerV3)
- Hafner et al., [*Mastering Atari with Discrete World Models*](https://arxiv.org/abs/2010.02193) (DreamerV2)
- Hafner, [*Crafter*](https://danijar.com/project/crafter/)

## License

MIT
