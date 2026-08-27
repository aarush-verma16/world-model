# Notebooks

Interactive work surface for this repo — not milestone checklists.

Create a notebook whenever you want to **run real functionality and see graphs
inline** (reconstructions, RSSM rollouts, loss curves, imagination drift, etc.).
Do this continuously as features land; do not wait for a milestone tag.

| Notebook | What it does |
|---|---|
| `00_device_and_env.ipynb` | CUDA + one Crafter frame + open tracked result PNGs |
| `01_perception_reconstructions.ipynb` | Load AE checkpoint, reconstruct frames, metric charts + error heatmaps |
| `02_rssm_live_diagnostics.ipynb` | Collect a short rollout, run RSSM observe/imagine, plot mechanism graphs inline |
| `03_world_model_loss.ipynb` | World-model forward + per-term losses (recon/reward/continue/KL) plotted inline |
| `04_inspect_replay.ipynb` | Load `data/m3_replay.pt`, browse frames, reward histograms, sample training windows |
| `05_train_world_model.ipynb` | Full world-model training with inline loss curves + recon grids (no CLI) |
| `06_decoder_probe.ipynb` | Probe-decodability test: is blurry recon a starved decoder or a blind latent? |
| `07_train_actor_critic.ipynb` | Frozen-WM actor-critic: 15-step `z_prior` imagination, loss/entropy curves, GIF |
| `08_train_outer_loop.ipynb` | M5 outer loop: policy collect, WM + AC updates, real-env eval return (user-run) |
| `09_train_baseline.ipynb` | M6 baseline: continue 100k → 1M, official Crafter gmean + 10×10k eval (user-run) |
| `10_train_paper_online.ipynb` | M7 paper-style 1M: fresh actor, 10k cap, default XL ~200M from scratch (user-run) |

## Conventions

- **Notebooks are for humans.** Prefer inline `matplotlib` figures (`plt.show()`).
  Save to `results/` only when you want a durable artifact.
- **Scripts stay canonical for CI / verify.** Shared logic lives in `src/`
  (e.g. `training.rssm_diagnostics`) so both paths stay in sync.
- **One concern per notebook.** Add a new file when you start a new interactive
  workflow — don't bolt unrelated cells onto an old notebook forever.
- **Long training runs** (world model, actor-critic, outer loop, baseline, paper-online) live in
  notebooks `05` / `07` / `08` / `09` / `10` so you can watch and stop them. CLIs in
  `scripts/` are the same loops without live plots.
- **Kernel:** `Python (worldmodel)` after `pip install -e ".[dev]"` and
  `python -m ipykernel install --user --name worldmodel --display-name "Python (worldmodel)"`.
- First cell should `chdir` to the repo root if the notebook was opened from
  `notebooks/`.
