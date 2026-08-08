# Notebooks

Interactive work surface for this repo — not milestone checklists.

Create a notebook whenever you want to **run real functionality and see graphs
inline** (reconstructions, RSSM rollouts, loss curves, imagination drift, etc.).
Do this continuously as features land; do not wait for a milestone tag.

| Notebook | What it does |
|---|---|
| `00_device_and_env.ipynb` | MPS + one Crafter frame + open tracked result PNGs |
| `01_perception_reconstructions.ipynb` | Load AE checkpoint, reconstruct frames, metric charts + error heatmaps |
| `02_rssm_live_diagnostics.ipynb` | Collect a short rollout, run RSSM observe/imagine, plot mechanism graphs inline |
| `03_world_model_loss.ipynb` | World-model forward + per-term losses (recon/reward/continue/KL) plotted inline |
| `04_inspect_replay.ipynb` | Load `data/m3_replay.pt`, browse frames, reward histograms, sample training windows |

## Conventions

- **Notebooks are for humans.** Prefer inline `matplotlib` figures (`plt.show()`).
  Save to `results/` only when you want a durable artifact.
- **Scripts stay canonical for CI / verify.** Shared logic lives in `src/`
  (e.g. `training.rssm_diagnostics`) so both paths stay in sync.
- **One concern per notebook.** Add a new file when you start a new interactive
  workflow — don't bolt unrelated cells onto an old notebook forever.
- **No long training loops here** unless you're deliberately debugging a few
  steps. Full training stays in `scripts/`.
- **Kernel:** `Python (worldmodel)` after `pip install -e ".[dev]"` and
  `python -m ipykernel install --user --name worldmodel --display-name "Python (worldmodel)"`.
- First cell should `chdir` to the repo root if the notebook was opened from
  `notebooks/`.
