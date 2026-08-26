# Finding 08 — Workstation constraints that actually changed the recipe

**Status:** established on this machine (RTX 5080 16 GiB, 32 GiB RAM, Windows); **host-RAM crash 2026-08-25**  
**Kind:** compute / systems results; DreamerV3 assumes cluster-scale defaults  
**Evidence:** `scripts/smoke_cuda_step.py`; seq-64 timing; replay sizes; Jupyter VRAM holds; step-628150 traceback

## Claim

Several “hyperparameters” in this repo are **not** scientific choices. They are what a 16 GiB Blackwell card plus a 32 GiB host can run without lying (thrashing, CPU wheels, leftover kernels). A paper that copies DreamerV3’s seq 64 / huge replay / always-on collect without this appendix will not reproduce our numbers — and should not pretend to.

## seq 64 fits in VRAM and is still unusable

DreamerV3 default sequence length is 64. On the **old XL** graph, seq 64:

| | seq 32 | seq 64 |
|---|---|---|
| peak VRAM | 8.8 GiB | **15.8 / 16.3 GiB** |
| time / step | 0.29 s | **33 s** |

It did not OOM. It **thrashed** (paging / allocator churn with the desktop compositor already holding ~1.5 GiB). We kept seq 32. The 32×32 categorical latent was **not** shrunk — VRAM knobs are batch/seq, not `stoch`.

![seq 64 VRAM](../figures/seq64_vram.png)

After the reset (~19M params, one decoder) a step is ~2.4–2.8 GiB / ~5.6–8.5 it/s at batch 16 × seq 32 (700k run sat at **2.6 / 2.8 GiB**). Headroom exists; we still have not promoted seq 64 without a new smoke. **Do not silently bump it in an ablation.**

## CUDA 12.8 is a correctness constraint, not a speed tip

RTX 50-series (sm_120) on a cu124/cu121 or default PyPI wheel: `cuda.is_available()` may be false, or “no kernel image.” Windows default `pip install torch` is **CPU-only**. Every timing number in this archive assumes `torch` from `https://download.pytorch.org/whl/cu128` and bf16 AMP (no GradScaler).

## Replay size on a frozen random policy is a data-coverage knob

| dump | what happened |
|---|---|
| ~80 random episodes | unique frames reused hundreds of times over 12k steps; mean grass |
| **600** random episodes, max 400 steps | current M3 reset; still sparse on cows/zombies/saplings (finding 05) |

M3 is **supervised on a frozen buffer**, not DreamerV3’s online loop. Comparing our 700k-step reconstructions to published DreamerV3 Crafter figures without stating that is misleading. The same dump was trained to 700k; extra steps after ~500k did not change coverage (finding 09).

## Dual XL decoders vs one size-S decoder

Residual decoder **twice** (embed path + `[h,z]` path) used **19 GiB** — over the card. One residual decoder at depth 32 is fine. Several “we can’t use DreamerV3’s CNN blocks” conclusions were **graph duplication**, not model size.

| graph | params | peak VRAM (batch 16×32, bf16) |
|---|---|---|
| pre-reset XL dual decoder | ~120–146M | ~9–13 GiB |
| reset size-S one decoder | **18.7M** | **~2.4 GiB** |

## Jupyter kernels hold the card after the loop stops

A finished or interrupted notebook kernel kept **~11 GiB** with GPU util ~1%. TensorBoard event files named `*.ghost.<pid>` matched that PID. This is not a PyTorch leak in the train step; it is the interactive workflow. Training and a 16 GiB game do not coexist (max-settings Spider-Man was the practical test: both want most of the 16 GiB; PyTorch’s caching allocator will not shrink).

## Host RAM, not VRAM, killed the 700k run at step 628150

Status line at the crash: `vram 2.6/2.8 GiB`, `5.64 steps/s`. Then matplotlib `FigureCanvas.print_figure` raised `MemoryError: Unable to allocate 9.00 MiB for an array with shape (192, 2048, 3) dtype=float64`. The 9 MiB array is the open-loop video-pred strip (`3×32×64` tiles) promoted to float64. IPython still prints the status line; the **next** CUDA step dies with `AcceleratorError` in `torch.multinomial` — a poisoned context, not a full card.

Cause: every 50 steps `show_progress` `imshow`’d the full strip and IPython retained prior figure outputs. After ~12k logs on a 32 GiB host that was already holding the 1.26 GB replay, the 9 MiB allocation was the one that failed. Weights last written at **620k** (`checkpoint_every: 10000`).

Fix that is now in the M3 notebook (and required for M4 dashboards): thin the loss history (~800 points), downsample the displayed strip (full PNG still saved), wrap the figure in try/except so a dashboard failure **does not abort training**, `plt.close("all")` + `gc.collect()`, compact JSON. Kernel restart is mandatory after a host OOM.

This is not a DreamerV3 result. It is why a “live matplotlib dashboard of 2048-px strips” is a workstation hazard.

## M5: the same dashboard class ate outer-loop throughput

The 100k env-step outer loop (`configs/m5_outer_loop.yaml`, notebook `08`) started at **~18 env/s** (peak 27) and decayed to **~1.1–1.4 env/s** by 50–58k. Wall clock: 10k at 01:32, 20k +41 min, 50k at 06:38, interrupt at **58624** env steps ~08:33. Halfway in 6–7 hours was the host, not CUDA.

`log_every` was **16** — every collect cycle. Each tick called `show_progress` (matplotlib `display`) **and** `gc.collect()` while IPython retained figure outputs and the online replay grew from the 1.26 GB seed toward **~1.98 GB**. Same class as the 628150 host OOM: the dashboard walks the replay-sized process every step. Saving the 1.98 GB dump every 10k is a pause, not the 18→1.1 slope.

| env_steps | mean env/s (`train_metrics.json`) |
|---|---|
| 16–2k | **18.4** (peak 27) |
| ~10k | 5.7 |
| ~20k | 3.1 |
| ~30k | 2.2 |
| ~40k | 1.7 |
| 49–50k | 1.4 |
| last logs ~58.6k | **1.12** |

Fix: status line every 256 env steps, matplotlib every 1000, `gc.collect()` only on checkpoint. After resume, the same run recovered **~38 env/s** (peak 44; VRAM 2.54 GiB). 60k→100k wall clock was **~18 min** (ckpt 60k 09:45, ckpt 100k 10:03). The dashboard **texture** change at 60k is that cadence: 3664 logs at stride 16, then 161 logs at stride 256. Entropy/L1 did not jump to a new optimizer; they got undersampled. `env_steps % eval_every == 0` with `collect_every=16` never hits 5000 (or 5000-image dumps) — evals in the first half are 0/10k/20k/… only. `crossed_interval` compares integer buckets so 4992→5008 fires (hence 65008, 75008, …). Last train-metric row is **99840** (`256 × 390`); the loop still hit **100000** (eval + `ckpt_step_100000.pt`). Resume from `ckpt_latest.pt` was `env_steps=58624`.

The noisy recon-L1 uptick on the dashboard is **not** the slowdown (online coverage vs a frozen buffer). 4-episode eval return bouncing −0.65 / 0.1 / −0.4 is M5 noise, not a Crafter score (finding 11).

![M5 notebook dashboard at ~50k](../figures/m5_notebook_dashboard_50k.png)

## `is_first` is not threaded yet — and that is consistent with our sampler

DreamerV3 replay is a stream; a sampled window can **cross an episode boundary**, so `is_first` resets GRU state mid-sequence. `ReplayBuffer.sample` here only returns windows with `start + seq_len <= episode_len`. There is no boundary to reset across.

M5 keeps that sampler on purpose. Online collect still stores **full episodes** and FIFO-drops the oldest when `max_steps` is hit. `is_first` is **not** an M5 prerequisite — adding a stream just to look more like the paper would be a new identifiability surface (mid-window GRU reset vs our current truncated-BPTT-from-zero, which Dreamer also does on windows that do not contain a boundary). Until we sample across concatenations, claiming “we match DreamerV3 replay” is still false.

## Paper spin

Hardware appendix: 16 GiB, seq 32, bf16, 600-episode frozen buffer, ~19M size-S, 700k steps, ~5.6–8 it/s, 32 GiB host that can OOM on a dashboard before the GPU does — and the same dashboard cadence turned an 18 env/s outer loop into 1.1 env/s by 58k. Limitations: single seed, single env, episode-bounded replay (`is_first` unused), seq 32. That is an honest small-scale study, which is the project’s stated research goal — not a competing DreamerV3 number.
