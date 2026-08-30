# Finding 22 — Continuing the sleep-island actor does not buy survival

**Status:** measured on M10-XL `m10_xl_r128` at 400k → ~523k env steps, 2026-08-30  
**Kind:** negative result on “just turn up `train_ratio` on the stuck policy”; not a reason to roll back the RSSM  
**Evidence:** `results/m10_xl_r128/eval_metrics.json`, `collect_episodes.jsonl` (866 lives, env 400192–523520); notebook 10 printed `RESUMING checkpoints/m10_xl_r128/ckpt_latest.pt at env_steps=500000`

## Claim

M10 loaded M9’s 400k weights (`seed_joint_ckpt` + `RESUME="auto"`) and ran **train_ratio 128** for another ~120k env steps. Mean length stayed **~183**. Last-200 `wake_up` is **96.5%**. **0%** of last-200 lives reach the starvation clock (338). Held-out gmean went **2.25 → 1.72 → 1.76** (n=10). The dashboard starting at 400k, then at 500k after a kernel restart, is that loader — not a hung counter and not “training from scratch.”

Zombie kills in the collect windows did rise **13% → 19% → 28%**. Length did not. Sleeping in the open still ends the life around 180 (finding 19). More actor-critic updates on this policy is not the same experiment as more updates from env_steps=0.

Do not resume `checkpoints/m10_xl_r128` or `checkpoints/m9_xl`. Next run is M11: same XL graph, ratio 128, **new dirs, `RESUME=None`, no `seed_joint_ckpt`**.

## What the 523k jsonl actually is

866 lives after the 400k seed. Last-200: mean/median length **183 / 178**, max **308**, **70%** dead before 200, **0% ≥ 338**. Stone **0**. Wood pickaxe **3%**. Plant **79%**. Wood **50%**.

| env window | n | mean len | wake % | zombie % | stone % |
|---|---|---|---|---|---|
| 400k | 277 | 180 | 95 | 13.4 | 0 |
| 450k | 270 | 185 | 97 | 19.3 | 0.4 |
| 500k | 319 | 186 | 97 | 27.6 | 0 |

Held-out ticks (n=10): **2.25 @ 400k**, **1.72 @ 450k**, **1.76 @ 500k**. All three have `wake_up` 90–100% and stone 0. The orange drop is the same lottery as finding 13 / 21, not a new collapse (`ac_H` stayed off the 0.08 floor).

A later kernel restart with `RESUME="auto"` loaded `ckpt_latest.pt` at **500000**. That is why the x-axis can look like “we started at 500k.”

## Failed alternatives

- Treating 400k/1M on the status line as a bug in the step counter. It was `load_checkpoint` of M9, then of M10’s own 500k.
- Grinding M10 to 1M because zombie % ticked up. Length and `wake_up` did not move.
- Rolling back the RSSM / extra recon heads / hunger bonuses. Findings 01, 05, 19, 21.
- `RESUME="auto"` on notebook 10 after the user said the run must never resume.

## What we changed

`configs/m11_xl_r128.yaml`: from scratch, `train_ratio` 128, no `seed_joint_ckpt` / `seed_replay`. Notebook 10: `RESUME=None`, writes `checkpoints/m11_xl_r128`, raises if a checkpoint path would be loaded. Keep the M10 dir as this negative result; do not seed from it.

## Paper spin

Results / recipe: raising `train_ratio` mid-run on a policy that already sleeps every life is not the paper’s 512-from-scratch curve. Survival (mean length, `wake_up` falling) is the Crafter score; extra replay on the island did not purchase it.
