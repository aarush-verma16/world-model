# Finding 38 — Cheap gate: M17 500k still has table mass; AC graph matches NM512; do not start another 5-day run

**Status:** measured on frozen `checkpoints/m17_xl_paper/ckpt_step_500000.pt` + `data/m17_xl_paper_replay.pt`, 2026-09-13  
**Kind:** identifiability / kill criteria; not a 14.5 recipe  
**Evidence:** `scripts/diag_policy_mass.py` (seed 0, batch 8, 32 windows, 2048 posterior states); `scripts/diag_advantage_alignment.py --config configs/m17_xl_paper.yaml --joint-ckpt …/ckpt_step_500000.pt --batch 8`; NM512 `dreamerv3-torch` `models.py` / `dreamer.py` / `configs.yaml` `crafter:` cloned 2026-09-13

## Claim

A 5-day outer loop cannot be the test for the next change. On the **same** 500k replay sample, the actor is **not** at the unimix floor on crafting: `P(place_table)=0.068` (**116×** floor `0.000588`), `P(do)=0.117`, `P(make_wood_pickaxe)=0.0035` (**5.9×** floor). Replay step mix is table **3.5%**, pickaxe **1.4%**, `do` **13.8%**. Last-200 *episode* table is **2.5%** and stone **0** (finding 37). They **spam** craft/mine actions and still do not complete the tree. That is failed execution (inventory / combat / timing), not missing logits.

Advantage on this joint ckpt (fixed definition; script prefills 2k random steps, then imagines — not the M17 replay): mean **+0.016**, std **0.145**, `R2_state_only=0.616`. Old off-by-one `G-V(s_{t+1})` R2 **0.424**. M6 1M fixed was **0.685** (catalog). This is not finding 17’s dead advantage. `feat.detach()` before the actor, critic on detached features, `imag_gradient=reinforce`, `V(s_t)` baseline, slow critic, and discount weights **match** NM512 Crafter.

Remaining mismatches vs NM512-torch are **not** a silent AC-graph bug: batch **8 vs 16** (same 512 transitions/env), **bf16 vs fp32** (jax Crafter is bf16), imag reward/value decoded as two-hot **mean** vs their `.mode()`, CNN as our residual `blocks=2` stack vs their `cnn_depth=96` ConvEncoder. `expl_behavior` is **greedy** in the reference too. None of those is a 30-minute proof of 14.5.

**Do not start another 500k/1M.** There is no Phase-1 graph fix whose replay-sample craft mass we can even re-measure against a before/after, because the before is already above the unimix kill.

## Frozen baseline (reproducible)

```
python scripts/diag_policy_mass.py
python scripts/diag_advantage_alignment.py --config configs/m17_xl_paper.yaml --joint-ckpt checkpoints/m17_xl_paper/ckpt_step_500000.pt --batch 8 --steps 2000
```

| action | P_actor | × unimix floor | replay steps % |
|---|---|---|---|
| move_* (sum) | 0.534 | — | 50.0 |
| do | 0.117 | 199 | 13.8 |
| place_table | **0.068** | **116** | 3.5 |
| sleep | 0.059 | 100 | 4.9 |
| place_plant | 0.046 | 79 | 4.1 |
| make_wood_pickaxe | **0.0035** | **5.9** | 1.4 |

Mean policy entropy on posteriors **0.162**. Collect log `ac_entropy` **0.079** is the greedy env average (finding 37), not a different network.

## Kill criteria (before any multi-day run)

1. **Unimix kill (this ckpt already passes).** If `max(P(place_table), P(make_wood_pickaxe)) < 2 × unimix/17` on seed-0 / 32 windows, do not train — imagination cannot invent crafting. **M17 500k does not trigger this kill** (table 116×). Do not use “no craft mass” as the story.
2. **Claimed AC-graph fix.** Re-run the **same** two commands. Abort the 5-day clock if `P(make_wood_pickaxe)` does **not** move by **≥2×** (0.0035 → ≥0.007) **and** advantage `R2_state_only` does **not** drop by **≥0.10** vs 0.616. Need at least one of those. A yaml entropy/warmup tweak that leaves both numbers inside noise is finding 26/36 again.
3. **25k env probe (only if 2 passed).** Abort if last-50 `place_table` and `make_wood_pickaxe` *episode* rates stay ~0 and `ac_H` stays in 0.15–0.20. Do not read last-200 gmean (finding 37).
4. **Never** a 500k because recon is 0.004, because teal is flat, or because NM512 uses fp32 / batch 16.

## NM512 Crafter vs this repo (only leftovers)

Diffed `src/training/ac_step.py`, `imagine.py`, `collect.py` against NM512 `dreamerv3-torch` `models.py` `ImagBehavior` and `dreamer.py` `_policy` (clone 2026-09-13).

**Match (do not “fix” again):** `imag_gradient=reinforce`, actor samples from `feat.detach()`, critic on detached features, slow critic 0.02 EMA, λ-returns with `V(s_t)` baseline, cumprod discount weights, unimix 0.01, entropy 3e-4, STE sample in collect, `expl_behavior=greedy` (no Plan2Explore), seq 64, GRU 4096, `dyn_hidden`/`units` 1024, replay 1e6, `train_ratio` 512 replayed transitions/env, `is_first` in `observe`, actor/critic 5-layer 1024, `free_nats=1` on both KL terms.

**Still different, not a 5-day justification:**

| leftover | NM512-torch Crafter | this repo (M17) |
|---|---|---|
| minibatch | 16 × 64, ~8 opt steps / 16 env | 8 × 64, 16 opt steps / 16 env (same 8192 tokens; batch 16 pages 16 GiB, finding 31) |
| dtype | `precision: 32` | bf16 AMP (jax Crafter is also bf16, finding 31) |
| imag reward/value decode | dist `.mode()` | `symlog_twohot_mean` |
| entropy graph | `actor(imag_feat).entropy()` on live features | entropy from `actor.policy(feat.detach())` |
| eval actions | `actor.mode()` when not training | `evaluate_policy` samples STE like collect |
| CNN | `cnn_depth: 96` ConvEncoder | residual `[96,192,384,768]` `blocks=2` (closer to jax Table B.1; do not copy their shallower CNN) |

Joint Adam with two param groups vs their separate actor/value opts is not a graph bug. None of these is a 30-minute proof of 14.5.

## Failed alternatives

- Starting M18 at ratio 512 because “something else is wrong with training.” Phase 1 says the trainer and the AC graph are live.
- Adding a stone bonus so table mass becomes table *success*. Findings 01, 05, 19.
- Treating `R2_state_only=0.616` as proof to reopen finding 17. M6 1M was 0.685 on the *fixed* definition.

## Paper spin

Methods / evaluation: a frozen-checkpoint action-mass table is the cheap falsifier for “the actor never crafts.” Crafter geometric mean can sit at ~1.6 while `P(place_table)` is 7% of steps, because *episodes* still die before a pickaxe. Compute: do not spend 5 days to rediscover that.
