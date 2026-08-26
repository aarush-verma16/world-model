# Finding 12 — A 400-step collect cap is a second Crafter time-limit

**Status:** measured on M5 100k (2026-08-26); fixed in the M6 harness before the 1M run  
**Kind:** evaluation / identifiability; not a DreamerV3 result  
**Evidence:** M5 `eval_metrics.json` lengths 174–257 vs cap 400; `crafter.Env(length=10000)` and `info['discount']`

## Claim

Crafter’s episode length is **10000**. M5 collected and evaluated at **400**. That cap was **not binding** on the 100k policy (they die at ~180–220), so it did not change the 0.10 staircase. It **would** become a silent second time-limit the first time a policy learns to eat and drink. A 400-cap 1M run is not the Crafter benchmark. Gymnasium `done` was also mapped entirely to `terminated`, so a real 10k timeout would have stored `continue=0` like death.

This is not in DreamerV3. The paper’s Crafter number uses the env’s 10k length and the official geometric-mean of training-episode unlock rates.

## Evidence the 400 cap was idle, not a recipe

| source | mean length | cap |
|---|---|---|
| M5 eval, 15 ticks | 171–257 | 400 |
| Crafter / gym registration | 10000 | 10000 |

Death, not truncation, ended every logged M5 eval. Unlock count stayed ~1. Raising the cap on a dying policy does not raise return; leaving it at 400 on a surviving policy would.

## `done` vs `discount`

`crafter.Env.step` sets `done = dead or (step >= length)` and `info['discount'] = 1 - float(dead)`. [`CrafterEnv`](../../src/envs/crafter_env.py) previously returned `terminated=done`, `truncated=False`. Collector stores `cont = 0` only on `terminated`. A timeout would have looked like death in the continue head.

M6: `terminated = discount < 0.5`, `truncated = done and not dead`. Collect/eval cap **10000**. Replay FIFO stays **500k** steps (~6 GB host); do not bump to 1e6 on this box (finding 08).

## Failed alternative we are not running

Keep the 400 cap “because M5 lengths were under 400.” That is true until the policy lives. Do not train M6 at 400 and then evaluate at 10k.

## Paper spin

Limitations / eval protocol: any Crafter score from this repo must state episode length 10000 and that M5’s 400 cap was a collect convenience, not the benchmark. Continue flags follow Crafter `discount`, not gym `done`.
