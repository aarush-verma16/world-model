# Finding 40 — Action masking + inventory head cost ~0.2 GiB, not the ~3 GiB it looks like from a stale baseline number; masked-random prefill puts real place_table presses at 0% illegal in a 3k-step sample

**Status:** smoke-tested only (`scripts/smoke_outer_loop.py`, `scripts/diag_craft_legality.py`), 2026-09-15. No env-step training has run on `configs/m18_masked_inventory.yaml` yet.
**Kind:** compute / identifiability; a labeled DreamerV3 deviation, not a 14.5 recipe
**Evidence:** `scripts/smoke_outer_loop.py --config configs/m18_masked_inventory.yaml|configs/m17_xl_paper.yaml --env-steps 32` (same script, same batch=8/seq=64/bf16, so the two numbers are a controlled diff); `scripts/diag_craft_legality.py` on a fresh 3000-step masked-random `prefill_random_steps` dump.

## Claim

Findings 38–39 showed the M17 checkpoint presses `place_table`/`make_wood_pickaxe` with real (non-floor) policy mass, but ~98.5%/81% of those presses are illegal (wood < 2 / no table nearby) — an execution problem. M18 adds three things on top of the same DreamerV3 graph, each independently gated off for m6/m16/m17: a ground-truth legality mask in real collect + masked-random prefill, an `InventoryHead` trained on real replay, and an inventory-conditioned actor/critic with a predicted-inventory legality mask in imagination. Two questions this finding answers before any 1M run: does this cost meaningful VRAM, and does the masking mechanism actually zero out illegal presses end-to-end (not just in the unit test)?

**VRAM: no.** Same smoke script, same batch/seq, only the config differs:

| | params | peak vram (alloc/reserved) |
|---|---|---|
| `dreamer_xl_paper.yaml` (M17, no inventory head) | 227.40M | 12.65 / 14.47 GiB |
| `dreamer_xl_paper_inventory.yaml` (M18) | 237.02M (+9.62M `InventoryHead`) | 12.76 / 14.70 GiB |

+4.2% params costs **+0.11 / +0.23 GiB**. That is well inside the 16 GiB budget and the ~1.2 GiB margin the workstation needs for the desktop compositor (finding 08). Do not confuse this controlled delta with finding 31's 11.4/11.8 GiB — that number came from the live notebook training loop, not this smoke script's eval+checkpoint path, so the absolute values differ but are not the comparison that matters here; the **within-script delta** is.

**Legal rate: yes, on the mechanism.** `prefill_random_steps(..., mask_illegal=True)` for 3000 env steps (16 lives) on a live `CrafterReward-v1` env, then `diag_craft_legality.py` (HUD-independent here — it reads the same replay dump either way) on the dump:

| | presses | illegal (wood < 2) | succeeded |
|---|---|---|---|
| `place_table` | 4 (0.13% of steps) | **0 (0.0%)** | 4 (100%) |
| `make_wood_pickaxe` | 0 | n/a | n/a |

Every `place_table` press in the masked sample had wood ≥ 2 and every one resolved (`wood -= 2`). Finding 39's unmasked M17 replay was 98.46% illegal on 18,820 presses with only 90 successes in 515k steps. This is not an apples-to-apples policy comparison (masked-random vs. a trained M17 actor), but it confirms the mask is wired correctly from raw pixel `info` through `crafter_rules.legal_action_mask` to the sampled action, not just in `tests/test_crafter_rules.py`'s synthetic-`info` unit tests.

`make_wood_pickaxe` never fired in this window. The mask only blocks *illegal* presses; it does nothing to make the *prerequisite* (wood ≥ 1 **and** a table within the 1-tile Moore neighborhood, simultaneously) more likely to occur under random movement. That prerequisite is still exploration, unchanged by M18. This is the expected, and intentional, limit of "masked-random only" versus a scripted movement assist — the user chose to keep movement unscripted to stay closer to vanilla Dreamer, so M18 fixes the execution failure mode (finding 39) but does not by itself fix the exploration failure mode (why 2-wood-near-a-table states are rare in the first place).

## Failed alternatives

- Estimating the VRAM cost from the smoke run's absolute numbers (12.76/14.70 GiB) against finding 31's 11.4/11.8 GiB and concluding a ~3 GiB regression. Those numbers come from different code paths (smoke script's eval+ckpt-roundtrip vs. the training-loop-only measurement in finding 31); the controlled same-script diff is +0.11/+0.23 GiB.
- Treating a masked-random sample's 0% illegal rate as a projection of what a trained M18 actor's Crafter score will be. It only validates the plumbing (env `info` → mask → sampled action); it says nothing about whether inventory-conditioning changes what the actor decides to attempt, or whether it ever reaches the mine-stone prerequisite chain.

## Paper spin

Compute appendix: ground-truth action masking plus a small auxiliary head is a ~4% parameter, ~2% VRAM cost on this workstation — not the reason to avoid it. Methods: a masked-random prefill diagnostic is the correct way to validate a legality mask end-to-end (real `info` dict, not synthetic), and it should report both the press rate and the still-unsolved prerequisite-reachability rate, not just "illegal presses are now 0%."
