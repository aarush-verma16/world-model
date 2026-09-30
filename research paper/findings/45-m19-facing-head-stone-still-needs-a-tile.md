# M19 gives the actor the facing tile, because M18 at 204k still never mines

**Claim.** At 203,648 env steps M18’s last-200 stone rate is still 0 (3 stone lives in the whole run, 37 pickaxe lives, mean length 171.5). More steps on the inventory mask will not create `collect_stone`. That achievement is `do` while a wood pickaxe is in inventory and the faced tile is stone. The mask does not touch `do`, and the XL CNN’s 4×4 cell is larger than the 7px tile, so the actor never sees it. M19 is a new run that feeds facing material, a faced-object bit, and nearby materials in as symbols.

## Why this is not “the paper”

No M19 score exists yet. This is the mechanism and the 204k measurement that justifies leaving M18’s graph alone. Do not caption 2.34 next to 14.5.

## Evidence

Collect `results/m18_masked_inventory`, 1,245 lives, last env step 203,648 (2026-09-29 evening). Last 200: geometric mean 2.34, wood 50.5%, table 17%, drink 18.5%, wood pickaxe 0.5%, stone 0, length 177.5. Last 50: pickaxe 2%, table 12%, stone 0, length 192.5. Since the 188k look (finding 44): one new pickaxe life, zero new stone lives. Pickaxe lives still last 171.5 steps. Train flush 199,936: `ac_H` 0.137, `recon_l1` 0.0068. Held-out at 200k is 1.28 on n=10 with wood 10 and no table, pickaxe, or stone.

`collect_stone` is not a `place_*` or `make_*` action. `legal_action_mask` leaves `do` legal because the same button chops trees and attacks (finding 39). Imagination’s inventory mask only checks item counts (finding 40), so a predicted pickaxe still does not say the faced tile is stone. Finding 04: one latent cell is 16×16 px, larger than a tree, a zombie, or the faced tile.

## What M19 changes

New dirs only (`configs/m19_facing.yaml`, `notebooks/12_train_m19_facing.ipynb`). It keeps M18’s legality mask and inventory head, and adds:

- Replay stores `facing`, `facing_object`, `nearby`, `has_spatial`. Old dumps load with `has_spatial=0`.
- `SpatialHead` on `[h, z]`: material cross-entropy, object logit, nearby multi-label BCE. Real collect conditions the actor on ground truth. Imagination uses the head’s argmax / threshold.
- Imagination’s `place_*` / `make_*` mask also requires the predicted facing tile and neighborhood. `do` stays legal so “pickaxe and facing stone” can be learned as a policy, not hardcoded.

m6/m17/m18 configs do not set `spatial:`, so their checkpoints still have no `spatial_head` keys. Do not load them into M19.

## What this does not fix

It does not make stone appear in the starting grass, and it does not stop a zombie at step ~170. If pickaxe lives still die at ~170 after they can see the tile, the next piece is combat (`make_wood_sword` / `defeat_zombie`), not another craft mask and not a stone reward bonus.

## Failed alternatives

Grinding M17 to 500k with paper knobs (finding 37). Grinding M18’s inventory mask past the point where pickaxe left the last-200 window (findings 43–44, and this 204k look). A hunger or stone bonus (findings 01, 05, 19).

## Paper spin

Methods: a pixel RSSM on Crafter does not represent the 7px predicate `do` needs, so the intervention is a supervised facing/nearby head with an imagination mask, not a longer DreamerV3 run. Results: M18’s 204k island is the before-number. The after-number is M19’s last-200 stone rate, which is not measured yet.
