# Every "unaided" iron pickaxe came from a demonstrator world the actor inherited

**Claim.** Every 25k env steps, `scripts/train_agent.py` runs `seed_teacher_episodes(collect_env, ...)`. That plays 8 demonstrator lives on the *collect* env object while the actor's life is still open. When the injection returns, the actor's open life continues inside the last demonstrator world, with its inventory and achievement counters. The six "unaided" iron pickaxes in M22, M23, and M24 (two each, findings 54, 56, 57, 59, 60) and all three in M20 are exactly the lives that span an injection. With those lives removed, the actor's unaided iron pickaxe count during training is **0** in all four runs.

## Why this is not the paper

DreamerV3 does not inject demonstrations. This is an accounting bug in our collect loop that produced a fake positive in the actor's training-time count. The held-out evals use a separate `eval_env` and are unaffected; they were already 0/10 everywhere.

## Evidence

- Pattern. Every unassisted iron-pickaxe life ends within 136 env steps after an injection (25,024; 25,136; 50,016; 75,024 ×3; M20: 25,024, 50,016, 100,064). Every one starts before the injection (`env_steps - length < k·25,000`).
- Lengths. A 48-step life (M23, 75,024) and a 19-step life (M20, 50,016) are credited with coal, a furnace, iron, and the iron pickaxe. Finding 60 noted that the only iron sword and diamond across all three runs sit on two of these lives, while the handoff teacher never made either; the v4 injection teacher does.
- Reproduction (`paper/scripts/check_inject_contamination.py`, env only, no model). A random life with zero achievements, then an 8-life v4 injection on the same `CrafterEnv`, then one no-op: the open life now reports `make_iron_pickaxe`, `make_stone_pickaxe`, `place_furnace`, and 16 more, and the inventory holds an iron pickaxe.
- Why the handoff missed them. `craft_assist_active` fires on stone pickaxe and no iron pickaxe. The inherited world already has the iron pickaxe, so the takeover never starts and the life is logged unassisted.
- Spanning lives per run: 3 in M22, M23, M24 (one per injection), 7 in M20. One M24 spanning life (25,024) was handed off and is dropped from the handoff count (23/114 → 23/113).

Corrected counts (`paper/scripts/build_assets.py`, `spans_injection`): M20 0/937 actor iron pickaxe; M22 0/357, M23 0/319, M24 0/347 actor lives. Teacher finishes 23/60, 44/121, 23/113.

Also confirmed while checking: M22's handoff started at env 20,704 (resume at 20,448). Before that, 17 actor lives made a stone pickaxe without a takeover, and none made the iron pickaxe.

## Failed alternatives

Reading two unaided lives per run as "the actor can craft occasionally." Reading their timing as irregular. Treating the starting checkpoint as having crafted 3 times.

## Fix (not applied to finished runs)

Injection must use its own env, or the collector must reset its open life after an injection. Until then, drop lives that span an injection from every count.

## Paper spin

Benchmark section: the actor never made the rare craft on its own, in training or held out. The 17 uncensored M22 stone-pickaxe lives before the handoff started are a small uncensored sample with 0 crafts. Appendix accounting: one paragraph on the leak and the exclusion rule.
