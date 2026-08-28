"""Measure how degenerate the old off-by-one advantage was (finding 17).

The pre-fix actor loss used `V(s_{t+1})` as the baseline for the action taken
at `s_t`. Because `returns[t] = r(s_{t+1}) + γ·c·[(1-λ)V(s_{t+1}) + λ·G_{t+1}]`,
that baseline cancels the bootstrap and leaves roughly
`reward + (γ·c - 1)·V(s_{t+1})` — a quantity that hardly depends on which
action was sampled. This script prints both advantages on the same imagined
rollout, plus how much of each is explained by a per-start-state constant
(`R²` of a state-only predictor). A high `R²` means the advantage carries no
action signal, so REINFORCE just pushes whatever action it drew.

    conda activate worldmodel
    python scripts/diag_advantage_alignment.py --config configs/m8_s_acfix.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from training.device import autocast_context, configure_runtime, get_device, parse_amp
from training.imagine import freeze_world_model, imagine_ahead
from training.replay_buffer import ReplayBuffer, prefill_random_steps
from training.returns import imagined_targets, lambda_returns

from train_agent import load_seed_world_model, make_actor_critic, make_envs
from train_world_model import build_model


def _r2_state_only(adv: torch.Tensor) -> float:
    """Fraction of variance explained by a per-start-state constant.

    `adv` is `[N, H]`. Averaging over the horizon gives the state-only
    predictor; `R² → 1` means the advantage is a property of the start state,
    not of the action that was sampled.
    """
    adv = adv.float()
    total = adv.var(unbiased=False)
    resid = (adv - adv.mean(dim=1, keepdim=True)).var(unbiased=False)
    if float(total) == 0.0:
        return float("nan")
    return float(1.0 - resid / total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m8_s_acfix.yaml"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--joint-ckpt",
        type=Path,
        default=None,
        help="joint payload to take world_model + actor + critic from. Without "
        "it the critic is freshly zero-initialised, which makes both advantage "
        "definitions trivially identical.",
    )
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train = cfg["train"]
    device = get_device()
    configure_runtime(device)
    amp_dtype = parse_amp(train.get("amp", "bf16"), device)

    wm_cfg = yaml.safe_load(Path(cfg["world_model_config"]).read_text(encoding="utf-8"))
    raw = cfg.get("world_model_ckpt")
    if args.joint_ckpt is not None:
        world_model = build_model(wm_cfg).to(device)
    elif raw and Path(raw).is_file():
        world_model, wm_cfg = load_seed_world_model(cfg, device)
    else:
        world_model = build_model(wm_cfg).to(device)
        print(f"random weights from {cfg['world_model_config']}")
    actor, critic = make_actor_critic(cfg, world_model, device)
    if args.joint_ckpt is not None:
        payload = torch.load(args.joint_ckpt, weights_only=False, map_location=device)
        world_model.load_state_dict(payload["world_model"], strict=True)
        actor.load_state_dict(payload["actor"], strict=True)
        critic.load_state_dict(payload["critic"], strict=True)
        print(f"joint {args.joint_ckpt} (env_steps {payload.get('env_steps', '?')})")
    freeze_world_model(world_model)
    actor.eval()
    critic.eval()

    collect_env, eval_env = make_envs(cfg)
    seq_len = int(train["seq_len"])
    buffer = ReplayBuffer(seed=0)
    prefill_random_steps(
        collect_env,
        buffer,
        steps=int(args.steps),
        max_episode_steps=int(train.get("max_episode_steps", 10_000)),
        seq_len=seq_len,
        seed=0,
    )
    batch = buffer.sample(int(args.batch), seq_len)
    obs = batch["obs"].to(device)
    actions = batch["actions"].to(device)

    lam = float(train.get("lam", 0.95))
    with torch.no_grad(), autocast_context(device, amp_dtype):
        rollout = imagine_ahead(
            world_model,
            actor,
            critic,
            obs,
            actions,
            horizon=int(train["horizon"]),
            start_mode="all",
            discount=float(train.get("discount", 0.997)),
        )
        # Fixed: returns target V(s_i), baseline is V(s_i).
        returns, base, weights = imagined_targets(
            rollout.reward, rollout.cont, rollout.value, lam=lam
        )
        adv_fixed = returns - base
        # Pre-fix: the whole H-step rollout was treated as one block and the
        # baseline was the value at the state *after* the action.
        old_returns = lambda_returns(
            rollout.reward[:, 1:], rollout.cont[:, 1:], rollout.value[:, 1:], lam=lam
        )
        adv_old = old_returns - rollout.value[:, 1:]

    for name, adv in (("fixed  Q(s_t,a_t) - V(s_t)", adv_fixed), ("old    G - V(s_t+1)", adv_old)):
        a = adv.float()
        print(
            f"{name:28s} mean={float(a.mean()):+.4f}  std={float(a.std()):.4f}  "
            f"|mean|/std={abs(float(a.mean())) / max(float(a.std()), 1e-8):.2f}  "
            f"frac_same_sign={float((a > 0).float().mean()):.3f}  "
            f"R2_state_only={_r2_state_only(a):.3f}"
        )
    print(f"weights mean={float(weights.float().mean()):.4f}")
    collect_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
