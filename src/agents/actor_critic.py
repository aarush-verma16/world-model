"""Actor and critic MLPs on RSSM features `[h, z]` (DreamerV3 M4).

Trained only on imagined rollouts. The world model (encoder, RSSM, decoder,
reward head, continue head) stays frozen. Discrete Crafter actions use the
same unimix + straight-through one-hot sampler as the RSSM latents.

Never name a rollout latent `z_t`: the seed is `z_posterior` from a real
window; every imagined step is `z_prior`.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from models.heads import MLPHead
from models.rssm import sample_onehot_ste, unimix_probs


class Actor(nn.Module):
    """Categorical policy: `feat → action_dim` logits, sampled with unimix STE.

    Args:
        feat_dim: `deter_dim + stoch * classes` (same as decoder / heads).
        action_dim: Crafter is 17.
    """

    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        hidden: int = 512,
        layers: int = 2,
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        if action_dim < 2:
            raise ValueError(f"action_dim must be >= 2, got {action_dim}")
        self.net = MLPHead(feat_dim, action_dim, hidden=hidden, layers=layers)
        self.feat_dim = feat_dim
        self.action_dim = action_dim
        self.unimix = float(unimix)

    def forward(self, feat: Tensor) -> Tensor:
        """`feat` `[..., feat_dim]` → logits `[..., action_dim]`."""
        return self.net(feat)

    def policy(self, feat: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample an action and return log-prob / entropy for the actor loss.

        Returns:
            `action` `[..., action_dim]` STE one-hot (hard forward, soft backward),
            `log_prob` `[...]` of the hard sample under the unimix dist,
            `entropy` `[...]` of the unimix dist,
            `probs` `[..., action_dim]`.
        """
        logits = self.forward(feat)
        probs = unimix_probs(logits, self.unimix)
        action = sample_onehot_ste(probs)
        log_prob = (action.detach() * probs.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        return action, log_prob, entropy, probs


class Critic(nn.Module):
    """Symlog two-hot value head, same discrete-regression recipe as reward.

    Bins live on this module (not shared weights with the frozen reward head)
    so the critic can move independently. Default range matches DreamerV3 /
    the world-model reward head (`symlog([-20, 20])`, 255 bins).

    `out_scale=0.0` is DreamerV3's `critic.outscale`: the last layer starts at
    zero so the initial value is exactly 0 everywhere.
    """

    def __init__(
        self,
        feat_dim: int,
        hidden: int = 512,
        layers: int = 2,
        num_bins: int = 255,
        low: float = -20.0,
        high: float = 20.0,
        out_scale: float | None = 0.0,
    ) -> None:
        super().__init__()
        self.net = MLPHead(
            feat_dim, num_bins, hidden=hidden, layers=layers, out_scale=out_scale
        )
        self.feat_dim = feat_dim
        self.register_buffer("bins", torch.linspace(low, high, num_bins))

    def forward(self, feat: Tensor) -> Tensor:
        """`feat` `[..., feat_dim]` → two-hot logits `[..., num_bins]`."""
        return self.net(feat)


class SlowCritic(nn.Module):
    """EMA copy of the critic, used as a second regression target.

    DreamerV3 regresses the critic onto the λ-return **and** onto this slow
    copy. Without the anchor the critic and the bootstrap inside its own target
    chase each other; the visible symptom is a saturated actor, not a bad
    critic metric. `fraction` is the paper's `slow_target_fraction` (0.02).
    """

    def __init__(
        self,
        critic: Critic,
        fraction: float = 0.02,
        update_every: int = 1,
    ) -> None:
        super().__init__()
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        self.critic = copy.deepcopy(critic)
        self.critic.requires_grad_(False)
        self.fraction = float(fraction)
        self.update_every = max(1, int(update_every))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, critic: Critic) -> None:
        """Mix `critic` weights into this copy (call once per critic step)."""
        if int(self.updates) % self.update_every == 0:
            mix = self.fraction
            for src, dst in zip(critic.parameters(), self.critic.parameters()):
                dst.mul_(1.0 - mix).add_(src.detach(), alpha=mix)
        self.updates += 1

    def reset(self, critic: Critic) -> None:
        """Copy `critic` in wholesale (used when a checkpoint has no EMA state)."""
        self.critic.load_state_dict(critic.state_dict(), strict=True)

    @property
    def bins(self) -> Tensor:
        return self.critic.bins

    def forward(self, feat: Tensor) -> Tensor:
        """`feat` `[..., feat_dim]` → two-hot logits `[..., num_bins]`."""
        return self.critic(feat)
