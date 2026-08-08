"""Auxiliary prediction heads on RSSM features `[h, z]`.

Reward and continue heads are what make imagined rollouts useful for the
actor-critic: they score trajectories without touching the real environment.
Both take the same feature vector `feat = concat(h, flatten(z))`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def rssm_features(h: Tensor, z: Tensor) -> Tensor:
    """Build Dreamer-style features from deterministic + stochastic state.

    Args:
        h: `[..., deter_dim]`
        z: `[..., stoch, classes]` one-hot (or soft) categorical samples

    Returns:
        `feat` `[..., deter_dim + stoch * classes]`
    """
    if z.ndim < 2:
        raise ValueError(f"expected z with at least 2 dims [..., stoch, classes], got {tuple(z.shape)}")
    z_flat = z.reshape(*z.shape[:-2], -1)
    if h.shape[:-1] != z_flat.shape[:-1]:
        raise ValueError(
            f"h/z batch-time mismatch: h {tuple(h.shape)} vs z {tuple(z.shape)}"
        )
    return torch.cat([h, z_flat], dim=-1)


class MLPHead(nn.Module):
    """Small MLP: `feat → out_dim` with LayerNorm + SiLU hidden layers."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 512,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")
        mods: list[nn.Module] = []
        prev = in_dim
        for _ in range(layers):
            mods.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.LayerNorm(hidden),
                    nn.SiLU(),
                ]
            )
            prev = hidden
        mods.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*mods)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, feat: Tensor) -> Tensor:
        """Args: `feat` `[..., in_dim]` → `[..., out_dim]`."""
        if feat.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected feat [..., {self.in_dim}], got {tuple(feat.shape)}"
            )
        return self.net(feat)


class RewardHead(MLPHead):
    """Predict scalar reward from `[h, z]`. Output `[..., 1]`."""

    def __init__(self, in_dim: int, hidden: int = 512, layers: int = 2) -> None:
        super().__init__(in_dim=in_dim, out_dim=1, hidden=hidden, layers=layers)


class ContinueHead(MLPHead):
    """Predict continue logit (episode not done) from `[h, z]`. Output `[..., 1]`.

    Train with BCE-with-logits against `continue = 1 - terminated` (and usually
    treating truncation as continue=1 so the bootstrap can still run).
    """

    def __init__(self, in_dim: int, hidden: int = 512, layers: int = 2) -> None:
        super().__init__(in_dim=in_dim, out_dim=1, hidden=hidden, layers=layers)
