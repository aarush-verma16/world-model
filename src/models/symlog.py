"""Symlog transform and two-hot discrete regression (DreamerV3 reward/critic).

Crafter rewards are almost always exactly 0 (sparse achievements give +1,
damage gives -0.1, etc.), so a plain MSE reward head learns to always predict
~0 and looks "trained" while carrying no signal about the rare nonzero steps.
DreamerV3 instead treats the (symlog-transformed) reward as a classification
target over a fixed set of bins ("two-hot": the two nearest bins get partial
mass so the target is exact even though the bins are discrete), trained with
categorical cross-entropy. This scales gracefully across unknown reward
magnitudes and does not saturate the way squared error does.

`symlog(x) = sign(x) * log(1 + |x|)` compresses large magnitudes so a fixed
bin range (`[-20, 20]` in symlog space) covers a huge dynamic range in real
units; `symexp` is its inverse, used to decode a prediction back to reward
units.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def symlog(x: Tensor) -> Tensor:
    """`sign(x) * log(1 + |x|)`. Compresses magnitude, preserves sign."""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: Tensor) -> Tensor:
    """Inverse of `symlog`: `sign(x) * (exp(|x|) - 1)`."""
    return torch.sign(x) * (torch.exp(x.abs()) - 1.0)


def twohot_encode(x: Tensor, bins: Tensor) -> Tensor:
    """Encode real values as a two-hot distribution over `bins`.

    Args:
        x: `[...]` real values, already in the same (e.g. symlog) space as
            `bins`. Values outside `[bins[0], bins[-1]]` are clamped.
        bins: `[K]` ascending bin centers.

    Returns:
        `[..., K]` distribution: the two bins bracketing each `x` get partial
        mass proportional to how close `x` is to each (exact, not rounded).
    """
    num_bins = bins.shape[0]
    x = x.clamp(float(bins[0]), float(bins[-1]))
    # Index of the largest bin <= x, clamped so `above = below + 1` is valid.
    below = torch.searchsorted(bins, x.contiguous(), right=True) - 1
    below = below.clamp(0, num_bins - 2)
    above = below + 1
    bin_below = bins[below]
    bin_above = bins[above]
    span = (bin_above - bin_below).clamp_min(1e-8)
    weight_above = (x - bin_below) / span
    weight_below = 1.0 - weight_above

    twohot = torch.zeros(*x.shape, num_bins, device=x.device, dtype=weight_above.dtype)
    twohot.scatter_add_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
    twohot.scatter_add_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
    return twohot


def twohot_decode(probs: Tensor, bins: Tensor) -> Tensor:
    """Expected bin value under `probs`: `[..., K] -> [...]` (still symlog space)."""
    return (probs * bins).sum(dim=-1)


def symlog_twohot_loss(logits: Tensor, bins: Tensor, target: Tensor) -> Tensor:
    """Categorical NLL of `target` under the two-hot label induced by `bins`.

    Args:
        logits: `[..., K]` unnormalized scores from the reward/value head.
        bins: `[K]` ascending bin centers, in symlog space.
        target: `[...]` real-valued targets (real reward units, NOT symlog).

    Returns:
        `[...]` per-element NLL, in nats.
    """
    label = twohot_encode(symlog(target), bins)
    logp = F.log_softmax(logits, dim=-1)
    return -(label * logp).sum(dim=-1)


def symlog_twohot_mean(logits: Tensor, bins: Tensor) -> Tensor:
    """Decode a scalar prediction (real units) from two-hot logits."""
    probs = F.softmax(logits, dim=-1)
    return symexp(twohot_decode(probs, bins))
