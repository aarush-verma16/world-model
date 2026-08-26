"""Lambda-returns and DreamerV3 percentile return normalization."""

from __future__ import annotations

import torch
from torch import Tensor


def lambda_returns(
    reward: Tensor,
    cont: Tensor,
    value: Tensor,
    *,
    lam: float,
) -> Tensor:
    """λ-returns over an imagined horizon.

    `reward[:, t]`, `cont[:, t]`, and `value[:, t]` are all properties of the
    imagined state after transition `t` (`z_prior` at that step). The last
    step bootstraps with `value[:, -1]` (no extra imagination step).

    Args:
        reward: `[N, H]` decoded scalar rewards (real units).
        cont: `[N, H]` continue / discount in `[0, 1]` (already includes
            `discount` if the caller multiplied it in).
        value: `[N, H]` decoded scalar values (real units).
        lam: λ in `[0, 1]`. DreamerV3 default 0.95.

    Returns:
        `returns` `[N, H]` in the same units as `reward` / `value`.
    """
    if reward.shape != cont.shape or reward.shape != value.shape:
        raise ValueError(
            f"reward/cont/value shape mismatch: {tuple(reward.shape)} "
            f"{tuple(cont.shape)} {tuple(value.shape)}"
        )
    if reward.ndim != 2:
        raise ValueError(f"expected [N, H], got {tuple(reward.shape)}")
    horizon = reward.shape[1]
    acc = value[:, -1]
    outs: list[Tensor] = []
    for t in range(horizon - 1, -1, -1):
        acc = reward[:, t] + cont[:, t] * ((1.0 - lam) * value[:, t] + lam * acc)
        outs.append(acc)
    outs.reverse()
    return torch.stack(outs, dim=1)


class PercentileReturnNorm:
    """EMA of the 5th–95th percentile range; divide returns by `max(limit, range)`.

    Matches DreamerV3 return normalization: a few +1 Crafter achievements
    must not dominate the actor. Percentiles are computed on the detached
    batch of returns; the scale is an EMA so a single batch cannot jump it.
    """

    def __init__(
        self,
        decay: float = 0.99,
        low_q: float = 0.05,
        high_q: float = 0.95,
        limit: float = 1.0,
    ) -> None:
        self.decay = float(decay)
        self.low_q = float(low_q)
        self.high_q = float(high_q)
        self.limit = float(limit)
        self._low: Tensor | None = None
        self._high: Tensor | None = None

    @torch.no_grad()
    def update(self, returns: Tensor) -> Tensor:
        """Update EMA from `returns` and return the current scale (scalar tensor)."""
        flat = returns.detach().float().reshape(-1)
        lo = torch.quantile(flat, self.low_q)
        hi = torch.quantile(flat, self.high_q)
        if self._low is None or self._high is None:
            self._low = lo
            self._high = hi
        else:
            self._low = self._low.to(device=lo.device, dtype=lo.dtype)
            self._high = self._high.to(device=lo.device, dtype=lo.dtype)
            d = self.decay
            self._low = d * self._low + (1.0 - d) * lo
            self._high = d * self._high + (1.0 - d) * hi
        return (self._high - self._low).clamp_min(self.limit)

    def normalize(self, returns: Tensor) -> tuple[Tensor, Tensor]:
        """Return `(returns / scale, scale)` after updating the EMA.

        `returns` may require grad; the scale does not.
        """
        scale = self.update(returns)
        return returns / scale, scale

    def state_dict(self) -> dict[str, float | None]:
        return {
            "low": None if self._low is None else float(self._low),
            "high": None if self._high is None else float(self._high),
            "decay": self.decay,
            "low_q": self.low_q,
            "high_q": self.high_q,
            "limit": self.limit,
        }

    def load_state_dict(self, state: dict) -> None:
        low = state.get("low")
        high = state.get("high")
        self._low = None if low is None else torch.tensor(float(low))
        self._high = None if high is None else torch.tensor(float(high))
        if "decay" in state:
            self.decay = float(state["decay"])
        if "limit" in state:
            self.limit = float(state["limit"])
