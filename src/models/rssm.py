"""Discrete categorical RSSM (DreamerV2/V3-style dynamics core).

Deterministic state `h` is a layer-normalized GRU (DreamerV3's variant, not a
plain `nn.GRUCell` — the extra layer norm on the gate pre-activations is what
keeps this stable once real training starts in M3). Stochastic latents are
categorical with a small "unimix" floor mixed into every category's
probability, which guarantees no category can ever hit exactly zero
probability — without it, a category that gets unlucky early in training can
never recover a gradient signal (dead latent).

Latents are always named as prior vs posterior, never a bare `z`:

- `z_posterior`: from `h` + encoder embedding (real data / world-model training)
- `z_prior`: from `h` alone (imagination)

Recurrence timing (critical):
    h_t = GRU(h_{t-1}, [z_posterior_{t-1}, action_{t-1}])
    z_*_t ~ Categorical(unimix(logits(h_t, ...)))

Sampling uses a straight-through estimator so gradients reach the logits.

DreamerV3 mechanism details matched here (beyond the basic prior/posterior
split), each independently verifiable against the real dreamerv3 codebase:
- GRU cell: single linear projection -> layer norm -> split into
  reset/candidate/update gates, candidate activated with the same `act` as
  the rest of the network (SiLU by default), not a plain-textbook tanh GRU.
- `unimix`: a uniform floor mixed into every categorical's probabilities.
- Learned initial state (`initial="learned"`): `h_0` is a trainable parameter
  (tanh-squashed) rather than hardcoded zeros, and the initial stochastic
  state is derived from `h_0` through the prior net rather than sampled from
  raw uniform logits. This matters because with `initial="zeros"`, `h_0` can
  never be gradient-corrected to a better starting point — `"learned"` fixes
  that.
- `rec_depth`: optionally run the GRU cell multiple times per outer timestep
  for extra recurrent compute (default 1, matching most DreamerV3 configs).

Scope note: this module still only covers M2 (forward pass / mechanism
correctness, verified via `scripts/verify_rssm_forward.py` and
`scripts/visualize_rssm.py`). KL loss, free bits, and KL balancing between
prior and posterior are M3 — deliberately not implemented here. The M2
verify config uses a 16x16 latent as a cheap shape check; M3 training uses
DreamerV3's 32x32. That is a milestone split, not a VRAM workaround.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def one_hot_action(actions: Tensor, action_dim: int) -> Tensor:
    """Int actions `[...,]` → float one-hot `[..., action_dim]`."""
    return F.one_hot(actions.long(), num_classes=action_dim).float()


def unimix_probs(logits: Tensor, unimix: float = 0.01) -> Tensor:
    """Softmax probabilities with a uniform floor mixed in.

    `probs = (1 - unimix) * softmax(logits) + unimix * uniform`. This is the
    DreamerV3 "unimix" trick: it guarantees every category keeps at least
    `unimix / num_classes` probability mass, so no category can go fully dead
    (zero probability → zero gradient forever) even if its logit collapses.

    Args:
        logits: `[..., classes]` unnormalized scores.
        unimix: floor probability mass to mix in, in `[0, 1)`. DreamerV3 uses
            `0.01`.

    Returns:
        Probabilities `[..., classes]` summing to 1 along the last dim.
    """
    if not 0.0 <= unimix < 1.0:
        raise ValueError(f"unimix must be in [0, 1), got {unimix}")
    probs = F.softmax(logits, dim=-1)
    if unimix > 0.0:
        uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
        probs = (1.0 - unimix) * probs + unimix * uniform
    return probs


def sample_onehot_ste(probs: Tensor) -> Tensor:
    """Sample a one-hot categorical with straight-through gradients.

    Args:
        probs: `[..., classes]` valid probability distributions (e.g. from
            `unimix_probs`). Must sum to 1 along the last dim.

    Returns:
        One-hot samples `[..., classes]` (STE: forward = hard sample,
        backward = gradient flows as if the output were `probs` directly).
    """
    flat = probs.reshape(-1, probs.shape[-1]).float()
    indices = torch.multinomial(flat, num_samples=1).squeeze(-1)
    hard = F.one_hot(indices, num_classes=probs.shape[-1]).float()
    hard = hard.view_as(probs)
    # Straight-through: hard in forward, soft (probs) in backward.
    return hard + probs - probs.detach()


def get_activation(name: str) -> Callable[[Tensor], Tensor]:
    """Resolve an activation function by name (`"silu"`, `"tanh"`, `"relu"`)."""
    name = name.lower()
    if name == "silu":
        return F.silu
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return F.relu
    raise ValueError(f"unknown activation {name!r}")


class GRUCellLayerNorm(nn.Module):
    """DreamerV3-style GRU cell: one linear projection, then layer norm, then
    split into reset/candidate/update gates.

    Two deviations from a textbook `nn.GRUCell` that matter:
    1. Layer norm on the concatenated gate pre-activations — plain GRUCell has
       no normalization there, so its scale can drift as `h` evolves over a
       long rollout (the "recurrence blows up after N steps" failure mode
       called out in `MILESTONES.md` for M2).
    2. The candidate gate is activated with `act` (SiLU by default, matching
       the rest of the network and DreamerV3's actual default), not the
       classic GRU's hardcoded `tanh`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        update_bias: float = -1.0,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.update_bias = update_bias
        self.act = get_activation(act)
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=False)
        self.norm = nn.LayerNorm(3 * hidden_size)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """One GRU step. `x`: `[B, input_size]`, `h`: `[B, hidden_size]`."""
        parts = self.norm(self.linear(torch.cat([x, h], dim=-1)))
        reset, cand, update = torch.chunk(parts, 3, dim=-1)
        reset = torch.sigmoid(reset)
        cand = self.act(reset * cand)
        # Negative update_bias initially favors keeping the old state (update gate
        # starts near sigmoid(-1) ~= 0.27 "let new info in" / 0.73 "keep memory"),
        # which is the same stabilizing init DreamerV3 uses.
        update = torch.sigmoid(update + self.update_bias)
        return update * cand + (1.0 - update) * h


@dataclass
class RSSMState:
    """Single-timestep RSSM state (no time dimension)."""

    h: Tensor  # [B, deter_dim]
    z_posterior: Tensor  # [B, stoch, classes] one-hot (STE)


@dataclass
class RSSMOutput:
    """Full-sequence RSSM outputs. All tensors are `[B, T, ...]`."""

    h: Tensor
    z_prior: Tensor
    z_posterior: Tensor
    prior_logits: Tensor
    posterior_logits: Tensor


class RSSM(nn.Module):
    """Layer-normalized GRU + unimix discrete categorical prior/posterior heads.

    Tensor convention for sequences: `[batch, time, features]`.
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        deter_dim: int = 512,
        stoch: int = 16,
        classes: int = 16,
        hidden: int = 512,
        unimix: float = 0.01,
        act: str = "silu",
        initial: str = "learned",
        rec_depth: int = 1,
    ) -> None:
        super().__init__()
        if stoch < 1 or classes < 2:
            raise ValueError("need stoch >= 1 and classes >= 2")
        if initial not in ("learned", "zeros"):
            raise ValueError(f"initial must be 'learned' or 'zeros', got {initial!r}")
        if rec_depth < 1:
            raise ValueError(f"rec_depth must be >= 1, got {rec_depth}")
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch = stoch
        self.classes = classes
        self.hidden = hidden
        self.unimix = unimix
        self.initial_mode = initial
        self.rec_depth = rec_depth
        self.z_flat_dim = stoch * classes

        # GRU input: previous stochastic latent + previous action.
        self.img_in = nn.Sequential(
            nn.Linear(self.z_flat_dim + action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.cell = GRUCellLayerNorm(hidden, deter_dim, act=act)

        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.z_flat_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.z_flat_dim),
        )

        if self.initial_mode == "learned":
            # Trainable starting deterministic state, DreamerV3-style: init at
            # zero (so behavior matches "zeros" until trained) but gets real
            # gradients once M3's loss backprops through h_0.
            self._initial_deter = nn.Parameter(torch.zeros(deter_dim))

    def initial(self, batch_size: int, device: torch.device | None = None) -> RSSMState:
        """Starting `h` and `z_posterior`.

        With `initial="learned"` (default): `h_0 = tanh(learned_param)`,
        broadcast to the batch, and `z_0` is sampled from the prior net
        conditioned on that `h_0` (matches DreamerV3's `initial="learned"`
        behavior) — so the starting state is part of what training can
        correct, not a fixed assumption.

        With `initial="zeros"`: `h_0` is exactly zero and `z_0` is sampled
        from raw unimix-uniform logits (simpler, useful for tests that want a
        fully deterministic, parameter-free starting point).
        """
        if self.initial_mode == "learned":
            device = device or self._initial_deter.device
            h = torch.tanh(self._initial_deter).unsqueeze(0).expand(batch_size, -1)
            h = h.to(device).contiguous()
            prior_logits = self._logits_to_stoch(self.prior_net(h))
            z_posterior = sample_onehot_ste(unimix_probs(prior_logits, self.unimix))
        else:
            device = device or torch.device("cpu")
            h = torch.zeros(batch_size, self.deter_dim, device=device)
            logits = torch.zeros(batch_size, self.stoch, self.classes, device=device)
            z_posterior = sample_onehot_ste(unimix_probs(logits, self.unimix))
        return RSSMState(h=h, z_posterior=z_posterior)

    def _logits_to_stoch(self, flat_logits: Tensor) -> Tensor:
        return flat_logits.view(-1, self.stoch, self.classes)

    def _recur(self, x: Tensor, h: Tensor) -> Tensor:
        """Apply the GRU cell `rec_depth` times with the same input `x`.

        `rec_depth > 1` gives the deterministic state extra recurrent compute
        per outer timestep (DreamerV3's `dyn_rec_depth` config knob). With the
        default `rec_depth=1` this is exactly one `self.cell(x, h)` call.
        """
        for _ in range(self.rec_depth):
            h = self.cell(x, h)
        return h

    def obs_step(
        self,
        prev_state: RSSMState,
        prev_action: Tensor,
        embed: Tensor,
    ) -> tuple[RSSMState, Tensor, Tensor, Tensor]:
        """One observe step (real embedding available).

        Args:
            prev_state: previous `h` and `z_posterior`.
            prev_action: one-hot action `[B, action_dim]` that led to this obs
                (zeros for the first timestep).
            embed: encoder embedding `[B, embed_dim]` for the current obs.

        Returns:
            (new_state, z_prior, prior_logits, posterior_logits)
            where latents/logits are `[B, stoch, classes]`. Logits are the raw
            (pre-unimix) scores — apply `unimix_probs` before computing KL or
            entropy so those numbers match what was actually sampled.
        """
        if prev_action.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected prev_action [..., {self.action_dim}], got {tuple(prev_action.shape)}"
            )
        if embed.shape[-1] != self.embed_dim:
            raise ValueError(
                f"expected embed [..., {self.embed_dim}], got {tuple(embed.shape)}"
            )

        z_flat = prev_state.z_posterior.reshape(prev_state.z_posterior.shape[0], -1)
        x = self.img_in(torch.cat([z_flat, prev_action], dim=-1))
        h = self._recur(x, prev_state.h)

        prior_logits = self._logits_to_stoch(self.prior_net(h))
        z_prior = sample_onehot_ste(unimix_probs(prior_logits, self.unimix))

        posterior_logits = self._logits_to_stoch(
            self.posterior_net(torch.cat([h, embed], dim=-1))
        )
        z_posterior = sample_onehot_ste(unimix_probs(posterior_logits, self.unimix))

        new_state = RSSMState(h=h, z_posterior=z_posterior)
        return new_state, z_prior, prior_logits, posterior_logits

    def img_step(
        self,
        prev_h: Tensor,
        prev_z_prior: Tensor,
        prev_action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One imagination step (`z_prior` only — no encoder embedding).

        Args:
            prev_h: `[B, deter_dim]`
            prev_z_prior: `[B, stoch, classes]`
            prev_action: `[B, action_dim]` one-hot

        Returns:
            (h, z_prior, prior_logits)
        """
        z_flat = prev_z_prior.reshape(prev_z_prior.shape[0], -1)
        x = self.img_in(torch.cat([z_flat, prev_action], dim=-1))
        h = self._recur(x, prev_h)
        prior_logits = self._logits_to_stoch(self.prior_net(h))
        z_prior = sample_onehot_ste(unimix_probs(prior_logits, self.unimix))
        return h, z_prior, prior_logits

    def observe(self, embeds: Tensor, actions: Tensor) -> RSSMOutput:
        """Roll the RSSM over a real sequence.

        Args:
            embeds: `[B, T, embed_dim]` encoder embeddings per timestep.
            actions: `[B, T, action_dim]` one-hot actions. `actions[:, t]` is the
                action taken *after* observing timestep `t` (used to form `h_{t+1}`).
                For timestep 0 the previous action is treated as zeros.

        Returns:
            `RSSMOutput` with `[B, T, ...]` tensors for `h`, `z_prior`,
            `z_posterior`, and both (pre-unimix) logit tensors.
        """
        if embeds.ndim != 3 or embeds.shape[-1] != self.embed_dim:
            raise ValueError(
                f"expected embeds [B, T, {self.embed_dim}], got {tuple(embeds.shape)}"
            )
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected actions [B, T, {self.action_dim}], got {tuple(actions.shape)}"
            )
        if embeds.shape[:2] != actions.shape[:2]:
            raise ValueError(
                f"embeds/actions batch-time mismatch: {tuple(embeds.shape)} vs {tuple(actions.shape)}"
            )

        batch, time, _ = embeds.shape
        state = self.initial(batch, device=embeds.device)

        hs: list[Tensor] = []
        z_priors: list[Tensor] = []
        z_posteriors: list[Tensor] = []
        prior_logits_t: list[Tensor] = []
        posterior_logits_t: list[Tensor] = []

        zero_action = torch.zeros(batch, self.action_dim, device=embeds.device)
        for t in range(time):
            prev_action = zero_action if t == 0 else actions[:, t - 1]
            state, z_prior, prior_logits, posterior_logits = self.obs_step(
                state, prev_action, embeds[:, t]
            )
            hs.append(state.h)
            z_priors.append(z_prior)
            z_posteriors.append(state.z_posterior)
            prior_logits_t.append(prior_logits)
            posterior_logits_t.append(posterior_logits)

        return RSSMOutput(
            h=torch.stack(hs, dim=1),
            z_prior=torch.stack(z_priors, dim=1),
            z_posterior=torch.stack(z_posteriors, dim=1),
            prior_logits=torch.stack(prior_logits_t, dim=1),
            posterior_logits=torch.stack(posterior_logits_t, dim=1),
        )

    def imagine(
        self,
        initial_h: Tensor,
        initial_z_prior: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Open-loop imagination using `z_prior` only.

        Args:
            initial_h: `[B, deter_dim]` starting deterministic state.
            initial_z_prior: `[B, stoch, classes]` starting stochastic state.
            actions: `[B, T, action_dim]` one-hot imagined actions. `actions[:, t]`
                is applied to transition into imagined step `t`.

        Returns:
            `(h, z_prior, prior_logits)` each `[B, T, ...]`.
        """
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected actions [B, T, {self.action_dim}], got {tuple(actions.shape)}"
            )
        batch, time, _ = actions.shape
        h = initial_h
        z_prior = initial_z_prior
        hs: list[Tensor] = []
        z_priors: list[Tensor] = []
        prior_logits_t: list[Tensor] = []
        for t in range(time):
            h, z_prior, prior_logits = self.img_step(h, z_prior, actions[:, t])
            hs.append(h)
            z_priors.append(z_prior)
            prior_logits_t.append(prior_logits)
        return (
            torch.stack(hs, dim=1),
            torch.stack(z_priors, dim=1),
            torch.stack(prior_logits_t, dim=1),
        )
