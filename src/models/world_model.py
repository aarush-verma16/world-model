"""WorldModel: encoder + RSSM + decoder + reward/continue heads (DreamerV3 M3).

Trains on real replay sequences. Imagination / actor-critic is M4
(`agents.actor_critic`, `training.imagine`, `training.ac_step`).

One decoder, live weights, decoding from `feat = concat(h, flatten(z))` —
this is the paper's actual graph (`decoder(get_feat(post))` in
NM512/dreamerv3-torch's `WorldModel._train`). There is no separate "embed
recon" path: an auxiliary decoder trained straight off the encoder embedding
turns into a plain autoencoder that makes low pixel loss trivially achievable
without the RSSM carrying anything, which is exactly the bypass that let two
weeks of M3 runs report falling loss while `[h,z]` stayed empty. See
`docs/experiments.md` ("DreamerV3 M3 reset") for the full postmortem.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.decoder import Decoder
from models.encoder import Encoder
from models.heads import (
    ContinueHead,
    InventoryHead,
    LocalMapHead,
    RewardHead,
    SpatialHead,
    rssm_features,
)
from models.preprocess import nhwc_uint8_to_nchw_unit
from models.rssm import RSSM, RSSMOutput, one_hot_action


@dataclass
class WorldModelOutput:
    """Forward outputs for one `[B, T]` batch (all time-aligned)."""

    embeds: Tensor
    rssm: RSSMOutput
    feat: Tensor
    recon: Tensor
    reward_pred: Tensor
    cont_logit: Tensor

    def __getattr__(self, name: str):
        if name in {"recon_embed", "recon_bottleneck", "recon_map"}:
            raise AttributeError(
                "WorldModelOutput has no 'recon_embed' — the aux embed decoder "
                "was removed. Reload notebooks/05_train_world_model.ipynb from "
                "disk (do not keep an unsaved old buffer), then "
                "Kernel → Restart → Run All. The training cell must call "
                "v_out.recon and model.video_predict, never recon_embed."
            )
        raise AttributeError(f"WorldModelOutput has no attribute {name!r}")


@dataclass
class VideoPrediction:
    """Open-loop rollout for visual/behavioral diagnosis (DreamerV3 `video_pred`).

    `context_recon` decodes `z_posterior` for the first `context_len` steps
    (real observations available). `imagined_recon` decodes `z_prior` for the
    remaining steps, rolled forward with **no** access to the real frames —
    only the actions and the world model's own dynamics. A healthy world
    model keeps `imagined_recon` Crafter-like for several steps before
    degrading; garbage from the first imagined step means the dynamics model
    itself (not just the decoder) is broken.
    """

    context_recon: Tensor  # [B, context_len, 3, 64, 64]
    imagined_recon: Tensor  # [B, T - context_len, 3, 64, 64]
    imagined_reward: Tensor  # [B, T - context_len] decoded scalar reward


class WorldModel(nn.Module):
    """End-to-end world model used for M3 supervised training on replay."""

    def __init__(
        self,
        encoder: Encoder,
        rssm: RSSM,
        decoder: Decoder,
        reward_head: RewardHead,
        continue_head: ContinueHead,
        inventory_head: InventoryHead | None = None,
        spatial_head: SpatialHead | None = None,
        local_map_head: LocalMapHead | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.decoder = decoder
        self.reward_head = reward_head
        self.continue_head = continue_head
        # Optional (finding 40, m18 only): `None` for every m6/m17-style
        # config, so `state_dict()` has no `inventory_head.*` keys and old
        # checkpoints keep loading with `strict=True` unchanged.
        self.inventory_head = inventory_head
        # Optional (finding 45, m19 only). `None` keeps m18's state_dict
        # free of `spatial_head.*` keys.
        self.spatial_head = spatial_head
        # Optional (finding 47, m20 only). `None` keeps m19's state_dict
        # free of `local_map_head.*` keys.
        self.local_map_head = local_map_head
        # Where real collect / eval get the actor's extra columns:
        # "truth" = `CrafterEnv` info (m18/m19), "predicted" = this model's
        # own heads, same as imagination (m20: pixel-only actor at test).
        # A plain attribute, not a buffer, so it never enters `state_dict`.
        self.actor_input_source = "truth"
        feat_dim = rssm.deter_dim + rssm.z_flat_dim
        if encoder.embed_dim != rssm.embed_dim:
            raise ValueError(
                f"encoder.embed_dim={encoder.embed_dim} != rssm.embed_dim={rssm.embed_dim}"
            )
        if decoder.embed_dim != feat_dim:
            raise ValueError(
                f"decoder.embed_dim={decoder.embed_dim} != feat_dim={feat_dim} "
                f"(deter {rssm.deter_dim} + z_flat {rssm.z_flat_dim})"
            )
        if reward_head.in_dim != feat_dim or continue_head.in_dim != feat_dim:
            raise ValueError("reward/continue heads must match RSSM feature dim")
        if inventory_head is not None and inventory_head.in_dim != feat_dim:
            raise ValueError("inventory_head must match RSSM feature dim")
        if spatial_head is not None and spatial_head.in_dim != feat_dim:
            raise ValueError("spatial_head must match RSSM feature dim")
        if local_map_head is not None and local_map_head.in_dim != feat_dim:
            raise ValueError("local_map_head must match RSSM feature dim")

    @property
    def feat_dim(self) -> int:
        return self.rssm.deter_dim + self.rssm.z_flat_dim

    @property
    def actor_extra_dim(self) -> int:
        """Columns concatenated onto `feat` for the actor and critic.

        0 for m6/m17. Inventory counts for m18. Inventory plus facing one-hot,
        faced-object bit, and nearby multi-hot for m19. m20 adds the local
        map as per-cell material + object distributions.
        """
        extra = 0
        if self.inventory_head is not None:
            extra += int(self.inventory_head.n_items)
        if self.spatial_head is not None:
            n = int(self.spatial_head.n_materials)
            extra += n + 1 + n
        if self.local_map_head is not None:
            head = self.local_map_head
            extra += head.cells * (head.n_material_classes + head.n_object_classes)
        return extra

    @classmethod
    def from_config_dims(
        cls,
        *,
        embed_dim: int,
        encoder_channels: tuple[int, ...],
        action_dim: int,
        deter_dim: int,
        stoch: int,
        classes: int,
        hidden: int,
        unimix: float = 0.01,
        act: str = "silu",
        initial: str = "learned",
        rec_depth: int = 1,
        prior_layers: int = 2,
        decoder_channels: tuple[int, ...] = (512, 256, 128, 64),
        head_hidden: int = 512,
        head_layers: int = 2,
        encoder_blocks: int = 2,
        decoder_blocks: int = 0,
        output_activation: str = "linear",
        reward_num_bins: int = 255,
        reward_low: float = -20.0,
        reward_high: float = 20.0,
        inventory_n_items: int | None = None,
        inventory_num_classes: int = 10,
        spatial_n_materials: int | None = None,
        local_map_cells: int | None = None,
        local_map_material_classes: int = 13,
        local_map_object_classes: int = 8,
    ) -> WorldModel:
        """Construct a consistently-sized world model from scalar dims.

        `output_activation="linear"` (DreamerV3 `cnn_sigmoid=False`, pixels in
        `[0, 1]`) is the M3 default; pass `"tanh"` only to reproduce the old
        `[-1, 1]` M1-style pixel convention.

        `inventory_n_items` (finding 40, m18 only): set to build an
        `InventoryHead` (default `None` — no head, matching m6/m17 exactly).

        `spatial_n_materials` (finding 45, m19 only): set to build a
        `SpatialHead`. Default `None` leaves m18 checkpoints loadable.

        `local_map_cells` (finding 47, m20 only): set to build a
        `LocalMapHead`. Default `None` leaves m19 checkpoints loadable.
        """
        encoder = Encoder(
            embed_dim=embed_dim,
            channels=encoder_channels,
            blocks=encoder_blocks,
        )
        rssm = RSSM(
            embed_dim=embed_dim,
            action_dim=action_dim,
            deter_dim=deter_dim,
            stoch=stoch,
            classes=classes,
            hidden=hidden,
            unimix=unimix,
            act=act,
            initial=initial,
            rec_depth=rec_depth,
            prior_layers=prior_layers,
        )
        feat_dim = deter_dim + stoch * classes
        decoder = Decoder(
            embed_dim=feat_dim,
            channels=decoder_channels,
            start_res=4,
            blocks=decoder_blocks,
            output_activation=output_activation,
        )
        reward_head = RewardHead(
            feat_dim,
            hidden=head_hidden,
            layers=head_layers,
            num_bins=reward_num_bins,
            low=reward_low,
            high=reward_high,
        )
        continue_head = ContinueHead(feat_dim, hidden=head_hidden, layers=head_layers)
        inventory_head = None
        if inventory_n_items is not None:
            inventory_head = InventoryHead(
                feat_dim,
                n_items=int(inventory_n_items),
                num_classes=int(inventory_num_classes),
                hidden=head_hidden,
                layers=head_layers,
            )
        spatial_head = None
        if spatial_n_materials is not None:
            spatial_head = SpatialHead(
                feat_dim,
                n_materials=int(spatial_n_materials),
                hidden=head_hidden,
                layers=head_layers,
            )
        local_map_head = None
        if local_map_cells is not None:
            local_map_head = LocalMapHead(
                feat_dim,
                cells=int(local_map_cells),
                n_material_classes=int(local_map_material_classes),
                n_object_classes=int(local_map_object_classes),
                hidden=head_hidden,
                layers=head_layers,
            )
        return cls(
            encoder,
            rssm,
            decoder,
            reward_head,
            continue_head,
            inventory_head,
            spatial_head,
            local_map_head,
        )

    def encode(self, obs_u8: Tensor) -> Tensor:
        """uint8 obs `[B, T, H, W, C]` or `[B, H, W, C]` → embeds with time dim."""
        squeeze = obs_u8.ndim == 4
        if squeeze:
            obs_u8 = obs_u8.unsqueeze(1)
        if obs_u8.ndim != 5:
            raise ValueError(f"expected obs [B,T,H,W,C] or [B,H,W,C], got {tuple(obs_u8.shape)}")
        batch, time = obs_u8.shape[:2]
        flat = obs_u8.reshape(batch * time, *obs_u8.shape[2:])
        embeds = self.encoder(nhwc_uint8_to_nchw_unit(flat))
        embeds = embeds.view(batch, time, -1)
        return embeds.squeeze(1) if squeeze else embeds

    def predict_inventory(self, feat: Tensor) -> Tensor:
        """`feat` `[..., feat_dim]` -> inventory logits `[..., n_items, num_classes]`.

        Raises if this model has no `inventory_head` (m6/m17-style config).
        """
        if self.inventory_head is None:
            raise RuntimeError(
                "predict_inventory called on a WorldModel with no inventory_head "
                "(inventory_n_items was not set in from_config_dims)"
            )
        return self.inventory_head(feat)

    def predict_spatial(self, feat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """`feat` `[..., feat_dim]` -> material, object, nearby logits.

        Raises if this model has no `spatial_head` (m6/m18-style config).
        """
        if self.spatial_head is None:
            raise RuntimeError(
                "predict_spatial called on a WorldModel with no spatial_head "
                "(spatial_n_materials was not set in from_config_dims)"
            )
        return self.spatial_head(feat)

    def predict_local_map(self, feat: Tensor) -> tuple[Tensor, Tensor]:
        """`feat` `[..., feat_dim]` -> `(material [..., C, M], object [..., C, O])` logits.

        Raises if this model has no `local_map_head` (m6-m19-style config).
        """
        if self.local_map_head is None:
            raise RuntimeError(
                "predict_local_map called on a WorldModel with no local_map_head "
                "(local_map_cells was not set in from_config_dims)"
            )
        return self.local_map_head(feat)

    def decode(self, feat: Tensor) -> Tensor:
        """`feat` `[..., feat_dim]` → images `[..., 3, 64, 64]` (any leading dims)."""
        lead = feat.shape[:-1]
        flat = self.decoder(feat.reshape(-1, feat.shape[-1]))
        return flat.view(*lead, 3, 64, 64)

    def forward(
        self,
        obs_u8: Tensor,
        actions: Tensor,
        *,
        actions_onehot: bool = False,
        is_first: Tensor | None = None,
    ) -> WorldModelOutput:
        """Observe a sequence and predict recon / reward / continue.

        Args:
            obs_u8: `[B, T, 64, 64, 3]` uint8
            actions: `[B, T]` int64 or `[B, T, action_dim]` one-hot
            actions_onehot: if True, `actions` is already one-hot
            is_first: optional `[B, T]` flags; see `RSSM.observe`.
        """
        if obs_u8.ndim != 5:
            raise ValueError(f"expected obs [B,T,H,W,C], got {tuple(obs_u8.shape)}")
        embeds = self.encode(obs_u8)
        act = actions.float() if actions_onehot else one_hot_action(actions, self.rssm.action_dim)
        rssm_out = self.rssm.observe(embeds, act, is_first=is_first)
        feat = rssm_features(rssm_out.h, rssm_out.z_posterior)
        recon = self.decode(feat)
        flat_feat = feat.reshape(-1, feat.shape[-1])
        batch, time = obs_u8.shape[:2]
        reward_pred = self.reward_head(flat_feat).view(batch, time, -1)
        cont_logit = self.continue_head(flat_feat).view(batch, time, 1)
        return WorldModelOutput(
            embeds=embeds,
            rssm=rssm_out,
            feat=feat,
            recon=recon,
            reward_pred=reward_pred,
            cont_logit=cont_logit,
        )

    @torch.no_grad()
    def video_predict(
        self,
        obs_u8: Tensor,
        actions: Tensor,
        *,
        context_len: int,
    ) -> VideoPrediction:
        """DreamerV3-style open-loop video prediction (diagnostic only).

        Observes the first `context_len` steps with real frames
        (`z_posterior`), then rolls the RSSM forward for the rest using only
        actions and `z_prior` — no encoder access — and decodes both halves.
        This is the actual test of whether the *dynamics* work, independent
        of how good posterior reconstruction looks (posterior recon can look
        fine even when the prior/imagination path is broken).

        Args:
            obs_u8: `[B, T, 64, 64, 3]` uint8, `T > context_len`.
            actions: `[B, T]` int64.
            context_len: number of real steps to condition on before
                switching to imagination.
        """
        from models.symlog import symlog_twohot_mean

        batch, time = obs_u8.shape[:2]
        if context_len < 1 or context_len >= time:
            raise ValueError(f"need 1 <= context_len < T={time}, got {context_len}")
        act = one_hot_action(actions, self.rssm.action_dim)

        embeds = self.encode(obs_u8[:, :context_len])
        rssm_ctx = self.rssm.observe(embeds, act[:, :context_len])
        ctx_feat = rssm_features(rssm_ctx.h, rssm_ctx.z_posterior)
        context_recon = self.decode(ctx_feat)

        h0 = rssm_ctx.h[:, -1]
        z0 = rssm_ctx.z_posterior[:, -1]
        future_actions = act[:, context_len:]
        h_img, z_img, _ = self.rssm.imagine(h0, z0, future_actions)
        img_feat = rssm_features(h_img, z_img)
        imagined_recon = self.decode(img_feat)
        flat_img_feat = img_feat.reshape(-1, img_feat.shape[-1])
        reward_logits = self.reward_head(flat_img_feat).view(batch, time - context_len, -1)
        imagined_reward = symlog_twohot_mean(reward_logits, self.reward_head.bins)

        return VideoPrediction(
            context_recon=context_recon,
            imagined_recon=imagined_recon,
            imagined_reward=imagined_reward,
        )
