"""WorldModel: encoder + RSSM + decoder + reward/continue heads.

Trains on real replay sequences (M3). Imagination / actor-critic is M4.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.decoder import Decoder
from models.encoder import Encoder
from models.heads import ContinueHead, RewardHead, rssm_features
from models.rssm import RSSM, RSSMOutput, one_hot_action
from models.preprocess import nhwc_uint8_to_nchw_float


@dataclass
class WorldModelOutput:
    """Forward outputs for one `[B, T]` batch (all time-aligned)."""

    embeds: Tensor
    rssm: RSSMOutput
    feat: Tensor
    recon: Tensor
    reward_pred: Tensor
    cont_logit: Tensor


class WorldModel(nn.Module):
    """End-to-end world model used for M3 supervised training on replay.

    Decoder and heads condition on `feat = concat(h, flatten(z_posterior))`,
    never on the encoder embedding alone — that keeps the RSSM latents on the
    reconstruction/reward path.
    """

    def __init__(
        self,
        encoder: Encoder,
        rssm: RSSM,
        decoder: Decoder,
        reward_head: RewardHead,
        continue_head: ContinueHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.decoder = decoder
        self.reward_head = reward_head
        self.continue_head = continue_head
        feat_dim = rssm.deter_dim + rssm.z_flat_dim
        if decoder.embed_dim != feat_dim:
            raise ValueError(
                f"decoder.embed_dim={decoder.embed_dim} != feat_dim={feat_dim} "
                f"(deter {rssm.deter_dim} + z_flat {rssm.z_flat_dim})"
            )
        if reward_head.in_dim != feat_dim or continue_head.in_dim != feat_dim:
            raise ValueError("reward/continue heads must match RSSM feature dim")

    @property
    def feat_dim(self) -> int:
        return self.rssm.deter_dim + self.rssm.z_flat_dim

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
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32),
        head_hidden: int = 512,
        head_layers: int = 2,
    ) -> WorldModel:
        """Construct a consistently-sized world model from scalar dims."""
        encoder = Encoder(embed_dim=embed_dim, channels=encoder_channels)
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
        )
        feat_dim = deter_dim + stoch * classes
        decoder = Decoder(embed_dim=feat_dim, channels=decoder_channels)
        reward_head = RewardHead(feat_dim, hidden=head_hidden, layers=head_layers)
        continue_head = ContinueHead(feat_dim, hidden=head_hidden, layers=head_layers)
        return cls(encoder, rssm, decoder, reward_head, continue_head)

    def encode(self, obs_u8: Tensor) -> Tensor:
        """uint8 obs `[B, T, H, W, C]` or `[B, H, W, C]` → embeds with time dim."""
        squeeze = obs_u8.ndim == 4
        if squeeze:
            obs_u8 = obs_u8.unsqueeze(1)
        if obs_u8.ndim != 5:
            raise ValueError(f"expected obs [B,T,H,W,C] or [B,H,W,C], got {tuple(obs_u8.shape)}")
        batch, time = obs_u8.shape[:2]
        flat = obs_u8.reshape(batch * time, *obs_u8.shape[2:])
        embeds = self.encoder(nhwc_uint8_to_nchw_float(flat))
        embeds = embeds.view(batch, time, -1)
        return embeds.squeeze(1) if squeeze else embeds

    def forward(
        self,
        obs_u8: Tensor,
        actions: Tensor,
        *,
        actions_onehot: bool = False,
    ) -> WorldModelOutput:
        """Observe a sequence and predict recon / reward / continue.

        Args:
            obs_u8: `[B, T, 64, 64, 3]` uint8
            actions: `[B, T]` int64 or `[B, T, action_dim]` one-hot
            actions_onehot: if True, `actions` is already one-hot

        Returns:
            `WorldModelOutput` with:
              - embeds `[B, T, embed_dim]`
              - rssm fields `[B, T, ...]`
              - feat `[B, T, feat_dim]`
              - recon `[B, T, 3, 64, 64]` in `[-1, 1]`
              - reward_pred / cont_logit `[B, T, 1]`
        """
        embeds = self.encode(obs_u8)
        if actions_onehot:
            act = actions.float()
        else:
            act = one_hot_action(actions, self.rssm.action_dim)
        rssm_out = self.rssm.observe(embeds, act)
        feat = rssm_features(rssm_out.h, rssm_out.z_posterior)
        batch, time, feat_dim = feat.shape
        flat_feat = feat.reshape(batch * time, feat_dim)
        recon = self.decoder(flat_feat).view(batch, time, 3, 64, 64)
        reward_pred = self.reward_head(flat_feat).view(batch, time, 1)
        cont_logit = self.continue_head(flat_feat).view(batch, time, 1)
        return WorldModelOutput(
            embeds=embeds,
            rssm=rssm_out,
            feat=feat,
            recon=recon,
            reward_pred=reward_pred,
            cont_logit=cont_logit,
        )
