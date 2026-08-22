"""WorldModel: skip-free encoder + RSSM + decoder + reward/continue heads.

Trains on real replay sequences (M3). Imagination / actor-critic is M4.

The encoder is the skip-free CNN (`Encoder`), not M1's `PerceptionAutoencoder`.
That U-Net was wired in as an aux recon path so the dashboard could show a
sharp "encoder works" panel. Its `stem_to_rgb(skips[0])` copies the real
frame and never has to put sprites/HUD into the embedding the RSSM actually
consumes — which is why `[h,z]` and "embed, no skips" stayed smeared while
the middle panel looked solved. M1 already proved skip recon; it does not
belong on the world-model training graph.

Two skip-free decode heads:
  - `decoder`: `[h, z_posterior]` → pixels (the imagination path)
  - `embed_decoder`: encoder embedding → pixels (supervises what RSSM sees)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.decoder import Decoder
from models.encoder import Encoder
from models.heads import ContinueHead, RewardHead, rssm_features
from models.preprocess import nhwc_uint8_to_nchw_float
from models.rssm import RSSM, RSSMOutput, one_hot_action


@dataclass
class WorldModelOutput:
    """Forward outputs for one `[B, T]` batch (all time-aligned)."""

    embeds: Tensor
    rssm: RSSMOutput
    feat: Tensor
    recon: Tensor
    recon_embed: Tensor
    recon_bottleneck: Tensor
    reward_pred: Tensor
    cont_logit: Tensor


class WorldModel(nn.Module):
    """End-to-end world model used for M3 supervised training on replay.

    Primary decoder conditions on `feat = concat(h, flatten(z_posterior))`.
    `embed_decoder` reconstructs from the encoder embedding alone so the
    vector that feeds `z_posterior` is directly supervised (no skip copy).
    `recon_embed` and `recon_bottleneck` are the same skip-free embed decode
    (two names so existing logs/dashboard keep working).
    """

    def __init__(
        self,
        encoder: Encoder,
        rssm: RSSM,
        decoder: Decoder,
        embed_decoder: Decoder,
        reward_head: RewardHead,
        continue_head: ContinueHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.decoder = decoder
        self.embed_decoder = embed_decoder
        # Alias used by older checkpoints/docs; same module.
        self.embed_decoder_bottleneck = embed_decoder
        self.reward_head = reward_head
        self.continue_head = continue_head
        feat_dim = rssm.deter_dim + rssm.z_flat_dim
        if decoder.embed_dim != feat_dim:
            raise ValueError(
                f"decoder.embed_dim={decoder.embed_dim} != feat_dim={feat_dim} "
                f"(deter {rssm.deter_dim} + z_flat {rssm.z_flat_dim})"
            )
        if encoder.embed_dim != rssm.embed_dim:
            raise ValueError(
                f"encoder.embed_dim={encoder.embed_dim} != rssm.embed_dim={rssm.embed_dim}"
            )
        if embed_decoder.embed_dim != rssm.embed_dim:
            raise ValueError(
                f"embed_decoder.embed_dim={embed_decoder.embed_dim} != "
                f"rssm.embed_dim={rssm.embed_dim}"
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
        decoder_channels: tuple[int, ...] = (512, 256, 128, 64),
        head_hidden: int = 512,
        head_layers: int = 2,
        stem_channels: int = 64,
        spatial: int = 4,
    ) -> WorldModel:
        """Construct a consistently-sized world model from scalar dims.

        `stem_channels` / `spatial` are ignored leftover kwargs from the
        U-Net / 8×8-bottleneck experiments. Encoder is always skip-free
        4-stride CNN; both decoders always start at 4×4.
        """
        del stem_channels, spatial
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
        decoder = Decoder(embed_dim=feat_dim, channels=decoder_channels, start_res=4)
        embed_decoder = Decoder(
            embed_dim=embed_dim, channels=decoder_channels, start_res=4
        )
        reward_head = RewardHead(feat_dim, hidden=head_hidden, layers=head_layers)
        continue_head = ContinueHead(feat_dim, hidden=head_hidden, layers=head_layers)
        return cls(encoder, rssm, decoder, embed_decoder, reward_head, continue_head)

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
            `WorldModelOutput` with recon from `[h, z_posterior]` and
            skip-free embed recon in both `recon_embed` and `recon_bottleneck`.
        """
        if obs_u8.ndim != 5:
            raise ValueError(f"expected obs [B,T,H,W,C], got {tuple(obs_u8.shape)}")
        batch, time = obs_u8.shape[:2]
        flat_obs = obs_u8.reshape(batch * time, *obs_u8.shape[2:])
        flat_embed = self.encoder(nhwc_uint8_to_nchw_float(flat_obs))
        embeds = flat_embed.view(batch, time, -1)
        if actions_onehot:
            act = actions.float()
        else:
            act = one_hot_action(actions, self.rssm.action_dim)
        rssm_out = self.rssm.observe(embeds, act)
        feat = rssm_features(rssm_out.h, rssm_out.z_posterior)
        _, _, feat_dim = feat.shape
        flat_feat = feat.reshape(batch * time, feat_dim)
        recon = self.decoder(flat_feat).view(batch, time, 3, 64, 64)
        recon_from_embed = self.embed_decoder(flat_embed).view(batch, time, 3, 64, 64)
        reward_pred = self.reward_head(flat_feat).view(batch, time, 1)
        cont_logit = self.continue_head(flat_feat).view(batch, time, 1)
        return WorldModelOutput(
            embeds=embeds,
            rssm=rssm_out,
            feat=feat,
            recon=recon,
            recon_embed=recon_from_embed,
            recon_bottleneck=recon_from_embed,
            reward_pred=reward_pred,
            cont_logit=cont_logit,
        )
