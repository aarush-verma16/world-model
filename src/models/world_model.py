"""WorldModel: encoder + RSSM + decoder + reward/continue heads.

Trains on real replay sequences (M3). Imagination / actor-critic is M4.

Keeps THREE decode heads, each answering a different question:
  - `decoder` ([h,z] -> pixels): skip-free, since imagination has no real
    frame to skip from. This is the one that actually matters long-term.
  - `perception.decode` (embed + U-Net skips -> pixels): tests whether the
    encoder CNN *can* extract fine detail at all (M1-proven mechanism).
  - `embed_decoder_bottleneck` (embed alone, no skips -> pixels): tests
    whether the bottleneck embedding *itself* -- the only thing the RSSM
    ever sees -- actually carries that detail.

That third head exists because of a failure mode the other two hid: once
skips were added, `perception.decode`'s loss converged to near-zero almost
immediately, which looked like the encoder had been "fixed". But a
probe-decoder test (`notebooks/06_decoder_probe.ipynb`) showed a skip-free
decode of that exact same frozen embedding was just as blind to
rocks/water/sand as `[h,z]` (~0.17 L1 either way, vs ~0.004 with skips) --
the skips did nearly all the work, so the embedding was never actually
forced to encode that content. Since the RSSM only ever receives the
embedding (never the skips), that low `recon_embed` loss was not evidence
of a richer signal reaching the RSSM. `embed_decoder_bottleneck` closes that
gap: the only way to lower its loss is for `embed` itself to carry the
content, which is what can actually help `[h,z]`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.autoencoder import PerceptionAutoencoder
from models.decoder import Decoder
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

    Primary decoder conditions on `feat = concat(h, flatten(z_posterior))`
    and is skip-free (must work purely from imagined state). `perception`
    (encoder + U-Net skip decoder, M1's `PerceptionAutoencoder`) reconstructs
    `recon_embed` straight from a real frame's own embedding + that same
    frame's skips -- legitimate since it never runs during imagination.
    `embed_decoder_bottleneck` decodes that same embedding with NO skips, to
    directly supervise the bottleneck vector itself (see module docstring).
    """

    def __init__(
        self,
        perception: PerceptionAutoencoder,
        rssm: RSSM,
        decoder: Decoder,
        embed_decoder_bottleneck: Decoder,
        reward_head: RewardHead,
        continue_head: ContinueHead,
    ) -> None:
        super().__init__()
        self.perception = perception
        self.rssm = rssm
        self.decoder = decoder
        self.embed_decoder_bottleneck = embed_decoder_bottleneck
        self.reward_head = reward_head
        self.continue_head = continue_head
        feat_dim = rssm.deter_dim + rssm.z_flat_dim
        if decoder.embed_dim != feat_dim:
            raise ValueError(
                f"decoder.embed_dim={decoder.embed_dim} != feat_dim={feat_dim} "
                f"(deter {rssm.deter_dim} + z_flat {rssm.z_flat_dim})"
            )
        if perception.embed_dim != rssm.embed_dim:
            raise ValueError(
                f"perception.embed_dim={perception.embed_dim} != "
                f"rssm.embed_dim={rssm.embed_dim}"
            )
        if embed_decoder_bottleneck.embed_dim != rssm.embed_dim:
            raise ValueError(
                f"embed_decoder_bottleneck.embed_dim={embed_decoder_bottleneck.embed_dim} "
                f"!= rssm.embed_dim={rssm.embed_dim}"
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
    ) -> WorldModel:
        """Construct a consistently-sized world model from scalar dims."""
        perception = PerceptionAutoencoder(
            embed_dim=embed_dim, channels=encoder_channels, stem_channels=stem_channels
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
        )
        feat_dim = deter_dim + stoch * classes
        decoder = Decoder(embed_dim=feat_dim, channels=decoder_channels)
        embed_decoder_bottleneck = Decoder(embed_dim=embed_dim, channels=decoder_channels)
        reward_head = RewardHead(feat_dim, hidden=head_hidden, layers=head_layers)
        continue_head = ContinueHead(feat_dim, hidden=head_hidden, layers=head_layers)
        return cls(
            perception, rssm, decoder, embed_decoder_bottleneck, reward_head, continue_head
        )

    def encode(self, obs_u8: Tensor) -> Tensor:
        """uint8 obs `[B, T, H, W, C]` or `[B, H, W, C]` → embeds with time dim.

        Skips (needed for `recon_embed`) are discarded here; use `forward`
        when you also need the reconstruction.
        """
        squeeze = obs_u8.ndim == 4
        if squeeze:
            obs_u8 = obs_u8.unsqueeze(1)
        if obs_u8.ndim != 5:
            raise ValueError(f"expected obs [B,T,H,W,C] or [B,H,W,C], got {tuple(obs_u8.shape)}")
        batch, time = obs_u8.shape[:2]
        flat = obs_u8.reshape(batch * time, *obs_u8.shape[2:])
        embeds, _skips = self.perception.encode(nhwc_uint8_to_nchw_float(flat))
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
              - recon `[B, T, 3, 64, 64]` from `[h, z_posterior]`
              - recon_embed `[B, T, 3, 64, 64]` from embedding + U-Net skips
              - recon_bottleneck `[B, T, 3, 64, 64]` from embedding alone,
                no skips -- the term that actually supervises what the RSSM
                receives (see module docstring)
              - reward_pred / cont_logit `[B, T, 1]`
        """
        if obs_u8.ndim != 5:
            raise ValueError(f"expected obs [B,T,H,W,C], got {tuple(obs_u8.shape)}")
        batch, time = obs_u8.shape[:2]
        flat_obs = obs_u8.reshape(batch * time, *obs_u8.shape[2:])
        flat_embed, skips = self.perception.encode(nhwc_uint8_to_nchw_float(flat_obs))
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
        recon_embed = self.perception.decode(flat_embed, skips).view(batch, time, 3, 64, 64)
        recon_bottleneck = self.embed_decoder_bottleneck(flat_embed).view(batch, time, 3, 64, 64)
        reward_pred = self.reward_head(flat_feat).view(batch, time, 1)
        cont_logit = self.continue_head(flat_feat).view(batch, time, 1)
        return WorldModelOutput(
            embeds=embeds,
            rssm=rssm_out,
            feat=feat,
            recon=recon,
            recon_embed=recon_embed,
            recon_bottleneck=recon_bottleneck,
            reward_pred=reward_pred,
            cont_logit=cont_logit,
        )
