"""WorldModel: skip-free encoder + RSSM + decoder + reward/continue heads.

Trains on real replay sequences (M3). Imagination / actor-critic is M4.

When the encoder flatten is a 4x4 map (Identity `embed_dim == C*4*4`), there
is **one** skip-free decoder upsample (sub-pixel conv — see `decoder`). Embed
recon reshapes that map and paints. `[h,z]` recon predicts the same 4x4 layout
(`HzToMap`) then uses the same upsample with decoder weights detached, so
`[h,z]` has to reproduce a map the encoder path already renders well instead
of bending the renderer toward blobs.

`z` is a 32×32 categorical (unstructured); per-cell `z` (2 cats × 16 cells)
held KL raw at 1.2–1.6 above `free_nats=1`. A separate HUD head that pasted
over rows 49–63 hid the inventory; that head is gone. `recon_blob` (tile-mean
L1) is off — it is a solid-color-per-tile objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.decoder import Decoder
from models.encoder import Encoder, ResidualBlock
from models.heads import ContinueHead, RewardHead, rssm_features
from models.preprocess import nhwc_uint8_to_nchw_float
from models.rssm import RSSM, RSSMOutput, one_hot_action


class HzToMap(nn.Module):
    """`h` + `z_posterior` → 4x4 feature map (encoder layout).

    `h` gets its own **spatial** projection. It used to be
    `h_proj(h).unsqueeze(-1).unsqueeze(-1)`: a `[B, C, 1, 1]` per-channel bias,
    identical in all 16 cells. That left `z_proj` as the only source of layout,
    and `z` is 32 categoricals x 32 classes = 160 bits per frame — nowhere near
    enough to place 63 Crafter tiles plus a 9-slot inventory, so `[h,z]` guessed
    where things were while embed recon looked fine. Crafter layout is mostly
    *persistent* (grass/water/trees do not move) which is exactly what `h`
    accumulates over the sequence; it simply had no spatial slot to write it to.
    """

    def __init__(
        self,
        deter_dim: int,
        stoch: int,
        classes: int,
        channels: int,
        spatial: int = 4,
        h_channels: int = 128,
        blocks: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.spatial = spatial
        self.h_channels = h_channels
        cells = spatial * spatial
        self.h_proj = nn.Linear(deter_dim, h_channels * cells)
        self.z_proj = nn.Linear(stoch * classes, channels * cells)
        self.mix = nn.Conv2d(channels + h_channels, channels, kernel_size=3, padding=1)
        refine: list[nn.Module] = [nn.SiLU()]
        refine.extend(ResidualBlock(channels) for _ in range(blocks))
        self.refine = nn.Sequential(*refine)

    def forward(self, h: Tensor, z_flat: Tensor) -> Tensor:
        """`h` `[B, deter]`, `z_flat` `[B, stoch*classes]` → `[B, C, S, S]`."""
        z_map = self.z_proj(z_flat).view(-1, self.channels, self.spatial, self.spatial)
        h_map = self.h_proj(h).view(-1, self.h_channels, self.spatial, self.spatial)
        return self.refine(self.mix(torch.cat([z_map, h_map], dim=1)))


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
    hz_map: Tensor | None = None
    embed_map: Tensor | None = None


class WorldModel(nn.Module):
    """End-to-end world model used for M3 supervised training on replay.

    With Identity encoder flatten matching `decoder.channels[0]`, `decoder`
    and `embed_decoder` are the same module. `HzToMap` is the only extra
    `[h,z]` path (imagination still has no pixels to skip from).
    """

    def __init__(
        self,
        encoder: Encoder,
        rssm: RSSM,
        decoder: Decoder,
        embed_decoder: Decoder,
        reward_head: RewardHead,
        continue_head: ContinueHead,
        hz_to_map: HzToMap | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.decoder = decoder
        self.embed_decoder = embed_decoder
        self.embed_decoder_bottleneck = embed_decoder
        self.hz_to_map = hz_to_map
        self.reward_head = reward_head
        self.continue_head = continue_head
        feat_dim = rssm.deter_dim + rssm.z_flat_dim
        if encoder.embed_dim != rssm.embed_dim:
            raise ValueError(
                f"encoder.embed_dim={encoder.embed_dim} != rssm.embed_dim={rssm.embed_dim}"
            )
        if embed_decoder.embed_dim != rssm.embed_dim:
            raise ValueError(
                f"embed_decoder.embed_dim={embed_decoder.embed_dim} != "
                f"rssm.embed_dim={rssm.embed_dim}"
            )
        if hz_to_map is None and decoder.embed_dim != feat_dim:
            raise ValueError(
                f"decoder.embed_dim={decoder.embed_dim} != feat_dim={feat_dim} "
                f"(deter {rssm.deter_dim} + z_flat {rssm.z_flat_dim})"
            )
        if hz_to_map is not None:
            if decoder is not embed_decoder:
                raise ValueError("shared 4x4 decoder requires decoder is embed_decoder")
            if hz_to_map.channels != decoder.channels0:
                raise ValueError("HzToMap channels must match decoder.channels[0]")
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
        prior_layers: int = 2,
        decoder_channels: tuple[int, ...] = (512, 256, 128, 64),
        head_hidden: int = 512,
        head_layers: int = 2,
        stem_channels: int = 64,
        spatial: int = 4,
        encoder_blocks: int = 2,
        decoder_blocks: int = 0,
    ) -> WorldModel:
        """Construct a consistently-sized world model from scalar dims."""
        del stem_channels, spatial
        encoder = Encoder(
            embed_dim=embed_dim,
            channels=encoder_channels,
            blocks=encoder_blocks,
        )
        embed_spatial = (
            encoder_channels[-1]
            if embed_dim == encoder_channels[-1] * 4 * 4
            else None
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
            embed_spatial=embed_spatial,
            prior_layers=prior_layers,
        )
        feat_dim = deter_dim + stoch * classes
        shared = (
            embed_spatial is not None and decoder_channels[0] == embed_spatial
        )
        if shared:
            decoder = Decoder(
                embed_dim=embed_dim,
                channels=decoder_channels,
                start_res=4,
                blocks=decoder_blocks,
            )
            embed_decoder = decoder
            hz_to_map: HzToMap | None = HzToMap(
                deter_dim, stoch, classes, embed_spatial
            )
        else:
            decoder = Decoder(
                embed_dim=feat_dim,
                channels=decoder_channels,
                start_res=4,
                blocks=decoder_blocks,
            )
            embed_decoder = Decoder(
                embed_dim=embed_dim,
                channels=decoder_channels,
                start_res=4,
                blocks=decoder_blocks,
            )
            hz_to_map = None
        reward_head = RewardHead(feat_dim, hidden=head_hidden, layers=head_layers)
        continue_head = ContinueHead(feat_dim, hidden=head_hidden, layers=head_layers)
        return cls(
            encoder, rssm, decoder, embed_decoder, reward_head, continue_head, hz_to_map
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
        hz_map = None
        embed_map = None
        if self.hz_to_map is not None:
            h_flat = rssm_out.h.reshape(batch * time, -1)
            z_flat = rssm_out.z_posterior.reshape(batch * time, -1)
            hz_map = self.hz_to_map(h_flat, z_flat)
            spatial = self.hz_to_map.spatial
            embed_map = flat_embed.view(
                batch * time, self.hz_to_map.channels, spatial, spatial
            )
            recon = self.decoder.from_map(hz_map, detach_weights=True).view(
                batch, time, 3, 64, 64
            )
            recon_from_embed = self.decoder.from_map(embed_map).view(
                batch, time, 3, 64, 64
            )
        else:
            hz_feat = self.decoder.fc(flat_feat)
            hz_feat = hz_feat.view(
                -1, self.decoder.channels0, self.decoder.start_res, self.decoder.start_res
            )
            recon = self.decoder.from_map(hz_feat).view(batch, time, 3, 64, 64)
            emb_feat = self.embed_decoder.fc(flat_embed)
            emb_feat = emb_feat.view(
                -1,
                self.embed_decoder.channels0,
                self.embed_decoder.start_res,
                self.embed_decoder.start_res,
            )
            recon_from_embed = self.embed_decoder.from_map(emb_feat).view(
                batch, time, 3, 64, 64
            )
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
            hz_map=hz_map,
            embed_map=embed_map,
        )
