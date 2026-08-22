"""CNN decoder: reconstructs Crafter frames from an embedding / RSSM feature.

Used by the M1 skip-free encode→decode path and by the M3 world model
(shared 4x4 upsample from the encoder map / `HzToMap`). Nearest upsample
+ conv avoids ConvTranspose checkerboard artifacts on pixel art.

Residual 3x3 blocks live on the *encoder* (DreamerV3 ImageEncoderResnet).
Putting the same blocks on both `[h,z]` and embed decoders doubled the
activation footprint (smoke: 19 GiB / 0.17 steps/s). Blob-level sprites
have to be *in the embedding* first; a skinny decoder can paint them.
"""

from __future__ import annotations

from models.crafter_layout import composite_hud
from torch import Tensor, nn
import torch.nn.functional as F


def _upsample_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """×2 nearest upsample then two 3×3 convs."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.SiLU(),
    )


def _apply_detached(module: nn.Module, x: Tensor) -> Tensor:
    """Run `module` with frozen weights so grads flow to `x` only."""
    if isinstance(module, nn.Sequential):
        for child in module:
            x = _apply_detached(child, x)
        return x
    if isinstance(module, nn.Conv2d):
        bias = None if module.bias is None else module.bias.detach()
        return F.conv2d(
            x,
            module.weight.detach(),
            bias,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
    if isinstance(module, (nn.Upsample, nn.SiLU, nn.Tanh, nn.Identity)):
        return module(x)
    raise TypeError(f"detach decode unsupported for {type(module)}")


class Decoder(nn.Module):
    """Embedding → image: `[B, embed_dim]` → `[B, 3, 64, 64]` in `[-1, 1]`.

    Path: embed → `start_res`×`start_res` map, then nearest×2 stages up to 64×64.
    """

    def __init__(
        self,
        embed_dim: int = 8192,
        out_channels: int = 3,
        channels: tuple[int, ...] = (512, 256, 128, 64),
        start_res: int = 4,
        blocks: int = 0,
    ) -> None:
        super().__init__()
        if start_res not in (4, 8):
            raise ValueError(f"start_res must be 4 or 8, got {start_res}")
        if blocks != 0:
            raise ValueError(
                "decoder residual blocks are disabled (VRAM). "
                f"got blocks={blocks}"
            )
        # 4→8→16→32→64 needs 4 upsamples; 8→16→32→64 needs 3.
        n_up = 4 if start_res == 4 else 3
        if len(channels) != n_up:
            raise ValueError(
                f"with start_res={start_res} expected {n_up} channel sizes, got {channels}"
            )

        flat_dim = channels[0] * start_res * start_res
        self.channels0 = channels[0]
        self.start_res = start_res
        self.channels = tuple(channels)
        self.blocks = 0
        self.fc: nn.Module = (
            nn.Identity() if embed_dim == flat_dim else nn.Linear(embed_dim, flat_dim)
        )

        stages: list[nn.Module] = []
        prev = channels[0]
        for ch in channels[1:]:
            stages.append(_upsample_block(prev, ch))
            prev = ch
        stages.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(prev, out_channels, kernel_size=3, padding=1),
                nn.Tanh(),
            )
        )
        self.up = nn.Sequential(*stages)
        self.embed_dim = embed_dim
        self.flat_dim = flat_dim

        # Crafter HUD is a 2×9 grid of 7px slots (not a 64×64 image). Decode
        # the bottom 4×4 row onto that grid so inventory digits/icons can
        # change without an 8px world decoder or U-Net skip-to-RGB.
        hud_ch = min(128, channels[0])
        self.hud_reduce = nn.Conv2d(channels[0], hud_ch, kernel_size=1)
        self.hud_up = nn.Sequential(
            nn.Conv2d(hud_ch, hud_ch, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=7, mode="nearest"),
            nn.Conv2d(hud_ch, hud_ch, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hud_ch, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def from_map(self, feat_map: Tensor, *, detach_weights: bool = False) -> Tensor:
        """Decode a `start_res` feature map: `[B, C, S, S]` → `[B, 3, 64, 64]`.

        `detach_weights=True` keeps grads on `feat_map` (for `HzToMap`) but
        does not train this upsample — the shared decoder is only fit on the
        embed path so `[h,z]` pixel loss cannot scramble "map → pixels".
        """
        expected = (self.channels0, self.start_res, self.start_res)
        if feat_map.ndim != 4 or feat_map.shape[1:] != expected:
            raise ValueError(
                f"expected feat map [B, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(feat_map.shape)}"
            )
        out = _apply_detached(self.up, feat_map) if detach_weights else self.up(feat_map)
        if out.shape[1:] != (3, 64, 64):
            raise RuntimeError(f"decoder produced unexpected shape {tuple(out.shape)}")
        return out

    def hud_from_map(self, feat_map: Tensor, *, detach_weights: bool = False) -> Tensor:
        """Bottom 4×4 row → Crafter inventory strip `[B, 3, 14, 63]`."""
        expected = (self.channels0, self.start_res, self.start_res)
        if feat_map.ndim != 4 or feat_map.shape[1:] != expected:
            raise ValueError(
                f"expected feat map [B, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(feat_map.shape)}"
            )
        bottom = feat_map[:, :, -1:, :]
        if detach_weights:
            x = _apply_detached(self.hud_reduce, bottom)
            x = F.interpolate(x, size=(2, 9), mode="bilinear", align_corners=False)
            x = _apply_detached(self.hud_up, x)
        else:
            x = self.hud_reduce(bottom)
            x = F.interpolate(x, size=(2, 9), mode="bilinear", align_corners=False)
            x = self.hud_up(x)
        if x.shape[1:] != (3, 14, 63):
            raise RuntimeError(f"HUD decoder produced unexpected shape {tuple(x.shape)}")
        return x

    def from_map_with_hud(
        self, feat_map: Tensor, *, detach_weights: bool = False
    ) -> Tensor:
        """World upsample plus HUD strip composited onto the inventory rows."""
        world = self.from_map(feat_map, detach_weights=detach_weights)
        hud = self.hud_from_map(feat_map, detach_weights=detach_weights)
        return composite_hud(world, hud)

    def forward(self, embed: Tensor) -> Tensor:
        """Decode embeddings to images.

        Args:
            embed: `[B, embed_dim]`.

        Returns:
            float images `[B, 3, 64, 64]` in `[-1, 1]`.
        """
        if embed.ndim != 2 or embed.shape[1] != self.embed_dim:
            raise ValueError(
                f"expected embed shape [B, {self.embed_dim}], got {tuple(embed.shape)}"
            )
        x = self.fc(embed)
        x = x.view(-1, self.channels0, self.start_res, self.start_res)
        return self.from_map(x)
