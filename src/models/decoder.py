"""CNN decoder: reconstructs Crafter frames from an embedding / RSSM feature.

Used by the M1 skip-free encode→decode path and by the M3 world model
(decoding from `concat(h, flatten(z))`). Nearest upsample + conv avoids
ConvTranspose checkerboard artifacts on pixel art.

Residual 3x3 blocks live on the *encoder* (DreamerV3 ImageEncoderResnet).
Putting the same blocks on both `[h,z]` and embed decoders doubled the
activation footprint (smoke: 19 GiB / 0.17 steps/s). Blob-level sprites
have to be *in the embedding* first; a skinny decoder can paint them.
"""

from __future__ import annotations

from torch import Tensor, nn


def _upsample_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """×2 nearest upsample then two 3×3 convs."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.SiLU(),
    )


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
                "decoder residual blocks are disabled (VRAM: two XL decoders). "
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
        out = self.up(x)
        if out.shape[1:] != (3, 64, 64):
            raise RuntimeError(f"decoder produced unexpected shape {tuple(out.shape)}")
        return out
