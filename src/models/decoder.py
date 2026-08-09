"""CNN decoder: reconstructs Crafter frames from an embedding / RSSM feature.

Used by the M1 skip-free encode→decode path and by the M3 world model
(decoding from `concat(h, flatten(z))`). Nearest upsample + conv avoids
ConvTranspose checkerboard artifacts on pixel art.
"""

from __future__ import annotations

from torch import Tensor, nn


def _upsample_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """×2 nearest upsample then two 3×3 convs (extra capacity vs single conv)."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.SiLU(),
    )


class Decoder(nn.Module):
    """Embedding → image: `[B, embed_dim]` → `[B, 3, 64, 64]` in `[-1, 1]`.

    Path: embed → 4×4 map, then four (nearest×2 + dual-conv) stages to 64×64.
    """

    def __init__(
        self,
        embed_dim: int = 8192,
        out_channels: int = 3,
        channels: tuple[int, ...] = (512, 256, 128, 64),
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"expected 4 upsample channel sizes, got {channels}")

        flat_dim = channels[0] * 4 * 4
        self.channels0 = channels[0]
        self.channels = tuple(channels)
        self.fc: nn.Module
        if embed_dim == flat_dim:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Sequential(
                nn.Linear(embed_dim, flat_dim),
                nn.LayerNorm(flat_dim),
                nn.SiLU(),
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
        x = x.view(-1, self.channels0, 4, 4)
        out = self.up(x)
        if out.shape[1:] != (3, 64, 64):
            raise RuntimeError(f"decoder produced unexpected shape {tuple(out.shape)}")
        return out
