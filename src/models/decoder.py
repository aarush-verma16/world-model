"""CNN decoder: reconstructs Crafter frames from an embedding.

Role in the Dreamer loop (M1): reconstruction target for the encoder. Later the
decoder will condition on `[h, z]` from the RSSM; for M1 it only sees the
encoder embedding (no recurrence yet).

Uses nearest-neighbor upsample + conv (not ConvTranspose2d) to avoid the
blurry/checkerboard look that hurts pixel-art reconstructions.
"""

from __future__ import annotations

from torch import Tensor, nn


class Decoder(nn.Module):
    """Embedding → image: `[B, embed_dim]` → `[B, 3, 64, 64]` in `[-1, 1]`.

    Path: embed → 4x4 map, then four (nearest×2 + Conv3x3) stages to 64x64.
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
            self.fc = nn.Linear(embed_dim, flat_dim)

        stages: list[nn.Module] = []
        prev = channels[0]
        for ch in channels[1:]:
            stages.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(prev, ch, kernel_size=3, padding=1),
                    nn.SiLU(),
                )
            )
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
