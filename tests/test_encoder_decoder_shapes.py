"""Shape / smoke tests for M1 encoder, decoder, and perception autoencoder."""

from __future__ import annotations

import torch

from models.autoencoder import PerceptionAutoencoder
from models.decoder import Decoder
from models.encoder import Encoder


def test_encoder_decoder_roundtrip_shapes() -> None:
    enc = Encoder(embed_dim=256)
    dec = Decoder(embed_dim=256)
    obs = torch.randn(4, 3, 64, 64)
    embed = enc(obs)
    recon = dec(embed)
    assert embed.shape == (4, 256)
    assert recon.shape == (4, 3, 64, 64)


def test_hud_from_map_shape_and_composite() -> None:
    from models.crafter_layout import HUD_H, HUD_W, HUD_TOP, composite_hud

    dec = Decoder(embed_dim=64 * 4 * 4, channels=(64, 32, 16, 8))
    feat = torch.randn(2, 64, 4, 4)
    hud = dec.hud_from_map(feat)
    assert hud.shape == (2, 3, HUD_H, HUD_W)
    world = dec.from_map(feat)
    out = composite_hud(world, hud)
    assert out.shape == (2, 3, 64, 64)
    assert torch.allclose(out[:, :, HUD_TOP : HUD_TOP + HUD_H, :HUD_W], hud)
    out.sum().backward()
    assert dec.hud_reduce.weight.grad is not None


def test_encoder_decoder_backward() -> None:
    enc = Encoder(embed_dim=256)
    dec = Decoder(embed_dim=256)
    obs = torch.randn(2, 3, 64, 64)
    recon = dec(enc(obs))
    loss = torch.nn.functional.mse_loss(recon, obs)
    loss.backward()
    assert enc.conv[0].weight.grad is not None


def test_perception_autoencoder_shapes_and_backward() -> None:
    model = PerceptionAutoencoder(embed_dim=8192, channels=(64, 128, 256, 512))
    obs = torch.randn(2, 3, 64, 64).clamp(-1, 1)
    recon, embed = model(obs)
    assert recon.shape == (2, 3, 64, 64)
    assert embed.shape == (2, 8192)
    assert torch.isfinite(recon).all()
    loss = torch.nn.functional.l1_loss(recon, obs)
    loss.backward()
    assert model.stem[0].weight.grad is not None
    assert torch.isfinite(model.stem[0].weight.grad).all()
