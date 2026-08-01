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
