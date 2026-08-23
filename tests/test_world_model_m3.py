"""Tests for M3 world-model heads, KL balancing, and sequential replay."""

from __future__ import annotations

import torch

from models.heads import ContinueHead, RewardHead, rssm_features
from models.preprocess import nhwc_uint8_to_nchw_float
from models.encoder import Encoder
from models.world_model import WorldModel
from training.losses import (
    blob_recon_loss,
    categorical_kl,
    content_weight_map,
    gradient_l1_loss,
    kl_balance,
    weighted_pixel_loss,
    world_model_loss,
)
from training.replay_buffer import ReplayBuffer


def _tiny_wm() -> WorldModel:
    return WorldModel.from_config_dims(
        embed_dim=64,
        encoder_channels=(16, 32, 64, 64),
        action_dim=5,
        deter_dim=32,
        stoch=4,
        classes=4,
        hidden=32,
        decoder_channels=(64, 32, 16, 8),
        head_hidden=32,
        head_layers=1,
    )


def test_rssm_features_shape() -> None:
    h = torch.randn(2, 8, 32)
    z = torch.zeros(2, 8, 4, 4)
    z[..., 0] = 1.0
    feat = rssm_features(h, z)
    assert feat.shape == (2, 8, 32 + 16)


def test_heads_forward_and_grad() -> None:
    feat = torch.randn(3, 48, requires_grad=True)
    reward = RewardHead(48, hidden=16, layers=1)
    cont = ContinueHead(48, hidden=16, layers=1)
    r = reward(feat)
    c = cont(feat)
    assert r.shape == (3, 1)
    assert c.shape == (3, 1)
    (r.sum() + c.sum()).backward()
    assert feat.grad is not None
    assert torch.isfinite(feat.grad).all()


def test_kl_balance_asymmetric_and_finite() -> None:
    post = torch.randn(2, 4, 3, 5, requires_grad=True)
    prior = torch.randn(2, 4, 3, 5, requires_grad=True)
    kl, dyn, rep, dyn_raw, rep_raw = kl_balance(
        post, prior, unimix=0.01, dyn_scale=0.5, rep_scale=0.1, free_nats=1.0
    )
    assert torch.isfinite(kl)
    assert dyn.item() >= 1.0 - 1e-5
    assert dyn_raw.item() <= dyn.item() + 1e-5
    assert rep_raw.item() <= rep.item() + 1e-5
    kl.backward()
    assert prior.grad is not None and post.grad is not None
    assert prior.grad.abs().sum() > 0
    assert post.grad.abs().sum() > 0


def test_categorical_kl_zero_when_identical() -> None:
    logits = torch.randn(2, 3, 4, 5)
    kl = categorical_kl(logits, logits.clone(), unimix=0.0)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)


def test_blob_recon_loss_zero_when_identical() -> None:
    img = torch.rand(2, 3, 64, 64)
    assert torch.allclose(blob_recon_loss(img, img), torch.zeros(()), atol=1e-6)
    seq = torch.rand(2, 4, 3, 64, 64)
    assert torch.allclose(blob_recon_loss(seq, seq), torch.zeros(()), atol=1e-6)


def test_blob_recon_loss_penalizes_erased_8px_sprite() -> None:
    """8x8 pooled L1 must care about a tile-sized sprite that 16x16 median grass erases."""
    grass, cow = 0.2, 0.9
    target = torch.full((1, 3, 64, 64), grass)
    target[:, :, 16:24, 16:24] = cow
    median_16 = torch.full((1, 3, 64, 64), grass)
    blob_8 = median_16.clone()
    blob_8[:, :, 16:24, 16:24] = cow
    assert float(blob_recon_loss(median_16, target)) > 0.01
    assert float(blob_recon_loss(blob_8, target)) < 1e-6
    assert float(blob_recon_loss(blob_8, target)) < float(
        blob_recon_loss(median_16, target)
    )


def test_blob_recon_loss_backward() -> None:
    pred = torch.randn(2, 3, 64, 64, requires_grad=True)
    target = torch.randn(2, 3, 64, 64)
    loss = blob_recon_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_tile_blob_loss_upweights_object_on_grass() -> None:
    from training.losses import tile_blob_loss

    grass = torch.full((1, 3, 64, 64), 0.2)
    target = grass.clone()
    # One Crafter tile at (row 2, col 3) of the 7×9 local view.
    target[:, :, 14:21, 21:28] = 0.9
    erased = grass.clone()
    blob = tile_blob_loss(erased, target)
    ok = tile_blob_loss(target, target)
    assert float(ok) < 1e-6
    assert float(blob) > float(ok) + 0.01


def test_avatar_and_hud_losses_are_localized() -> None:
    from models.crafter_layout import AVATAR_H, AVATAR_W, HUD_H, HUD_W
    from training.losses import avatar_recon_loss, hud_recon_loss

    target = torch.zeros(1, 3, 64, 64)
    pred = torch.zeros(1, 3, 64, 64)
    pred[:, :, 14:35, 21:42] = 1.0
    av = avatar_recon_loss(pred, target)
    assert av.shape == ()
    assert float(av) > 0.0
    pred2 = torch.zeros(1, 3, 64, 64)
    pred2[:, :, 49:63, :63] = 1.0
    hud = hud_recon_loss(pred2, target)
    assert float(hud) > 0.0
    assert AVATAR_H == 21 and AVATAR_W == 21
    assert HUD_H == 14 and HUD_W == 63


def test_gradient_l1_loss_zero_when_identical_and_positive_when_shifted() -> None:
    img = torch.rand(2, 3, 3, 8, 8)
    assert torch.allclose(gradient_l1_loss(img, img), torch.zeros(()), atol=1e-6)

    flat = torch.zeros(2, 3, 3, 8, 8)
    edgy = flat.clone()
    edgy[..., 4:] = 1.0  # a hard edge the flat target doesn't have
    loss = gradient_l1_loss(edgy, flat)
    assert loss.item() > 0.0


def test_gradient_l1_loss_backward() -> None:
    pred = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randn(2, 3, 8, 8)
    loss = gradient_l1_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_content_weight_map_boosts_edges_over_flat_regions() -> None:
    target = torch.zeros(2, 3, 8, 8)
    target[..., 4:] = 1.0  # a hard edge down the middle
    weight = content_weight_map(target, edge_weight=8.0)
    assert weight.shape == (2, 1, 8, 8)
    # flat interior (away from the edge) should sit near the base weight of 1.0
    assert weight[..., 0, 0].allclose(torch.ones(2, 1), atol=1e-4)
    # the column right at the edge should be boosted well above 1.0
    assert (weight[..., :, 3] > 2.0).all()


def test_content_weight_map_disabled_is_uniform_one() -> None:
    target = torch.rand(2, 3, 8, 8)
    weight = content_weight_map(target, edge_weight=0.0)
    assert torch.allclose(weight, torch.ones_like(weight))


def test_weighted_pixel_loss_matches_plain_when_edge_weight_zero() -> None:
    pred = torch.rand(2, 3, 8, 8)
    target = torch.rand(2, 3, 8, 8)
    plain = torch.nn.functional.l1_loss(pred, target)
    weighted = weighted_pixel_loss(pred, target, "l1", edge_weight=0.0)
    assert torch.allclose(plain, weighted)


def test_weighted_pixel_loss_backward() -> None:
    pred = torch.rand(2, 3, 8, 8, requires_grad=True)
    target = torch.rand(2, 3, 8, 8)
    loss = weighted_pixel_loss(pred, target, "l1", edge_weight=5.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_world_model_forward_shapes_and_loss_backward() -> None:
    wm = _tiny_wm()
    b, t = 2, 6
    obs = torch.randint(0, 256, (b, t, 64, 64, 3), dtype=torch.uint8)
    actions = torch.randint(0, 5, (b, t), dtype=torch.int64)
    rewards = torch.randn(b, t)
    cont = torch.ones(b, t)

    out = wm(obs, actions)
    assert out.recon.shape == (b, t, 3, 64, 64)
    assert out.recon_embed.shape == (b, t, 3, 64, 64)
    assert out.recon_bottleneck.shape == (b, t, 3, 64, 64)
    assert out.reward_pred.shape == (b, t, 1)
    assert out.cont_logit.shape == (b, t, 1)
    assert out.rssm.z_prior.shape == (b, t, 4, 4)
    assert out.rssm.z_posterior.shape == (b, t, 4, 4)

    obs_f = nhwc_uint8_to_nchw_float(obs.reshape(b * t, 64, 64, 3)).view(b, t, 3, 64, 64)
    loss = world_model_loss(
        obs=obs_f,
        recon=out.recon,
        recon_embed=out.recon_embed,
        recon_bottleneck=out.recon_bottleneck,
        reward=rewards,
        reward_pred=out.reward_pred,
        cont=cont,
        cont_logit=out.cont_logit,
        post_logits=out.rssm.posterior_logits,
        prior_logits=out.rssm.prior_logits,
        unimix=wm.rssm.unimix,
        grad_scale=2.0,
        recon_loss_type="l1",
        edge_weight=4.0,
    )
    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.grad)
    assert torch.isfinite(loss.recon_l1)
    assert torch.isfinite(loss.recon_blob)
    assert torch.isfinite(loss.recon_embed_blob)
    assert torch.isfinite(loss.recon_avatar)
    assert torch.isfinite(loss.recon_hud)
    # Content-weighting can only raise (or match) the unweighted pixel term.
    assert float(loss.recon.detach()) >= float(loss.recon_l1.detach()) - 1e-5
    assert float(loss.recon_bottleneck.detach()) >= float(loss.recon_bottleneck_l1.detach()) - 1e-5
    loss.total.backward()
    assert wm.encoder.conv[0].weight.grad is not None
    assert wm.rssm.prior_net[0].weight.grad is not None
    assert wm.reward_head.net[0].weight.grad is not None
    assert wm.embed_decoder.fc.weight.grad is not None or isinstance(
        wm.embed_decoder.fc, torch.nn.Identity
    )


def test_embed_recon_trains_encoder() -> None:
    wm = _tiny_wm()
    b, t = 2, 4
    obs = torch.randint(0, 256, (b, t, 64, 64, 3), dtype=torch.uint8)
    actions = torch.randint(0, 5, (b, t), dtype=torch.int64)
    out = wm(obs, actions)
    out.recon_embed.abs().mean().backward()
    assert wm.encoder.conv[0].weight.grad is not None
    assert float(wm.encoder.conv[0].weight.grad.abs().sum()) > 0.0
    assert out.recon_embed is out.recon_bottleneck
    assert not hasattr(wm, "perception")


def test_m3_yaml_is_skip_free_identity_flatten() -> None:
    """M3 config must not reintroduce U-Net skips, 8×8 cells, or a mixing Linear."""
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(Path("configs/m3_world_model.yaml").read_text(encoding="utf-8"))
    enc = cfg["encoder"]
    channels = tuple(int(c) for c in enc["channels"])
    assert "spatial" not in enc
    assert "stem_channels" not in enc
    assert int(enc["embed_dim"]) == channels[-1] * 4 * 4
    assert float(cfg["train"]["recon_bottleneck_scale"]) == 0.0
    assert float(cfg["train"]["recon_map_scale"]) == 1.0
    assert float(cfg["train"]["recon_blob_scale"]) == 0.0
    # Crop losses at 5.0 (alongside a pasted HUD head) flattened early recon.
    # The head is gone and these are plain L1 on a crop, so a small weight is
    # allowed — but full-frame L1 gives a pixel 5.0/4096, and anything above
    # ~1.0 here swamps that and re-flattens the rest of the frame.
    assert 0.0 <= float(cfg["train"]["recon_avatar_scale"]) <= 1.0
    assert 0.0 <= float(cfg["train"]["recon_hud_scale"]) <= 1.0
    assert float(cfg["train"]["free_nats"]) == 1.0
    assert float(cfg["train"]["edge_weight"]) == 0.0
    assert int(enc.get("blocks", 2)) == 2
    assert int(cfg["decoder"].get("blocks", 0)) == 0
    assert int(cfg["decoder"]["channels"][0]) == channels[-1]

    # Leftover spatial=8 kwargs must not switch the decoder to 8×8.
    wm = WorldModel.from_config_dims(
        embed_dim=64,
        encoder_channels=(16, 32, 64, 64),
        action_dim=5,
        deter_dim=32,
        stoch=4,
        classes=4,
        hidden=32,
        decoder_channels=(64, 32, 16, 8),
        head_hidden=32,
        head_layers=1,
        spatial=8,
        stem_channels=64,
    )
    assert isinstance(wm.encoder, Encoder)
    assert not hasattr(wm, "perception")
    assert wm.embed_decoder.start_res == 4
    assert wm.decoder.start_res == 4
    assert wm.encoder.blocks == 2
    assert wm.rssm.embed_spatial is None  # tiny embed_dim != C*4*4

    identity_wm = WorldModel.from_config_dims(
        embed_dim=64 * 4 * 4,
        encoder_channels=(16, 32, 64, 64),
        action_dim=5,
        deter_dim=32,
        stoch=4,
        classes=4,
        hidden=32,
        decoder_channels=(64, 32, 16, 8),
        head_hidden=32,
        head_layers=1,
    )
    from models.rssm import SpatialPosterior

    assert identity_wm.rssm.embed_spatial == 64
    assert isinstance(identity_wm.rssm.posterior_net, SpatialPosterior)
    assert identity_wm.embed_decoder is identity_wm.decoder
    assert identity_wm.hz_to_map is not None
    obs = torch.randint(0, 256, (2, 4, 64, 64, 3), dtype=torch.uint8)
    actions = torch.randint(0, 5, (2, 4), dtype=torch.int64)
    out = identity_wm(obs, actions)
    assert out.hz_map is not None and out.embed_map is not None
    assert out.hz_map.shape == out.embed_map.shape
    out.recon.mean().backward()
    assert identity_wm.hz_to_map.z_proj.weight.grad is not None
    # [h,z] is a 4×4 map. Live upsample weights learn 16×16 solid cells
    # (no grass/HUD texture). Freeze them; fit the renderer from embed.
    assert all(
        p.grad is None or float(p.grad.abs().sum()) == 0.0
        for p in identity_wm.decoder.up.parameters()
    )
    identity_wm.zero_grad()
    out_embed = identity_wm(obs, actions)
    out_embed.recon_embed.mean().backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in identity_wm.decoder.up.parameters()
    )


def test_hz_to_map_linear_z_when_stoch_divides_16() -> None:
    """Per-cell z is off (it parked KL above free_nats=1). Linear z → 4×4."""
    from models.rssm import SpatialPosterior

    wm = WorldModel.from_config_dims(
        embed_dim=64 * 4 * 4,
        encoder_channels=(16, 32, 64, 64),
        action_dim=5,
        deter_dim=32,
        stoch=16,
        classes=4,
        hidden=32,
        decoder_channels=(64, 32, 16, 8),
        head_hidden=32,
        head_layers=1,
    )
    assert isinstance(wm.rssm.posterior_net, SpatialPosterior)
    assert isinstance(wm.rssm.posterior_net.to_logits, torch.nn.Linear)
    assert isinstance(wm.rssm.prior_net, torch.nn.Sequential)
    assert wm.hz_to_map is not None
    assert isinstance(wm.hz_to_map.z_proj, torch.nn.Linear)
    obs = torch.randint(0, 256, (2, 4, 64, 64, 3), dtype=torch.uint8)
    actions = torch.randint(0, 5, (2, 4), dtype=torch.int64)
    out = wm(obs, actions)
    assert out.rssm.z_posterior.shape == (2, 4, 16, 4)
    out.recon.mean().backward()
    assert wm.hz_to_map.z_proj.weight.grad is not None
    assert float(wm.hz_to_map.z_proj.weight.grad.abs().sum()) > 0.0


def test_decoder_upsamples_with_subpixel_conv() -> None:
    """Nearest upsample makes sub-cell position undecodable.

    It replicates each latent cell into a 2x2 block of identical values, so the
    following 3x3 conv cannot tell where inside the cell it is. At a 4x4 latent
    a cell is 16x16 px, which is where 7px sprites and 1-2px inventory digits
    live — that is the flat-16x16-block failure, not a loss-weight problem.
    """
    from models.decoder import Decoder

    dec = Decoder(embed_dim=16 * 4 * 4, channels=(16, 8, 4, 4))
    mods = list(dec.up.modules())
    assert any(isinstance(m, torch.nn.PixelShuffle) for m in mods)
    assert not any(isinstance(m, torch.nn.Upsample) for m in mods)
    out = dec(torch.randn(2, 16 * 4 * 4))
    assert out.shape == (2, 3, 64, 64)
    # detach_weights path has to understand PixelShuffle too ([h,z] uses it).
    feat = torch.randn(2, 16, 4, 4, requires_grad=True)
    dec.from_map(feat, detach_weights=True).mean().backward()
    assert feat.grad is not None and float(feat.grad.abs().sum()) > 0.0


def test_icnr_init_starts_as_nearest_upsample() -> None:
    """ICNR seeds one filter per output channel and replicates it.

    Without it a fresh sub-pixel conv gives each output sub-position an
    independent filter, so the decoder starts checkerboarded and spends early
    steps unlearning that.
    """
    from models.decoder import icnr_

    conv = torch.nn.Conv2d(4, 3 * 4, kernel_size=3, padding=1)
    icnr_(conv.weight, 2)
    sub_filters = conv.weight.view(3, 4, 4, 3, 3)
    for sub in range(1, 4):
        assert torch.equal(sub_filters[:, 0], sub_filters[:, sub])


def test_hz_to_map_writes_h_per_cell() -> None:
    """`h` must project to a full map, not a `[B, C, 1, 1]` broadcast bias.

    Broadcast `h` leaves `z` (32 cats x 32 classes = 160 bits/frame) as the only
    source of layout, which cannot place 63 Crafter tiles plus a 9-slot
    inventory — so `[h,z]` guessed placement while embed recon looked fine.
    """
    from models.world_model import HzToMap

    torch.manual_seed(0)
    hz = HzToMap(deter_dim=32, stoch=4, classes=4, channels=8, spatial=4)
    assert hz.h_proj.out_features == hz.h_channels * hz.spatial * hz.spatial

    with torch.no_grad():
        h = torch.randn(2, 32)
        h_map = hz.h_proj(h).view(2, hz.h_channels, hz.spatial, hz.spatial)
        # Spread *across cells* within a channel is what broadcast h lacked.
        assert float(h_map.flatten(2).std(dim=-1).mean()) > 1e-3

        z = torch.zeros(2, 16)
        z[:, 0] = 1.0
        delta = (hz(h, z) - hz(torch.randn(2, 32), z)).abs().mean(dim=1)
        assert float(delta.flatten(1).std(dim=-1).mean()) > 1e-4


def test_recon_map_trains_hz_to_map() -> None:
    """L1(hz_map, embed_map.detach()) must move HzToMap."""
    wm = WorldModel.from_config_dims(
        embed_dim=64 * 4 * 4,
        encoder_channels=(16, 32, 64, 64),
        action_dim=5,
        deter_dim=32,
        stoch=4,
        classes=4,
        hidden=32,
        decoder_channels=(64, 32, 16, 8),
        head_hidden=32,
        head_layers=1,
    )
    obs = torch.randint(0, 256, (2, 4, 64, 64, 3), dtype=torch.uint8)
    actions = torch.randint(0, 5, (2, 4), dtype=torch.int64)
    rewards = torch.zeros(2, 4)
    cont = torch.ones(2, 4)
    out = wm(obs, actions)
    obs_f = nhwc_uint8_to_nchw_float(obs.reshape(8, 64, 64, 3)).view(2, 4, 3, 64, 64)
    loss = world_model_loss(
        obs=obs_f,
        recon=out.recon,
        recon_embed=out.recon_embed,
        recon_bottleneck=out.recon_bottleneck,
        reward=rewards,
        reward_pred=out.reward_pred,
        cont=cont,
        cont_logit=out.cont_logit,
        post_logits=out.rssm.posterior_logits,
        prior_logits=out.rssm.prior_logits,
        unimix=wm.rssm.unimix,
        recon_scale=0.0,
        recon_embed_scale=0.0,
        recon_bottleneck_scale=0.0,
        reward_scale=0.0,
        continue_scale=0.0,
        kl_scale=0.0,
        hz_map=out.hz_map,
        embed_map=out.embed_map,
        recon_map_scale=1.0,
    )
    assert float(loss.recon_map.detach()) > 0.0
    loss.total.backward()
    assert wm.hz_to_map is not None
    assert wm.hz_to_map.z_proj.weight.grad is not None
    assert float(wm.hz_to_map.z_proj.weight.grad.abs().sum()) > 0.0


def test_resnet_encoder_identity_flatten_and_backward() -> None:
    enc = Encoder(embed_dim=12288, channels=(96, 192, 384, 768), blocks=2)
    assert isinstance(enc.fc, torch.nn.Identity)
    obs = torch.randn(2, 3, 64, 64)
    embed = enc(obs)
    assert embed.shape == (2, 12288)
    embed.sum().backward()
    assert enc.conv[0].weight.grad is not None


def test_replay_buffer_samples_contiguous_windows() -> None:
    buf = ReplayBuffer(seed=0)
    T = 20
    for i in range(3):
        obs = torch.arange(T * 64 * 64 * 3, dtype=torch.uint8).view(T, 64, 64, 3).clone()
        obs[:, 0, 0, 0] = i
        actions = torch.arange(T, dtype=torch.int64)
        rewards = torch.arange(T, dtype=torch.float32)
        cont = torch.ones(T)
        cont[-1] = 0.0
        buf.add_episode(obs, actions, rewards, cont)

    batch = buf.sample(batch_size=4, seq_len=8)
    assert batch["obs"].shape == (4, 8, 64, 64, 3)
    assert batch["actions"].shape == (4, 8)
    for i in range(4):
        acts = batch["actions"][i]
        assert torch.equal(acts[1:] - acts[:-1], torch.ones(7, dtype=torch.int64))


def test_world_model_step_updates_weights() -> None:
    from training.device import make_grad_scaler
    from training.wm_step import world_model_step

    device = torch.device("cpu")
    wm = _tiny_wm().to(device)
    before = wm.encoder.conv[0].weight.detach().clone()
    optim = torch.optim.Adam(wm.parameters(), lr=1e-3)
    train_cfg = {
        "dyn_scale": 0.5,
        "rep_scale": 0.1,
        "free_nats": 1.0,
        "recon_scale": 1.0,
        "recon_embed_scale": 1.0,
        "recon_bottleneck_scale": 0.0,
        "reward_scale": 1.0,
        "continue_scale": 1.0,
        "kl_scale": 1.0,
        "grad_scale": 0.0,
        "recon_loss": "l1",
        "edge_weight": 0.0,
    }
    batch = {
        "obs": torch.randint(0, 256, (2, 4, 64, 64, 3), dtype=torch.uint8),
        "actions": torch.randint(0, 5, (2, 4), dtype=torch.int64),
        "rewards": torch.zeros(2, 4),
        "cont": torch.ones(2, 4),
    }
    scaler = make_grad_scaler(device, None)
    loss, metrics = world_model_step(
        wm, optim, batch, device=device, train_cfg=train_cfg, amp_dtype=None, scaler=scaler
    )
    assert torch.isfinite(loss.total)
    assert "recon" in metrics
    assert "recon_l1" in metrics
    assert "recon_blob" in metrics
    assert "recon_avatar" in metrics
    assert "recon_hud" in metrics
    assert not torch.equal(before, wm.encoder.conv[0].weight.detach())
