"""Tests for the M3 world model: heads, KL balancing, symlog reward, replay,
and the end-to-end forward/loss/step graph (DreamerV3 M3 reset)."""

from __future__ import annotations

import torch

from models.heads import ContinueHead, RewardHead, rssm_features
from models.preprocess import nhwc_uint8_to_nchw_unit
from models.encoder import Encoder
from models.symlog import (
    symexp,
    symlog,
    symlog_twohot_loss,
    symlog_twohot_mean,
    twohot_decode,
    twohot_encode,
)
from models.world_model import WorldModel
from training.losses import (
    categorical_kl,
    image_mse_loss,
    kl_balance,
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
        encoder_blocks=1,
        decoder_blocks=0,
    )


# --- symlog / two-hot reward -------------------------------------------------


def test_symlog_symexp_are_inverses() -> None:
    x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-4)


def test_twohot_encode_sums_to_one_and_brackets_value() -> None:
    bins = torch.linspace(-20.0, 20.0, 255)
    x = torch.tensor([0.0, 3.3, -3.3, 19.9, -19.9])
    enc = twohot_encode(x, bins)
    assert enc.shape == (5, 255)
    assert torch.allclose(enc.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(twohot_decode(enc, bins), x, atol=0.2)


def test_twohot_encode_exact_on_bin_center_is_effectively_onehot() -> None:
    bins = torch.linspace(-20.0, 20.0, 41)  # integer-spaced bins
    x = bins[10:11].clone()
    enc = twohot_encode(x, bins)
    assert float(enc[0, 10]) > 0.99


def test_symlog_twohot_loss_lower_for_correct_bin() -> None:
    bins = torch.linspace(-20.0, 20.0, 255)
    target = torch.tensor([1.0])
    logits_right = torch.zeros(1, 255)
    logits_right[0, twohot_encode(symlog(target), bins)[0].argmax()] = 10.0
    logits_wrong = torch.zeros(1, 255)
    logits_wrong[0, 0] = 10.0
    loss_right = symlog_twohot_loss(logits_right, bins, target)
    loss_wrong = symlog_twohot_loss(logits_wrong, bins, target)
    assert float(loss_right) < float(loss_wrong)


def test_symlog_twohot_mean_decodes_near_target() -> None:
    bins = torch.linspace(-20.0, 20.0, 255)
    target = torch.tensor([2.5, -0.3, 10.0])
    label = twohot_encode(symlog(target), bins)
    logits = torch.log(label.clamp_min(1e-8)) * 20.0  # sharp logits matching the label
    decoded = symlog_twohot_mean(logits, bins)
    assert torch.allclose(decoded, target, atol=0.5)


def test_reward_head_outputs_twohot_logits_and_backward() -> None:
    feat = torch.randn(3, 48, requires_grad=True)
    reward = RewardHead(48, hidden=16, layers=1, num_bins=255)
    cont = ContinueHead(48, hidden=16, layers=1)
    r = reward(feat)
    c = cont(feat)
    assert r.shape == (3, 255)
    assert c.shape == (3, 1)
    assert reward.bins.shape == (255,)
    (r.sum() + c.sum()).backward()
    assert feat.grad is not None
    assert torch.isfinite(feat.grad).all()


def test_rssm_features_shape() -> None:
    h = torch.randn(2, 8, 32)
    z = torch.zeros(2, 8, 4, 4)
    z[..., 0] = 1.0
    feat = rssm_features(h, z)
    assert feat.shape == (2, 8, 32 + 16)


# --- KL balancing (unchanged mechanism, still the paper's Eq. 3/5) ----------


def _near_identical_logits(scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior + a prior `scale` away from it, so total KL is controllable."""
    torch.manual_seed(0)
    post = torch.randn(2, 3, 8, 6)
    prior = (post + scale * torch.randn_like(post)).clone().requires_grad_(True)
    return post, prior


def test_dyn_floor_freezes_the_prior_below_free_nats() -> None:
    """The dyn floor stops the dynamics model learning once within `free_nats`.

    `kl_dyn` detaches the posterior, so it can only train the prior; it cannot
    restrict information. DreamerV3's default floors this too
    (`free_nats_dyn=None` reuses `free_nats`) — this test documents the
    mechanism, not a recommendation to always open it.
    """
    post, prior = _near_identical_logits(0.01)

    kl_floored, _, _, dyn_raw, _ = kl_balance(post, prior, free_nats=1.0)
    assert float(dyn_raw.detach()) < 1.0, "fixture must sit below the floor"
    kl_floored.backward()
    assert prior.grad is not None
    assert float(prior.grad.abs().sum()) == 0.0

    prior.grad = None
    kl_open, _, _, _, _ = kl_balance(post, prior, free_nats=1.0, free_nats_dyn=0.0)
    kl_open.backward()
    assert prior.grad is not None
    assert float(prior.grad.abs().sum()) > 0.0


def test_rep_floor_still_caps_the_rate_when_dyn_is_open() -> None:
    """`free_nats_dyn=0` must not loosen the posterior's rate budget."""
    torch.manual_seed(0)
    prior = torch.randn(2, 3, 8, 6)
    post = (prior + 0.01 * torch.randn_like(prior)).clone().requires_grad_(True)

    kl, _, _, _, rep_raw = kl_balance(post, prior, free_nats=1.0, free_nats_dyn=0.0)
    assert float(rep_raw.detach()) < 1.0
    kl.backward()
    assert post.grad is None or float(post.grad.abs().sum()) == 0.0


def test_free_nats_dyn_defaults_to_free_nats() -> None:
    """Omitting it must reproduce the DreamerV3 default (floor on both terms)."""
    post, prior = _near_identical_logits(0.5)
    default = kl_balance(post, prior, free_nats=1.0)
    explicit = kl_balance(post, prior, free_nats=1.0, free_nats_dyn=1.0)
    assert torch.allclose(default[0], explicit[0])


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


def test_prior_net_depth_follows_prior_layers() -> None:
    """Prior accuracy decides how much detail fits under free_nats."""
    from models.rssm import RSSM

    def n_linear(rssm: RSSM) -> int:
        return sum(isinstance(layer, torch.nn.Linear) for layer in rssm.prior_net)

    shallow = RSSM(
        embed_dim=32, action_dim=5, deter_dim=16, stoch=4, classes=4,
        hidden=32, prior_layers=1,
    )
    deep = RSSM(
        embed_dim=32, action_dim=5, deter_dim=16, stoch=4, classes=4,
        hidden=32, prior_layers=2,
    )
    assert n_linear(shallow) == 2
    assert n_linear(deep) == 3
    assert deep.prior_net[-1].out_features == 4 * 4


# --- image loss reduction (paper scale, not a per-pixel mean) --------------


def test_image_mse_loss_uses_pixel_sum_not_pixel_mean() -> None:
    """A per-pixel-mean loss is orders of magnitude weaker than KL; the paper's
    sum-over-pixels reduction is what makes reconstruction the dominant term.
    """
    pred = torch.zeros(2, 4, 3, 8, 8)
    target = torch.full((2, 4, 3, 8, 8), 0.1)
    loss = image_mse_loss(pred, target)
    per_pixel_mean = torch.nn.functional.mse_loss(pred, target)
    n_pixels = 3 * 8 * 8
    assert torch.allclose(loss, per_pixel_mean * n_pixels, atol=1e-4)


def test_image_mse_loss_zero_when_identical() -> None:
    img = torch.rand(2, 4, 3, 8, 8)
    assert torch.allclose(image_mse_loss(img, img), torch.zeros(()), atol=1e-6)


# --- end-to-end world model forward / loss / step ---------------------------


def test_world_model_forward_shapes_and_loss_backward() -> None:
    wm = _tiny_wm()
    b, t = 2, 6
    obs = torch.randint(0, 256, (b, t, 64, 64, 3), dtype=torch.uint8)
    actions = torch.randint(0, 5, (b, t), dtype=torch.int64)
    rewards = torch.randn(b, t)
    cont = torch.ones(b, t)

    out = wm(obs, actions)
    assert out.recon.shape == (b, t, 3, 64, 64)
    assert out.reward_pred.shape == (b, t, 255)
    assert out.cont_logit.shape == (b, t, 1)
    assert out.rssm.z_prior.shape == (b, t, 4, 4)
    assert out.rssm.z_posterior.shape == (b, t, 4, 4)
    # Decoder default is DreamerV3's linear output_activation: pixels in [0, 1].
    recon_detached = out.recon.detach()
    assert float(recon_detached.min()) >= -0.5 and float(recon_detached.max()) <= 1.5

    obs_f = nhwc_uint8_to_nchw_unit(obs.reshape(b * t, 64, 64, 3)).view(b, t, 3, 64, 64)
    loss = world_model_loss(
        obs=obs_f,
        recon=out.recon,
        reward=rewards,
        reward_pred=out.reward_pred,
        reward_bins=wm.reward_head.bins,
        cont=cont,
        cont_logit=out.cont_logit,
        post_logits=out.rssm.posterior_logits,
        prior_logits=out.rssm.prior_logits,
        unimix=wm.rssm.unimix,
    )
    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.recon_l1)
    assert torch.isfinite(loss.reward_mae)
    loss.total.backward()
    assert wm.encoder.conv[0].weight.grad is not None
    assert wm.rssm.prior_net[0].weight.grad is not None
    assert wm.reward_head.net[0].weight.grad is not None
    # The decoder is on the ONLY path from [h,z] to pixels now (no frozen
    # copy, no bypass) — it must receive gradient from the recon loss.
    assert wm.decoder.up[0][0].weight.grad is not None
    assert float(wm.decoder.up[0][0].weight.grad.abs().sum()) > 0.0


def test_no_auxiliary_embed_decoder_exists() -> None:
    """The pre-reset graph had a second decoder trained straight off the
    encoder embedding, which let recon loss drop without the RSSM carrying
    anything. That path must not exist anymore.
    """
    wm = _tiny_wm()
    assert not hasattr(wm, "embed_decoder")
    assert not hasattr(wm, "hz_to_map")
    from models.world_model import WorldModelOutput

    fields = WorldModelOutput.__dataclass_fields__
    assert "recon_embed" not in fields
    assert "recon_bottleneck" not in fields


def test_video_predict_open_loop_shapes() -> None:
    wm = _tiny_wm()
    b, t, context_len = 2, 10, 4
    obs = torch.randint(0, 256, (b, t, 64, 64, 3), dtype=torch.uint8)
    actions = torch.randint(0, 5, (b, t), dtype=torch.int64)
    vp = wm.video_predict(obs, actions, context_len=context_len)
    assert vp.context_recon.shape == (b, context_len, 3, 64, 64)
    assert vp.imagined_recon.shape == (b, t - context_len, 3, 64, 64)
    assert vp.imagined_reward.shape == (b, t - context_len)
    assert torch.isfinite(vp.context_recon).all()
    assert torch.isfinite(vp.imagined_recon).all()
    assert torch.isfinite(vp.imagined_reward).all()


def test_resnet_encoder_identity_flatten_and_backward() -> None:
    enc = Encoder(embed_dim=12288, channels=(96, 192, 384, 768), blocks=2)
    assert isinstance(enc.fc, torch.nn.Identity)
    obs = torch.randn(2, 3, 64, 64)
    embed = enc(obs)
    assert embed.shape == (2, 12288)
    embed.sum().backward()
    assert enc.conv[0].weight.grad is not None


def test_decoder_supports_residual_blocks_and_linear_output() -> None:
    from models.decoder import Decoder

    dec = Decoder(
        embed_dim=16 * 4 * 4, channels=(16, 8, 4, 4), blocks=1, output_activation="linear"
    )
    mods = list(dec.up.modules())
    assert any(isinstance(m, torch.nn.PixelShuffle) for m in mods)
    assert not any(isinstance(m, torch.nn.Tanh) for m in mods)
    out = dec(torch.randn(2, 16 * 4 * 4)).detach()
    assert out.shape == (2, 3, 64, 64)
    # Untrained linear decoder starts centered near 0.5 (gray), not squashed
    # through tanh/sigmoid.
    assert 0.0 < float(out.mean()) < 1.0


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


def test_fmt_duration_buckets() -> None:
    from training.replay_buffer import _fmt_duration

    assert _fmt_duration(8.2) == "8s"
    assert _fmt_duration(65) == "1m05s"
    assert _fmt_duration(3600) == "1h00m"


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
        "free_nats_dyn": None,
        "recon_scale": 1.0,
        "reward_scale": 1.0,
        "continue_scale": 1.0,
        "kl_scale": 1.0,
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
    for key in ("recon", "recon_l1", "reward", "reward_mae", "continue", "kl", "kl_dyn_raw", "kl_rep_raw"):
        assert key in metrics
    assert not torch.equal(before, wm.encoder.conv[0].weight.detach())


def test_m3_dreamer_s_config_matches_paper_recipe() -> None:
    """The new config must not reintroduce the pre-reset bypass/frozen-decoder
    graph or its loss knobs."""
    import sys
    from pathlib import Path

    import yaml

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    cfg = yaml.safe_load(Path("configs/m3_dreamer_s.yaml").read_text(encoding="utf-8"))
    enc = cfg["encoder"]
    dec = cfg["decoder"]
    train = cfg["train"]
    channels = tuple(int(c) for c in enc["channels"])

    assert int(enc["embed_dim"]) == channels[-1] * 4 * 4
    assert dec["output_activation"] == "linear"
    assert "recon_embed_scale" not in train
    assert "recon_bottleneck_scale" not in train
    assert "recon_blob_scale" not in train
    assert "recon_avatar_scale" not in train
    assert "recon_hud_scale" not in train
    assert "edge_weight" not in train
    assert float(train["free_nats"]) == 1.0
    assert train.get("free_nats_dyn") is None
    assert float(train["dyn_scale"]) == 0.5
    assert float(train["rep_scale"]) == 0.1
    assert float(train["recon_scale"]) == 1.0

    from train_world_model import build_model

    wm = build_model(cfg)
    assert not hasattr(wm, "hz_to_map")
    assert wm.decoder.output_activation == "linear"
