"""World-model loss terms, including Dreamer-style KL balancing.

Loss = recon_[h,z] + recon_embed + reward + continue + KL. Logged per term
so a healthy sum cannot hide a collapsed/exploding KL (M3 exit criterion).

`recon_embed` is skip-free embed→pixels (what the RSSM actually receives).
`recon_bottleneck` is a log-compat alias of that same tensor on `WorldModel`;
its scale must stay 0 or the aux recon is counted twice. Do not put M1's
U-Net skip decoder on this graph — `stem_to_rgb` can copy the frame without
the embedding carrying anything the RSSM can use.

`edge_weight` / `grad_scale` are optional 1px-edge terms, off in the M3
config. `recon_blob` is 7px-tile mean L1, **off** in the M3 config (it
is a solid-color-per-tile objective). `recon_avatar` / `recon_hud` are
optional crop terms, also off. `recon_l1` is always unweighted 64x64 L1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from models.crafter_layout import TILE, avatar_slice, hud_slice, world_slice
from torch import Tensor


@dataclass
class WorldModelLossBreakdown:
    """Per-term losses (already reduced to scalars) plus the weighted total.

    `recon` / `recon_embed` / `recon_bottleneck` are the values actually
    multiplied into `total` (content-weighted when `edge_weight > 0`).
    `*_l1` are always plain unweighted pixel loss — the number to compare
    across runs and against the historical ~0.10 early / ~0.05 late band.
    """

    total: Tensor
    recon: Tensor
    recon_embed: Tensor
    recon_bottleneck: Tensor
    recon_l1: Tensor
    recon_embed_l1: Tensor
    recon_bottleneck_l1: Tensor
    recon_map: Tensor
    recon_blob: Tensor
    recon_embed_blob: Tensor
    recon_avatar: Tensor
    recon_embed_avatar: Tensor
    recon_hud: Tensor
    recon_embed_hud: Tensor
    grad: Tensor
    reward: Tensor
    continue_loss: Tensor
    kl: Tensor
    kl_dyn: Tensor
    kl_rep: Tensor
    kl_dyn_raw: Tensor
    kl_rep_raw: Tensor


def categorical_kl(
    post_logits: Tensor,
    prior_logits: Tensor,
    unimix: float = 0.01,
) -> Tensor:
    """KL(post || prior) per categorical variable, after unimix on both sides.

    Args:
        post_logits: `[..., stoch, classes]` raw posterior logits
        prior_logits: `[..., stoch, classes]` raw prior logits
        unimix: same floor used at sample time (must match RSSM.unimix)

    Returns:
        KL `[..., stoch]` in nats (summed over classes).
    """
    from models.rssm import unimix_probs

    post = unimix_probs(post_logits, unimix)
    prior = unimix_probs(prior_logits, unimix)
    return (post * (post.clamp_min(1e-8).log() - prior.clamp_min(1e-8).log())).sum(dim=-1)


def kl_balance(
    post_logits: Tensor,
    prior_logits: Tensor,
    *,
    unimix: float = 0.01,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
    free_nats: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """DreamerV2/V3 KL balancing with a free-nats floor on the *whole latent*.

    The `stoch` categorical variables are independent, so the joint KL of the
    full latent at one timestep is the **sum** (not average) of the per-variable
    KLs. Dreamer's `max(free_nats, KL)` floor applies to that per-timestep sum.
    Clamping each of the ~16-32 categorical variables individually (i.e. before
    summing) makes the floor far too strict — it would require *every single*
    variable to individually carry a full free nat, instead of the latent as a
    whole carrying one. That bug kept the KL loss pinned at the floor almost
    indefinitely even once the latent was already informative.

    Returns:
        `(kl_loss, kl_dyn_mean, kl_rep_mean, kl_dyn_raw_mean, kl_rep_raw_mean)`
        where the `_raw` values are the mean per-timestep *total* KL (summed
        over `stoch`, averaged over batch/time) before the free-nats floor.
    """
    kl_dyn_raw = categorical_kl(post_logits.detach(), prior_logits, unimix=unimix).sum(dim=-1)
    kl_rep_raw = categorical_kl(post_logits, prior_logits.detach(), unimix=unimix).sum(dim=-1)
    kl_dyn_raw_mean = kl_dyn_raw.mean()
    kl_rep_raw_mean = kl_rep_raw.mean()
    kl_dyn = kl_dyn_raw.clamp_min(free_nats) if free_nats > 0.0 else kl_dyn_raw
    kl_rep = kl_rep_raw.clamp_min(free_nats) if free_nats > 0.0 else kl_rep_raw
    kl_dyn_mean = kl_dyn.mean()
    kl_rep_mean = kl_rep.mean()
    kl_loss = dyn_scale * kl_dyn_mean + rep_scale * kl_rep_mean
    return kl_loss, kl_dyn_mean, kl_rep_mean, kl_dyn_raw_mean, kl_rep_raw_mean


def _pixel_loss(pred: Tensor, target: Tensor, kind: str) -> Tensor:
    if kind == "l1":
        return F.l1_loss(pred, target)
    if kind == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"unknown recon_loss_type {kind!r}")


def content_weight_map(target: Tensor, edge_weight: float) -> Tensor:
    """Per-pixel weight `[..., 1, H, W]`, boosted where `target` has structure.

    Plain per-pixel L1/MSE rewards matching each region's *average* color.
    In Crafter, flat grass/dirt is ~90% of pixels while HUD icons, sprite
    silhouettes, and tile edges -- the actually informative content -- are a
    tiny minority. Getting the grass right already gets loss very low, so
    the optimizer has little incentive to spend capacity on the sparse stuff
    (confirmed via a probe: `[h,z]` correctly picks up large regions like
    water, but leaves HUD/sprites blank even as pixel loss drops nicely).

    This weights each pixel by the squared local gradient magnitude of the
    *real* frame only (never the prediction -- it's a fixed target-side
    weight, detached from the graph). Squaring matters: Crafter's grass/dirt
    dithering has small but nonzero gradients everywhere, while HUD icon
    edges and sprite outlines are near-maximal-contrast jumps. Squaring
    keeps the dithering noise's weight close to 1x while sharp structural
    edges reach 10-100x, so the loss actually cares whether icons/sprites
    are there instead of averaging them away.

    Args:
        target: `[..., C, H, W]`, any leading dims, values in `[-1, 1]`.
        edge_weight: strength of the boost; `0.0` means uniform weight `1.0`
            (equivalent to unweighted L1/MSE).
    """
    dx = F.pad(target[..., :, 1:] - target[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(target[..., 1:, :] - target[..., :-1, :], (0, 0, 0, 1))
    mag = dx.abs().mean(dim=-3, keepdim=True) + dy.abs().mean(dim=-3, keepdim=True)
    return 1.0 + edge_weight * mag.pow(2)


def blob_recon_loss(pred: Tensor, target: Tensor, pool: int = 8) -> Tensor:
    """L1 on average-pooled images (sprite-scale cells, not 1px edges).

    64x64 L1 is a median: an 8px cow in a 16x16 decoder cell is a minority
    of that cell, so the optimum is to paint grass and the cow never even
    appears as a blur. Crafter tiles are 8px. Pooling to 8x8 makes that cow
    a majority of its cell, so the same L1 has to match the cow's color — a
    blob in the right place. This is not `edge_weight` (HUD digit strokes)
    and not an 8px decoder bottleneck (RSSM flatten stays 4x4).

    Args:
        pred / target: `[..., C, H, W]` in `[-1, 1]`. `H` and `W` must be
            divisible by `pool`.
        pool: window size. `8` → 8x8 cells on a 64x64 frame.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
    if pred.ndim < 4:
        raise ValueError(f"expected [..., C, H, W], got {tuple(pred.shape)}")
    pred_n = pred.reshape(-1, *pred.shape[-3:])
    target_n = target.reshape(-1, *target.shape[-3:])
    _, _, height, width = pred_n.shape
    if height % pool != 0 or width % pool != 0:
        raise ValueError(f"spatial {(height, width)} not divisible by pool={pool}")
    return F.l1_loss(F.avg_pool2d(pred_n, pool), F.avg_pool2d(target_n, pool))


def tile_blob_loss(pred: Tensor, target: Tensor, local_scale: float = 4.0) -> Tensor:
    """L1 on Crafter's 7px tiles, weighted by local 3×3 deviation.

    Trees are large/common so plain 8×8 blob L1 finds them. Cows, zombies,
    skeletons, and saplings are one 7px tile and rare; this term equalizes
    tiles and boosts cells that differ from their neighbors (an object on
    grass) without 1px `edge_weight`.
    """
    pred_w = world_slice(pred.reshape(-1, *pred.shape[-3:]))
    target_w = world_slice(target.reshape(-1, *target.shape[-3:]))
    pred_c = F.avg_pool2d(pred_w, TILE)
    target_c = F.avg_pool2d(target_w, TILE)
    diff = (pred_c - target_c).abs()
    bg = F.avg_pool2d(target_c, kernel_size=3, stride=1, padding=1)
    dev = (target_c - bg).abs().mean(dim=1, keepdim=True)
    weight = 1.0 + local_scale * dev.detach()
    return (weight * diff).mean()


def avatar_recon_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 on the 3×3 tiles around the (fixed) Crafter player camera."""
    return F.l1_loss(
        avatar_slice(pred.reshape(-1, *pred.shape[-3:])),
        avatar_slice(target.reshape(-1, *target.shape[-3:])),
    )


def hud_recon_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 on the 14×63 inventory strip plus 2×9 slot-mean L1.

    Amount glyphs are ~4px in a 7px slot. Slot-mean L1 makes occupancy
    (empty vs health=9 vs health=3) required; pixel L1 on the strip can
    then change the digit texture. Not skip-to-RGB; the HUD head paints
    this region from the bottom 4×4 row.
    """
    pred_h = hud_slice(pred.reshape(-1, *pred.shape[-3:]))
    target_h = hud_slice(target.reshape(-1, *target.shape[-3:]))
    pixel = F.l1_loss(pred_h, target_h)
    slot = F.l1_loss(F.avg_pool2d(pred_h, TILE), F.avg_pool2d(target_h, TILE))
    return pixel + slot


def weighted_pixel_loss(pred: Tensor, target: Tensor, kind: str, edge_weight: float) -> Tensor:
    """`_pixel_loss` but reweighted by `content_weight_map` when `edge_weight > 0`."""
    if edge_weight <= 0.0:
        return _pixel_loss(pred, target, kind)
    weight = content_weight_map(target, edge_weight).detach()
    diff = (pred - target).abs() if kind == "l1" else (pred - target).pow(2)
    return (weight * diff).mean()


def gradient_l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss between image finite-difference gradients (edge-aware term).

    Plain per-pixel L1/MSE rewards matching a region's *average* color, so a
    1-2px sprite outline or HUD icon/number stroke (a tiny fraction of a
    region's pixels) barely moves that average -- the optimizer has little
    incentive to sharpen it even as pixel loss keeps dropping. Comparing
    horizontal/vertical finite differences instead explicitly rewards
    matching *where* the image changes, which is what "sharp edges" actually
    means. This directly targets blurry backgrounds and illegible HUD bars
    that persist despite falling pixel loss.

    Args:
        pred / target: `[..., C, H, W]`, any leading dims.
    """
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def world_model_loss(
    *,
    obs: Tensor,
    recon: Tensor,
    recon_embed: Tensor,
    recon_bottleneck: Tensor,
    reward: Tensor,
    reward_pred: Tensor,
    cont: Tensor,
    cont_logit: Tensor,
    post_logits: Tensor,
    prior_logits: Tensor,
    unimix: float = 0.01,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
    free_nats: float = 1.0,
    recon_scale: float = 1.0,
    recon_embed_scale: float = 1.0,
    recon_bottleneck_scale: float = 0.0,
    reward_scale: float = 1.0,
    continue_scale: float = 1.0,
    kl_scale: float = 1.0,
    grad_scale: float = 0.0,
    recon_loss_type: str = "l1",
    edge_weight: float = 0.0,
    hz_map: Tensor | None = None,
    embed_map: Tensor | None = None,
    recon_map_scale: float = 0.0,
    recon_blob_scale: float = 0.0,
    recon_avatar_scale: float = 0.0,
    recon_hud_scale: float = 0.0,
) -> WorldModelLossBreakdown:
    """Assemble world-model terms.

    Args:
        obs / recon / recon_embed / recon_bottleneck: float images
            `[B, T, 3, H, W]` in `[-1, 1]`
        reward: `[B, T]`
        reward_pred / cont_logit: `[B, T, 1]` or `[B, T]`
        cont: `[B, T]` in `{0, 1}`
        post_logits / prior_logits: `[B, T, stoch, classes]`
        grad_scale: weight on the edge-aware gradient term (see
            `gradient_l1_loss`); `0.0` disables it entirely (default, for
            backward compat with earlier checkpoints/configs).
        edge_weight: weight on the content/edge reweighting of the pixel
            loss terms (see `content_weight_map`); `0.0` keeps plain
            uniform-weight L1/MSE (default, for backward compat).
        recon_blob_scale: weight on 7px-tile blob L1 (`tile_blob_loss`) for
            `[h,z]` and embed recon. `0.0` keeps it out of `total` (still
            logged).
        recon_avatar_scale: weight on the 3×3 player crop.
        recon_hud_scale: weight on the inventory strip (composited HUD head).
    """
    shared_embed = recon_embed is recon_bottleneck
    recon_l1 = _pixel_loss(recon, obs, recon_loss_type)
    recon_embed_l1 = _pixel_loss(recon_embed, obs, recon_loss_type)
    recon_loss = weighted_pixel_loss(recon, obs, recon_loss_type, edge_weight)
    recon_embed_loss = weighted_pixel_loss(recon_embed, obs, recon_loss_type, edge_weight)
    if shared_embed:
        recon_bottleneck_l1 = recon_embed_l1
        recon_bottleneck_loss = recon_embed_loss
        # Same tensor as recon_embed — never add it twice into `total`.
        recon_bottleneck_scale = 0.0
    else:
        recon_bottleneck_l1 = _pixel_loss(recon_bottleneck, obs, recon_loss_type)
        recon_bottleneck_loss = weighted_pixel_loss(
            recon_bottleneck, obs, recon_loss_type, edge_weight
        )
    if grad_scale > 0.0:
        grad_loss = gradient_l1_loss(recon, obs) + gradient_l1_loss(recon_embed, obs)
        if not shared_embed:
            grad_loss = grad_loss + gradient_l1_loss(recon_bottleneck, obs)
    else:
        grad_loss = recon.new_zeros(())

    if hz_map is not None and embed_map is not None and recon_map_scale != 0.0:
        recon_map = F.l1_loss(hz_map, embed_map.detach())
    else:
        recon_map = recon.new_zeros(())
        recon_map_scale = 0.0

    recon_blob = tile_blob_loss(recon, obs)
    recon_embed_blob = tile_blob_loss(recon_embed, obs)
    recon_avatar = avatar_recon_loss(recon, obs)
    recon_embed_avatar = avatar_recon_loss(recon_embed, obs)
    recon_hud = hud_recon_loss(recon, obs)
    recon_embed_hud = hud_recon_loss(recon_embed, obs)

    reward_pred = reward_pred.squeeze(-1)
    reward_loss = F.mse_loss(reward_pred, reward)

    cont_logit = cont_logit.squeeze(-1)
    continue_loss = F.binary_cross_entropy_with_logits(cont_logit, cont.float())

    kl_loss, kl_dyn, kl_rep, kl_dyn_raw, kl_rep_raw = kl_balance(
        post_logits,
        prior_logits,
        unimix=unimix,
        dyn_scale=dyn_scale,
        rep_scale=rep_scale,
        free_nats=free_nats,
    )

    total = (
        recon_scale * recon_loss
        + recon_embed_scale * recon_embed_loss
        + recon_bottleneck_scale * recon_bottleneck_loss
        + reward_scale * reward_loss
        + continue_scale * continue_loss
        + kl_scale * kl_loss
        + grad_scale * grad_loss
        + recon_map_scale * recon_map
        + recon_blob_scale * recon_blob
        + recon_blob_scale * recon_embed_blob
        + recon_avatar_scale * recon_avatar
        + recon_avatar_scale * recon_embed_avatar
        + recon_hud_scale * recon_hud
        + recon_hud_scale * recon_embed_hud
    )
    return WorldModelLossBreakdown(
        total=total,
        recon=recon_loss,
        recon_embed=recon_embed_loss,
        recon_bottleneck=recon_bottleneck_loss,
        recon_l1=recon_l1,
        recon_embed_l1=recon_embed_l1,
        recon_bottleneck_l1=recon_bottleneck_l1,
        recon_map=recon_map,
        recon_blob=recon_blob,
        recon_embed_blob=recon_embed_blob,
        recon_avatar=recon_avatar,
        recon_embed_avatar=recon_embed_avatar,
        recon_hud=recon_hud,
        recon_embed_hud=recon_embed_hud,
        grad=grad_loss,
        reward=reward_loss,
        continue_loss=continue_loss,
        kl=kl_loss,
        kl_dyn=kl_dyn,
        kl_rep=kl_rep,
        kl_dyn_raw=kl_dyn_raw,
        kl_rep_raw=kl_rep_raw,
    )
