"""Analyze m3_dreamer_s 700k run: metrics windows + video-pred strip L1."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"c:\Users\Aarus\Desktop\world-model")
METRICS = ROOT / "results" / "m3_dreamer_s" / "train_metrics.json"
RESULTS = ROOT / "results" / "m3_dreamer_s"
FRAME = 64
CONTEXT = 8  # seq_len 32 // 4
KEYS = ["recon", "recon_l1", "reward", "reward_mae", "continue", "kl", "kl_dyn_raw", "kl_rep_raw"]


def slope(pts: list[tuple[float, float]]) -> tuple[float, float, int] | None:
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(s for s, _ in pts) / n
    my = sum(v for _, v in pts) / n
    num = sum((s - mx) * (v - my) for s, v in pts)
    den = sum((s - mx) ** 2 for s, v in pts)
    if den == 0:
        return None
    return num / den, my, n


def window_mean(h, lo, hi):
    pts = [x for x in h if lo <= x["step"] <= hi]
    if not pts:
        return None
    out = {"n": len(pts), "lo": lo, "hi": hi}
    for k in KEYS:
        if k not in pts[0]:
            continue
        out[k] = sum(x[k] for x in pts) / len(pts)
    return out


def strip_l1(path: Path) -> dict:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    n_frames = w // FRAME
    row_h = h // 3
    truth = arr[:FRAME, :, :]
    model = arr[row_h : row_h + FRAME, :, :]
    per = []
    for i in range(n_frames):
        t = truth[:, i * FRAME : (i + 1) * FRAME]
        m = model[:, i * FRAME : (i + 1) * FRAME]
        per.append(float(np.mean(np.abs(t - m))))
    ctx = per[:CONTEXT]
    img = per[CONTEXT:]
    return {
        "path": path.name,
        "shape": [h, w],
        "n_frames": n_frames,
        "context_mean_l1": float(np.mean(ctx)) if ctx else None,
        "imagine_mean_l1": float(np.mean(img)) if img else None,
        "imagine_last8_l1": float(np.mean(img[-8:])) if len(img) >= 8 else None,
        "per_frame": [round(x, 5) for x in per],
    }


def main() -> None:
    h = json.loads(METRICS.read_text(encoding="utf-8"))
    last = int(h[-1]["step"])
    print("n_logs", len(h), "last_step", last)

    print("\n=== window means ===")
    windows = [
        (1, 5000),
        (45000, 50000),
        (95000, 100000),
        (195000, 200000),
        (245000, 250000),
        (295000, 300000),
        (345000, 350000),
        (395000, 400000),
        (445000, 450000),
        (495000, 500000),
        (545000, 550000),
        (595000, 600000),
        (645000, 650000),
        (675000, 680000),
        (690000, 700000),
    ]
    for lo, hi in windows:
        m = window_mean(h, lo, hi)
        if not m:
            continue
        print(
            f"{lo}-{hi} n={m['n']}  recon_l1={m['recon_l1']:.5f}  recon={m['recon']:.3f}  "
            f"kl_rep={m['kl_rep_raw']:.3f}  kl_dyn={m['kl_dyn_raw']:.3f}  "
            f"rew={m['reward']:.4f}  mae={m.get('reward_mae', float('nan')):.5f}"
        )

    print("\n=== slopes per 10k ===")
    for span in [20000, 50000, 100000, 200000, 300000, 400000]:
        lo = last - span
        pts_base = [x for x in h if x["step"] >= lo]
        print(f"last {span} (from {lo}):")
        for k in ["recon_l1", "recon", "kl_rep_raw", "kl_dyn_raw", "reward", "reward_mae"]:
            s = slope([(x["step"], x[k]) for x in pts_base])
            if s:
                b, my, n = s
                print(f"  {k}: mean={my:.5f}  d/10k={b * 10000:+.5f}  n={n}")

    print("\n=== KL bins last 50k / 100k ===")
    for span in [50000, 100000]:
        pts = [x for x in h if x["step"] >= last - span]
        print(f"last {span} n={len(pts)}")
        for thresh in [1.0, 1.5, 1.7, 1.8, 2.0, 2.2]:
            frac = sum(1 for x in pts if x["kl_rep_raw"] < thresh) / len(pts)
            print(f"  kl_rep_raw < {thresh}: {frac:.1%}")
        print(
            f"  min={min(x['kl_rep_raw'] for x in pts):.3f}  "
            f"p50={sorted(x['kl_rep_raw'] for x in pts)[len(pts)//2]:.3f}  "
            f"max={max(x['kl_rep_raw'] for x in pts):.3f}"
        )

    late = [x for x in h if x["step"] >= last - 100000]
    n = len(late)
    mx = sum(x["kl_rep_raw"] for x in late) / n
    my = sum(x["recon_l1"] for x in late) / n
    num = sum((x["kl_rep_raw"] - mx) * (x["recon_l1"] - my) for x in late)
    denx = math.sqrt(sum((x["kl_rep_raw"] - mx) ** 2 for x in late))
    deny = math.sqrt(sum((x["recon_l1"] - my) ** 2 for x in late))
    print("\ncorr(kl_rep, recon_l1) last 100k:", round(num / (denx * deny), 3) if denx * deny else None)

    print("\n=== 5k-bucket series (for canvas) ===")
    bucket = 20000
    buckets: dict[int, list] = {}
    for x in h:
        b = int(x["step"] // bucket) * bucket
        buckets.setdefault(b, []).append(x)
    cats = []
    recon_l1s = []
    kls = []
    rews = []
    maes = []
    recs = []
    for b in sorted(buckets):
        pts = buckets[b]
        cats.append(b)
        recon_l1s.append(round(sum(x["recon_l1"] for x in pts) / len(pts), 5))
        kls.append(round(sum(x["kl_rep_raw"] for x in pts) / len(pts), 4))
        rews.append(round(sum(x["reward"] for x in pts) / len(pts), 4))
        maes.append(round(sum(x["reward_mae"] for x in pts) / len(pts), 5))
        recs.append(round(sum(x["recon"] for x in pts) / len(pts), 3))
    print("cats", cats)
    print("recon_l1", recon_l1s)
    print("recon_mse", recs)
    print("kl_rep", kls)
    print("reward", rews)
    print("reward_mae", maes)

    print("\n=== video-pred strip L1 (truth vs model row, 64px tiles) ===")
    names = [
        "video_pred_step_300000.png",
        "video_pred_step_400000.png",
        "video_pred_step_500000.png",
        "video_pred_step_600000.png",
        "video_pred_step_700000.png",
        "video_pred_final.png",
    ]
    for name in names:
        path = RESULTS / name
        if not path.exists():
            print("missing", name)
            continue
        s = strip_l1(path)
        print(
            f"{s['path']} {s['shape']} frames={s['n_frames']}  "
            f"ctx_L1={s['context_mean_l1']:.4f}  imag_L1={s['imagine_mean_l1']:.4f}  "
            f"imag_last8={s['imagine_last8_l1']:.4f}"
        )
        print("  per_frame", s["per_frame"])


if __name__ == "__main__":
    main()
