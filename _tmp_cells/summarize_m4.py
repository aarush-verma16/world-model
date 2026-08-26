"""Summarize M4 train_metrics.json at ~7.5k."""
import json
from pathlib import Path

p = Path(r"c:\Users\Aarus\Desktop\world-model\results\m4_actor_critic\train_metrics.json")
h = json.loads(p.read_text(encoding="utf-8"))
print("n", len(h), "first_step", h[0]["step"], "last_step", h[-1]["step"])
print("keys", sorted(h[-1].keys()))
print("first", {k: h[0][k] for k in h[0] if k != "step"})
print("last", {k: h[-1][k] for k in h[-1] if k != "step"})

keys = ["total", "actor", "critic", "entropy", "return", "reward", "value", "retnorm_scale"]
idx = {int(x["step"]): x for x in h}
steps = sorted(idx)


def nearest(t):
    s = min(steps, key=lambda s: abs(s - t))
    return idx[s]


print("snapshots")
for t in [1, 200, 1000, 2000, 4000, 6000, 7500, 7800]:
    r = nearest(t)
    print(int(r["step"]), {k: round(float(r.get(k, float("nan"))), 4) for k in keys if k in r})


def win(lo, hi):
    pts = [x for x in h if lo <= x["step"] <= hi]
    if not pts:
        return
    n = len(pts)
    print(f"{lo}-{hi} n={n}", {k: round(sum(x[k] for x in pts) / n, 4) for k in keys if k in pts[0]})


print("windows")
for lo, hi in [(1, 400), (3600, 4000), (5600, 6000), (7200, 7800)]:
    win(lo, hi)
