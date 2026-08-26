"""Full M4 20k metrics summary for go/no-go."""
from __future__ import annotations

import json
import math
from pathlib import Path

p = Path(r"c:\Users\Aarus\Desktop\world-model\results\m4_actor_critic\train_metrics.json")
h = json.loads(p.read_text(encoding="utf-8"))
print("n", len(h), "first", h[0]["step"], "last", h[-1]["step"])
print("keys", sorted(h[-1].keys()))

keys = [
    "total",
    "actor",
    "critic",
    "entropy",
    "return",
    "reward",
    "value",
    "retnorm_scale",
    "return_std",
    "adv",
    "reinforce",
    "backprop",
    "steps_per_sec",
    "vram_alloc",
]


def finite(x):
    return math.isfinite(float(x))


bad = []
for row in h:
    for k, v in row.items():
        if isinstance(v, (int, float)) and not finite(v):
            bad.append((row["step"], k, v))
print("nonfinite", len(bad), bad[:10])

idx = {int(x["step"]): x for x in h}
steps = sorted(idx)


def nearest(t):
    s = min(steps, key=lambda s: abs(s - t))
    return idx[s]


print("\nsnapshots")
for t in [1, 200, 1000, 2000, 4000, 8000, 12000, 16000, 20000]:
    r = nearest(t)
    print(int(r["step"]), {k: round(float(r[k]), 4) for k in keys if k in r})


def win(lo, hi):
    pts = [x for x in h if lo <= x["step"] <= hi]
    n = len(pts)
    means = {k: sum(x[k] for x in pts) / n for k in keys if k in pts[0]}
    print(f"{lo}-{hi} n={n}", {k: round(v, 4) for k, v in means.items()})
    return means


print("\nwindows")
w_early = win(1, 400)
w_mid = win(9600, 10400)
w_late = win(19200, 20000)
w_first1k = win(1, 1000)
w_last2k = win(18000, 20000)

print("\nmin/max last 5k")
last5 = [x for x in h if x["step"] >= 15000]
for k in ["entropy", "reward", "return", "critic", "actor", "value"]:
    xs = [x[k] for x in last5]
    print(k, "min", round(min(xs), 4), "max", round(max(xs), 4), "mean", round(sum(xs) / len(xs), 4))

print("\nentropy floor counts")
for thresh in [0.1, 0.2, 0.3, 0.5]:
    n = sum(1 for x in h if x["entropy"] < thresh)
    n_late = sum(1 for x in last5 if x["entropy"] < thresh)
    print(f"H<{thresh}: all {n}/{len(h)} last5k {n_late}/{len(last5)}")

print("\nreward extremes")
rw = [x["reward"] for x in h]
print("reward min", min(rw), "max", max(rw), "abs>1", sum(1 for x in rw if abs(x) > 1), "abs>5", sum(1 for x in rw if abs(x) > 5))
print("return min", min(x["return"] for x in h), "max", max(x["return"] for x in h))

print("\ncritic first400 vs last2k")
print("critic", round(w_early["critic"], 4), "->", round(w_last2k["critic"], 4))
print("entropy", round(w_early["entropy"], 4), "->", round(w_last2k["entropy"], 4))
print("reward", round(w_early["reward"], 4), "->", round(w_last2k["reward"], 4))
print("return", round(w_early["return"], 4), "->", round(w_last2k["return"], 4))
print("value", round(w_early["value"], 4), "->", round(w_last2k["value"], 4))
print("retnorm", round(w_early["retnorm_scale"], 4), "->", round(w_last2k["retnorm_scale"], 4))

# slope of critic last 5k
xs = [x["step"] for x in last5]
ys = [x["critic"] for x in last5]
mx = sum(xs) / len(xs)
my = sum(ys) / len(ys)
num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
den = sum((a - mx) ** 2 for a in xs)
slope = num / den if den else 0.0
print("critic slope last5k per 1k steps", round(slope * 1000, 6))

# last row
print("\nlast row")
print({k: h[-1][k] for k in h[-1]})
