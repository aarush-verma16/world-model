"""One-shot: slim 700k metrics JS, copy stills, plot KL vs recon L1."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(r"c:\Users\Aarus\Desktop\world-model")
src = root / "results" / "m3_dreamer_s"
fig = root / "research paper" / "figures"
fig.mkdir(parents=True, exist_ok=True)

h = json.loads((src / "train_metrics.json").read_text(encoding="utf-8"))
h.sort(key=lambda x: int(x["step"]))
idx = {int(x["step"]): x for x in h}
steps = sorted(idx)
picked = []
seen: set[int] = set()
for t in range(0, 700001, 5000):
    s = min(steps, key=lambda s, t=t: abs(s - max(t, 1)))
    if s not in seen:
        seen.add(s)
        picked.append(idx[s])
if int(h[-1]["step"]) not in seen:
    picked.append(h[-1])

keys = [
    "step",
    "recon",
    "recon_l1",
    "reward",
    "reward_mae",
    "continue",
    "kl",
    "kl_dyn_raw",
    "kl_rep_raw",
]
points = [{k: float(p[k]) for k in keys} for p in picked]
js = root / "research paper" / "data" / "metrics_700k.js"
js.parent.mkdir(parents=True, exist_ok=True)
payload = {"n_logged": len(h), "n_points": len(points), "points": points}
js.write_text("window.M3_700K = " + json.dumps(payload) + ";\n", encoding="utf-8")
print("wrote", js, "points", len(points), "from", len(h))

shutil.copy2(src / "recon_final.png", fig / "reset_recon_final_700k.png")
shutil.copy2(src / "video_pred_final.png", fig / "reset_video_pred_final_700k.png")
if (src / "recon_step_700000.png").exists():
    shutil.copy2(src / "recon_step_700000.png", fig / "reset_recon_step_700000.png")
if (src / "video_pred_step_700000.png").exists():
    shutil.copy2(src / "video_pred_step_700000.png", fig / "reset_video_pred_step_700000.png")
print("copied stills")

xs = [p["step"] / 1000 for p in points]
figp, ax1 = plt.subplots(figsize=(7.2, 3.6), dpi=140)
ax2 = ax1.twinx()
(l1,) = ax1.plot(xs, [p["kl_rep_raw"] for p in points], color="#c45c26", lw=1.6, label="kl_rep_raw")
(l2,) = ax2.plot(xs, [p["recon_l1"] for p in points], color="#1f4e79", lw=1.6, label="recon_l1")
ax1.axhline(1.0, color="#888", ls=":", lw=1.0)
ax1.set_xlabel("training step (thousands)")
ax1.set_ylabel("kl_rep_raw (nats)")
ax2.set_ylabel("recon_l1")
ax1.set_ylim(0, 7)
ax2.set_ylim(0, 0.04)
ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper right", fontsize=8)
ax1.set_title("M3 size-S frozen replay: KL vs recon L1 to 700k")
figp.tight_layout()
outp = fig / "m3_700k_recon_vs_kl.png"
figp.savefig(outp)
print("wrote", outp)
