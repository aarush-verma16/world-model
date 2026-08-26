"""Plot M5 env_steps_per_sec vs env_steps to confirm dashboard/GC decay."""
import json
from pathlib import Path

p = Path(r"c:\Users\Aarus\Desktop\world-model\results\m5_outer_loop\train_metrics.json")
h = json.loads(p.read_text(encoding="utf-8"))
print("n", len(h), "first", h[0].get("env_steps"), "last", h[-1].get("env_steps"))
print("keys", sorted(h[-1].keys()))

def nearest(t):
    return min(h, key=lambda x: abs(x.get("env_steps", 0) - t))

for t in [16, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000]:
    r = nearest(t)
    sps = r.get("env_steps_per_sec")
    print(int(r["env_steps"]), "sps", None if sps is None else round(float(sps), 3),
          "wm_l1", round(float(r.get("wm_recon_l1", float("nan"))), 4),
          "H", round(float(r.get("ac_entropy", float("nan"))), 3))

# window mean sps
def win(lo, hi):
    pts = [x for x in h if lo <= x.get("env_steps", 0) <= hi and "env_steps_per_sec" in x]
    if not pts:
        print(lo, hi, "none")
        return
    m = sum(x["env_steps_per_sec"] for x in pts) / len(pts)
    print(f"window {lo}-{hi} n={len(pts)} mean_sps={m:.3f} min={min(x['env_steps_per_sec'] for x in pts):.3f} max={max(x['env_steps_per_sec'] for x in pts):.3f}")

for lo, hi in [(16, 2000), (8000, 12000), (18000, 22000), (28000, 32000), (38000, 42000), (48000, 52000), (56000, 62000)]:
    win(lo, hi)

ev = json.loads(Path(r"c:\Users\Aarus\Desktop\world-model\results\m5_outer_loop\eval_metrics.json").read_text(encoding="utf-8"))
print("evals", [(int(x["env_steps"]), round(x["eval_return"], 3), round(x["eval_length"], 1)) for x in ev])
