import json
from pathlib import Path

rows = json.loads(Path("results/m20_local_map/train_metrics.json").read_text(encoding="utf-8"))
print("n", len(rows), "last", rows[-1].get("env_steps"))
print("--- tail ---")
for r in rows[-25:]:
    print(
        f"{r.get('env_steps'):6} sps={float(r.get('env_steps_per_sec') or 0):6.3f} "
        f"last200={float(r.get('online_crafter_score_last200') or 0):6.2f} "
        f"online={float(r.get('online_crafter_score') or 0):6.2f} "
        f"H={float(r.get('ac_entropy') or 0):.3f} "
        f"len={r.get('collect_ep_len')}"
    )
ev = json.loads(Path("results/m20_local_map/eval_metrics.json").read_text(encoding="utf-8"))
print("--- eval ---")
for e in ev:
    print(e.get("env_steps"), "score", round(float(e.get("eval_crafter_score", 0)), 3), "len", e.get("eval_length"))
