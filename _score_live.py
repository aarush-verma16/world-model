import sys
from pathlib import Path
sys.path.insert(0, "src")
from training.crafter_score import load_jsonl, score_from_episodes, ACHIEVEMENT_NAMES

rows = load_jsonl(Path("results/m20_local_map/collect_episodes.jsonl"))
print("episodes", len(rows), "last_env", rows[-1]["env_steps"], "last_len", rows[-1]["length"])
# score in chunks of the recent lives
for n in (50, 100, 200):
    score, pct = score_from_episodes(rows[-n:])
    print(
        f"last {n:3d} gmean={score:.2f} stone={pct['collect_stone']:.1f} "
        f"stone_pick={pct['make_stone_pickaxe']:.1f} coal={pct['collect_coal']:.1f} "
        f"iron={pct['collect_iron']:.1f} table={pct['place_table']:.1f} "
        f"wood_pick={pct['make_wood_pickaxe']:.1f}"
    )
# before the 58k resume vs after
before = [r for r in rows if int(r["env_steps"]) <= 58368]
after = [r for r in rows if int(r["env_steps"]) > 58368]
print("before", len(before), "after", len(after))
if after:
    score, pct = score_from_episodes(after)
    print(
        f"since resume gmean={score:.2f} n={len(after)} stone={pct['collect_stone']:.1f} "
        f"stone_pick={pct['make_stone_pickaxe']:.1f} coal={pct['collect_coal']:.1f} "
        f"wood_pick={pct['make_wood_pickaxe']:.1f} table={pct['place_table']:.1f}"
    )
    print("after env", after[0]["env_steps"], "->", after[-1]["env_steps"])
# sliding last-200 at a few cut points
for cut in (40000, 50000, 55000, 58368, 10**9):
    sub = [r for r in rows if int(r["env_steps"]) <= cut]
    if len(sub) < 20:
        continue
    score, pct = score_from_episodes(sub[-200:])
    print(f"at <={cut} last200={score:.2f} n={len(sub)} env={sub[-1]['env_steps']}")
