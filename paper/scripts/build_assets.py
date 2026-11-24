"""Rebuild every number, table, and figure the paper prints from `paper/data/`.

Run from the repository root:

    c:/Users/Aarus/miniforge3/envs/worldmodel/python.exe paper/scripts/build_assets.py

Writes `paper/latex/generated/` (macros and tables) and `paper/latex/figures/`
(PDF plots). Nothing in the LaTeX source should restate a count by hand.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "data"
OUT_GEN = ROOT / "paper" / "latex" / "generated"
OUT_FIG = ROOT / "paper" / "latex" / "figures"

BENCH = (
    ("m22_field_craft", "beside resource", "anchored"),
    ("m23_uniform_window", "beside resource", "uniform"),
    ("m24_walk_home", "at home", "anchored"),
)
BENCH_LABEL = {
    "m22_field_craft": "Beside resource, anchored",
    "m23_uniform_window": "Beside resource, uniform",
    "m24_walk_home": "At home, anchored",
}
ROOM = (
    ("room_in_anchored", True, "anchored"),
    ("room_in_uniform", True, "uniform"),
    ("room_out_anchored", False, "anchored"),
    ("room_out_uniform", False, "uniform"),
)
ROOM_LABEL = {
    "room_in_anchored": "In crop, anchored",
    "room_in_uniform": "In crop, uniform",
    "room_out_anchored": "Behind agent, anchored",
    "room_out_uniform": "Behind agent, uniform",
}
CLASSES = ("grass", "stone", "path", "coal", "iron", "table", "furnace")
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def macro_name(*parts: str) -> str:
    """LaTeX macro names cannot hold digits or underscores."""
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
              "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    out = []
    for part in parts:
        for chunk in part.replace("-", "_").split("_"):
            if not chunk:
                continue
            chunk = "".join(digits.get(ch, ch) for ch in chunk)
            out.append(chunk[0].upper() + chunk[1:])
    return "\\" + "".join(out)


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


# --------------------------------------------------------------------------- benchmark


def benchmark_counts(slug: str) -> dict:
    # Several lives can end in the same logged chunk and share `env_steps`; every row is a distinct life.
    unique = load_jsonl(DATA / "benchmark" / slug / "collect_episodes.jsonl")
    assisted = [r for r in unique if r.get("assisted")]
    actor = [r for r in unique if not r.get("assisted")]
    handoff = [r for r in assisted if r.get("achievement_make_stone_pickaxe", 0) > 0]
    teacher_rare = [r for r in handoff if r.get("achievement_make_iron_pickaxe", 0) > 0]
    actor_rare = [r for r in actor if r.get("achievement_make_iron_pickaxe", 0) > 0]
    stone_all = [r for r in unique if r.get("achievement_make_stone_pickaxe", 0) > 0]
    evals = json.loads((DATA / "benchmark" / slug / "eval_metrics.json").read_text(encoding="utf-8"))
    held = [(int(e["env_steps"]), int(round(e["eval_pct_make_iron_pickaxe"] / 10.0))) for e in evals]
    held_stone = [(int(e["env_steps"]), int(round(e["eval_pct_make_stone_pickaxe"] / 10.0))) for e in evals]
    return {
        "lives": len(unique),
        "actor_lives": len(actor),
        "assisted_lives": len(assisted),
        "handoffs": len(handoff),
        "assisted_without_handoff": len(assisted) - len(handoff),
        "assisted_without_handoff_env": [int(r["env_steps"]) for r in assisted if r not in handoff][:1]
        + [int(r["env_steps"]) for r in assisted if r not in handoff][-1:],
        "teacher_rare": len(teacher_rare),
        "actor_rare": len(actor_rare),
        "actor_rare_lives": [(int(r["env_steps"]), int(r["length"])) for r in actor_rare],
        "stone_all": len(stone_all),
        "held_rare": held,
        "held_stone": held_stone,
        "last_env": max(int(r["env_steps"]) for r in unique),
    }


def start_checkpoint_counts() -> dict:
    rows = load_jsonl(DATA / "benchmark" / "m20_local_map" / "collect_episodes.jsonl")
    rare = [r for r in rows if r.get("achievement_make_iron_pickaxe", 0) > 0]
    return {"lives": len(rows), "rare": len(rare), "last_env": max(int(r["env_steps"]) for r in rows)}


# --------------------------------------------------------------------------- room


def room_counts(slug: str) -> dict:
    rows = load_jsonl(DATA / "room" / slug / "episodes.jsonl")
    metrics = json.loads((DATA / "room" / slug / "train_metrics.json").read_text(encoding="utf-8"))
    teacher = [r for r in rows if r.get("teacher")]
    actor = [r for r in rows if not r.get("teacher")]
    success = [r for r in actor if r["return"] >= 1]
    lengths = np.asarray([r["length"] for r in success], dtype=float)
    evals = [(int(m["env_steps"]), float(m["eval_success"]), float(m.get("eval_episodes", 8)))
             for m in metrics if "eval_success" in m]
    steps = np.asarray([r["env_steps"] for r in actor], dtype=float)
    cut = np.quantile(steps, 0.75)
    late = [r for r in actor if r["env_steps"] > cut]
    late_success = sum(1 for r in late if r["return"] >= 1)
    pose_ok = all(int(r["in_view_at_use"]) == (1 if ROOM_POSE[slug] else 0) for r in success)
    return {
        "teacher": len(teacher),
        "teacher_success": sum(1 for r in teacher if r["return"] >= 1),
        "teacher_len": sorted({int(r["length"]) for r in teacher}),
        "actor": len(actor),
        "success": len(success),
        "timeouts": sum(1 for r in actor if r["return"] < 1),
        "len_p10": float(np.percentile(lengths, 10)),
        "len_p50": float(np.percentile(lengths, 50)),
        "len_p90": float(np.percentile(lengths, 90)),
        "lengths": lengths.tolist(),
        "evals": evals,
        "late": len(late),
        "late_success": late_success,
        "pose_ok": pose_ok,
        "first_success": int(min(r["env_steps"] for r in success)),
        "last_env": int(max(m["env_steps"] for m in metrics)),
    }


ROOM_POSE = {slug: pose for slug, pose, _ in ROOM}


# --------------------------------------------------------------------------- figures


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })


def fig_field(field: dict, room_probe: dict) -> None:
    models = field["models"]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.15), gridspec_kw={"width_ratios": [2.3, 1.15, 1.0]})

    ax = axes[0]
    width = 0.26
    x = np.arange(len(CLASSES))
    for i, (slug, _pose, _win) in enumerate(BENCH):
        probs = [models[slug]["posterior"][f"{c}_prob"] for c in CLASSES]
        ax.bar(x + (i - 1) * width, probs, width, color=COLORS[i], label=BENCH_LABEL[slug])
    counts = [field["baseline"][f"{c}_cells"] for c in CLASSES]
    ax.set_xticks(x, [f"{c}\n{n:,}" for c, n in zip(CLASSES, counts)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(r"mean $P(\mathrm{true\ class})$")
    ax.set_title("(a) Benchmark field head, mode of $z^{\\mathrm{post}}$")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    horizons = [-2, 1, 5, 15]
    for i, (slug, _pose, _win) in enumerate(BENCH):
        accs = [models[slug]["posterior"]["material_accuracy"]] + [
            models[slug]["prior"][str(k)]["material_accuracy"] for k in (1, 5, 15)]
        ax.plot(horizons, accs, marker="o", ms=3, color=COLORS[i], lw=1.2)
    base = field["baseline"]["material_accuracy"]
    ax.axhline(base, color="0.4", ls="--", lw=0.9)
    ax.text(15, base - 0.025, "all-grass", ha="right", va="top", color="0.35", fontsize=6.5)
    ax.set_xticks(horizons, ["post", "1", "5", "15"])
    ax.set_xlabel("imagination step ($z^{\\mathrm{prior}}$)")
    ax.set_ylabel("cell accuracy")
    ax.set_ylim(0.45, 0.8)
    ax.set_title("(b) Field under imagination")

    ax = axes[2]
    labels, values = [], []
    short = {"room_in_anchored": "in crop\nanch.", "room_in_uniform": "in crop\nunif.",
             "room_out_anchored": "behind\nanch.", "room_out_uniform": "behind\nunif."}
    for slug, _pose, _win in ROOM:
        labels.append(short[slug])
        values.append(room_probe[slug]["in_view"]["station_prob"])
    bench_table = [models[slug]["posterior"]["table_prob"] for slug, _p, _w in BENCH]
    ax.bar(np.arange(4), values, color=COLORS, zorder=2)
    ax.axhspan(min(bench_table), max(bench_table), color="0.55", alpha=0.8, lw=0, zorder=3)
    ax.set_xticks(np.arange(4), labels, fontsize=6.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("station $P$ (station-facing ep.)")
    ax.set_title("(c) Room field head")

    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT_FIG / "field_head.pdf")
    plt.close(fig)


def fig_room(room: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.1), gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    markers = ("o", "s", "^", "D")
    for i, (slug, _pose, _win) in enumerate(ROOM):
        evals = room[slug]["evals"]
        xs = [e[0] / 1000 for e in evals]
        ys = [e[1] * e[2] for e in evals]
        ax.plot(xs, ys, marker=markers[i], ms=3.5, lw=1.1, color=COLORS[i], label=ROOM_LABEL[slug],
                ls="-" if "anchored" in slug else "--")
    ax.set_ylim(-0.3, 8.4)
    ax.set_yticks(range(0, 9, 2))
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("eval successes / 8")
    ax.set_title("(a) Evaluation without the teacher")
    ax.legend(frameon=False, loc="lower right", fontsize=6.3)

    ax = axes[1]
    data = [room[slug]["lengths"] for slug, _p, _w in ROOM]
    parts = ax.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True, showfliers=True,
                       flierprops={"markersize": 2, "markeredgewidth": 0.4})
    for patch, color in zip(parts["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for median in parts["medians"]:
        median.set_color("black")
    ax.set_yticks(range(1, 5), [ROOM_LABEL[s] for s, _p, _w in ROOM])
    ax.invert_yaxis()
    ax.axvline(52, color="0.3", ls=":", lw=0.9)
    ax.axvline(54, color="0.3", ls=":", lw=0.9)
    ax.text(150, 0.45, "dotted: teacher uses at 52 / 54", fontsize=6.3, color="0.3", va="top", ha="right")
    ax.set_xlabel("length of a successful episode (steps)")
    ax.set_title("(b) When the actor presses use")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT_FIG / "room_results.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- tex


def write_tables(bench: dict, room: dict, field: dict, room_probe: dict, start: dict) -> None:
    macros: list[str] = ["% Generated by paper/scripts/build_assets.py. Do not edit."]

    def m(name: str, value) -> None:
        macros.append(f"\\newcommand{{{name}}}{{{value}}}")

    m("\\StartLives", fmt_int(start["lives"]))
    m("\\StartRare", start["rare"])
    m("\\StartSteps", fmt_int(start["last_env"]))

    for slug, _pose, _win in BENCH:
        c = bench[slug]
        m(macro_name("bench", slug, "lives"), fmt_int(c["lives"]))
        m(macro_name("bench", slug, "actor"), fmt_int(c["actor_lives"]))
        m(macro_name("bench", slug, "handoffs"), c["handoffs"])
        m(macro_name("bench", slug, "teacherrare"), c["teacher_rare"])
        m(macro_name("bench", slug, "actorrare"), c["actor_rare"])
        m(macro_name("bench", slug, "teacherpct"), f"{100 * c['teacher_rare'] / c['handoffs']:.0f}")
        m(macro_name("bench", slug, "heldstone"), ", ".join(str(k) for _s, k in c["held_stone"]))
        m(macro_name("bench", slug, "heldstonefinal"), c["held_stone"][-1][1])

    base = field["baseline"]
    m("\\FieldCells", fmt_int(base["cells"]))
    m("\\FieldBaseline", f"{100 * base['material_accuracy']:.1f}")
    for c in CLASSES:
        m(macro_name("field", c, "cells"), fmt_int(base[f"{c}_cells"]))

    for slug, _pose, _win in ROOM:
        c = room[slug]
        m(macro_name(slug, "success"), c["success"])
        m(macro_name(slug, "actor"), c["actor"])
        m(macro_name(slug, "timeouts"), c["timeouts"])
        m(macro_name(slug, "median"), f"{c['len_p50']:.0f}")
        m(macro_name(slug, "late"), f"{c['late_success']}/{c['late']}")

    (OUT_GEN / "numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    # Table: benchmark field head
    lines = [
        "% Generated by paper/scripts/build_assets.py. Do not edit.",
        "\\begin{tabular}{l r r r r r r r r}",
        "\\toprule",
        " & & \\multicolumn{7}{c}{mean $P(\\text{true class})$ under the mode of $z^{\\text{post}}$} \\\\",
        "\\cmidrule(lr){3-9}",
        "Run & Acc. & grass & stone & path & coal & iron & table & furnace \\\\",
        "\\midrule",
        f"Cells & --- & {' & '.join(fmt_int(base[f'{c}_cells']) for c in CLASSES)} \\\\",
        f"All-grass & {100 * base['material_accuracy']:.1f}\\% & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\",
        "\\midrule",
    ]
    for slug, _pose, _win in BENCH:
        post = field["models"][slug]["posterior"]
        vals = " & ".join(f"{post[f'{c}_prob']:.3f}" for c in CLASSES)
        lines.append(f"{BENCH_LABEL[slug]} & {100 * post['material_accuracy']:.1f}\\% & {vals} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT_GEN / "table_field.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table: benchmark switches
    lines = [
        "% Generated by paper/scripts/build_assets.py. Do not edit.",
        "\\begin{tabular}{l l r r r l}",
        "\\toprule",
        "Station & Window & Lives & Teacher finishes & Actor, unaided & Held-out, per checkpoint \\\\",
        "\\midrule",
    ]
    for slug, pose, win in BENCH:
        c = bench[slug]
        held = c["held_rare"]
        held_text = ", ".join(f"{k}" for _s, k in held) + " (of 10)"
        lines.append(
            f"{pose} & {win} & "
            f"{fmt_int(c['lives'])} & {c['teacher_rare']}/{c['handoffs']} & {c['actor_rare']} & {held_text} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT_GEN / "table_switches.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table: benchmark field under imagination
    shown = ("grass", "stone", "path", "table", "furnace")
    lines = [
        "% Generated by paper/scripts/build_assets.py. Do not edit.",
        "\\begin{tabular}{l l r " + "r " * len(shown) + "}",
        "\\toprule",
        " & & & \\multicolumn{" + str(len(shown)) + "}{c}{recall} \\\\",
        "\\cmidrule(lr){4-" + str(3 + len(shown)) + "}",
        "Run & Source & Acc. & " + " & ".join(shown) + " \\\\",
        "\\midrule",
    ]
    for slug, _pose, _win in BENCH:
        model = field["models"][slug]
        rows = [("posterior", model["posterior"])] + [
            (f"prior, step {k}", model["prior"][str(k)]) for k in (1, 5, 15)]
        for i, (name, stats) in enumerate(rows):
            label = BENCH_LABEL[slug] if i == 0 else ""
            vals = " & ".join(f"{stats[f'{c}_recall']:.2f}" for c in shown)
            lines.append(f"{label} & {name} & {100 * stats['material_accuracy']:.1f}\\% & {vals} \\\\")
        if slug != BENCH[-1][0]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT_GEN / "table_prior.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for slug, _pose, _win in BENCH:
        for k in (1, 5, 15):
            stats = field["models"][slug]["prior"][str(k)]
            m(macro_name("prior", slug, "acc", str(k)), f"{100 * stats['material_accuracy']:.0f}")
            m(macro_name("prior", slug, "table", str(k)), f"{stats['table_recall']:.2f}")
    (OUT_GEN / "numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    # Table: room
    lines = [
        "% Generated by paper/scripts/build_assets.py. Do not edit.",
        "\\begin{tabular}{l l l r r l r}",
        "\\toprule",
        "Station at \\texttt{use} & Window & Teacher & Actor \\texttt{use} & Last quarter & Eval (/8) & Median length \\\\",
        "\\midrule",
    ]
    for slug, pose, win in ROOM:
        c = room[slug]
        evals = ", ".join(f"{int(round(s * n))}" for _e, s, n in c["evals"])
        lo, hi = wilson(c["success"], c["actor"])
        lines.append(
            f"{'in crop' if pose else 'behind agent'} & {win} & {c['teacher_success']}/{c['teacher']} & "
            f"{c['success']}/{c['actor']} & {c['late_success']}/{c['late']} & {evals} & {c['len_p50']:.0f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT_GEN / "table_room.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table: room probe
    lines = [
        "% Generated by paper/scripts/build_assets.py. Do not edit.",
        "\\begin{tabular}{l r r r r r r}",
        "\\toprule",
        " & \\multicolumn{3}{c}{station-facing episode} & \\multicolumn{3}{c}{facing-away episode} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Trained on & recall & $P$ & false & recall & $P$ & false \\\\",
        "\\midrule",
    ]
    for slug, _pose, _win in ROOM:
        a = room_probe[slug]["in_view"]
        b = room_probe[slug]["out_of_view"]
        lines.append(
            f"{ROOM_LABEL[slug]} & {a['station_recall']:.2f} & {a['station_prob']:.3f} & {a['false_station']:.2f} & "
            f"{b['station_recall']:.2f} & {b['station_prob']:.3f} & {b['false_station']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT_GEN / "table_room_probe.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_GEN.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    style()
    field = json.loads((DATA / "probes" / "field_probe_summary.json").read_text(encoding="utf-8"))
    room_probe = json.loads((DATA / "probes" / "room_probe_summary.json").read_text(encoding="utf-8"))
    bench = {slug: benchmark_counts(slug) for slug, _p, _w in BENCH}
    room = {slug: room_counts(slug) for slug, _p, _w in ROOM}
    start = start_checkpoint_counts()
    write_tables(bench, room, field, room_probe, start)
    fig_field(field, room_probe)
    fig_room(room)
    summary = {
        "start_checkpoint": start,
        "benchmark": {k: {kk: vv for kk, vv in v.items()} for k, v in bench.items()},
        "room": {k: {kk: vv for kk, vv in v.items() if kk != "lengths"} for k, v in room.items()},
    }
    (OUT_GEN / "numbers.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
