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


INJECT_EVERY = 25_000


def spans_injection(row: dict) -> bool:
    """Teacher injection replays demonstrator lives on the collect env, so the actor life open at
    that moment resumes inside the demonstrator's last world and inherits its achievements."""
    end = int(row["env_steps"])
    start = end - int(row["length"])
    return end // INJECT_EVERY > start // INJECT_EVERY


def load_lives(slug: str) -> tuple[list[dict], list[dict]]:
    """Returns `(kept, spanning)`. Every log row is a distinct life, even when rows share `env_steps`."""
    rows = load_jsonl(DATA / "benchmark" / slug / "collect_episodes.jsonl")
    return [r for r in rows if not spans_injection(r)], [r for r in rows if spans_injection(r)]


def benchmark_counts(slug: str) -> dict:
    unique, spanning = load_lives(slug)
    assisted = [r for r in unique if r.get("assisted")]
    actor = [r for r in unique if not r.get("assisted")]
    handoff = [r for r in assisted if r.get("achievement_make_stone_pickaxe", 0) > 0]
    teacher_rare = [r for r in handoff if r.get("achievement_make_iron_pickaxe", 0) > 0]
    actor_rare = [r for r in actor if r.get("achievement_make_iron_pickaxe", 0) > 0]
    stone_all = [r for r in unique if r.get("achievement_make_stone_pickaxe", 0) > 0]
    evals = json.loads((DATA / "benchmark" / slug / "eval_metrics.json").read_text(encoding="utf-8"))
    held = [(int(e["env_steps"]), int(round(e["eval_pct_make_iron_pickaxe"] / 10.0))) for e in evals]
    held_stone = [(int(e["env_steps"]), int(round(e["eval_pct_make_stone_pickaxe"] / 10.0))) for e in evals]
    first_handoff = min(int(r["env_steps"]) for r in handoff)
    early_stone = [r for r in actor if int(r["env_steps"]) < first_handoff
                   and r.get("achievement_make_stone_pickaxe", 0) > 0]
    return {
        "lives": len(unique),
        "spanning": len(spanning),
        "spanning_rare": sum(1 for r in spanning if r.get("achievement_make_iron_pickaxe", 0) > 0),
        "spanning_short": min(int(r["length"]) for r in spanning
                              if r.get("achievement_make_iron_pickaxe", 0) > 0) if any(
            r.get("achievement_make_iron_pickaxe", 0) > 0 for r in spanning) else None,
        "first_handoff": first_handoff,
        "early_stone": len(early_stone),
        "early_rare": sum(1 for r in early_stone if r.get("achievement_make_iron_pickaxe", 0) > 0),
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
    rows, spanning = load_lives("m20_local_map")
    rare = [r for r in rows if r.get("achievement_make_iron_pickaxe", 0) > 0]
    return {
        "lives": len(rows),
        "rare": len(rare),
        "spanning": len(spanning),
        "spanning_rare": sum(1 for r in spanning if r.get("achievement_make_iron_pickaxe", 0) > 0),
        "last_env": max(int(r["env_steps"]) for r in rows + spanning),
    }


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


CRAFTER_MATERIALS = ("outside", "water", "grass", "stone", "path", "sand", "tree", "lava",
                     "coal", "iron", "diamond", "table", "furnace")
MATERIAL_COLORS = {
    "outside": "#000000", "water": "#3a7bd5", "grass": "#7cc35a", "stone": "#9a9a9a",
    "path": "#d9c49a", "sand": "#efe2a2", "tree": "#2f6b2a", "lava": "#e2541b",
    "coal": "#333333", "iron": "#c98d6b", "diamond": "#bff3f5", "table": "#8b5a2b",
    "furnace": "#5b2c2c",
}


def _field_image(codes: np.ndarray) -> np.ndarray:
    """`codes` `[63]` x-major material codes -> RGB `[7, 9, 3]`."""
    from matplotlib.colors import to_rgb

    grid = codes.reshape(9, 7).T
    return np.asarray([[to_rgb(MATERIAL_COLORS[CRAFTER_MATERIALS[c]]) for c in row] for row in grid])


def fig_field_example() -> dict:
    data = np.load(DATA / "probes" / "field_example.npz")
    frame = data["frame"]
    true = data["true_mat"].astype(int)
    post, prior = data["post_probs"], data["prior_probs"]
    horizon = int(data["horizon"][0])
    station = np.isin(true, [CRAFTER_MATERIALS.index("table"), CRAFTER_MATERIALS.index("furnace")])
    station_xy = [(i // 7, i % 7) for i in np.flatnonzero(station)]

    fig, axes = plt.subplots(1, 5, figsize=(6.9, 1.75), gridspec_kw={"width_ratios": [1.0, 1.25, 1.25, 1.25, 1.45]})
    ax = axes[0]
    ax.imshow(frame, interpolation="nearest", extent=(0, 64, 64, 0))
    for k in range(10):
        ax.plot([7 * k, 7 * k], [0, 49], color="white", lw=0.35, alpha=0.8)
    for k in range(8):
        ax.plot([0, 63], [7 * k, 7 * k], color="white", lw=0.35, alpha=0.8)
    for k in range(5):
        ax.axvline(16 * k, color="#D55E00", lw=1.1)
        ax.axhline(16 * k, color="#D55E00", lw=1.1)
    ax.set_xlim(0, 64)
    ax.set_ylim(64, 0)
    ax.set_title("(a) Observation", pad=3)

    def panel(ax, codes, title):
        ax.imshow(_field_image(codes), interpolation="nearest", extent=(-0.5, 8.5, 6.5, -0.5))
        for x, y in station_xy:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, ec="#E69F00", lw=1.6))
        ax.plot(4, 3, marker="o", ms=2.5, color="white", mec="black", mew=0.4)
        ax.set_title(title, pad=3)

    panel(axes[1], true, "(b) True field")
    panel(axes[2], post.argmax(-1), "(c) Field head, $z^{\\mathrm{post}}$")
    panel(axes[3], prior.argmax(-1), f"(d) After {horizon} steps, $z^{{\\mathrm{{prior}}}}$")

    ax = axes[4]
    p_true = post[np.arange(true.size), true].reshape(9, 7).T
    im = ax.imshow(p_true, cmap="viridis", vmin=0, vmax=1, interpolation="nearest", extent=(-0.5, 8.5, 6.5, -0.5))
    for x, y in station_xy:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, ec="#E69F00", lw=1.6))
    ax.set_title("(e) $P$(true class), $z^{\\mathrm{post}}$", pad=3)
    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
    cbar.ax.tick_params(labelsize=6, length=2)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

    shown = sorted(set(true.tolist()) | set(post.argmax(-1).tolist()) | set(prior.argmax(-1).tolist()))
    handles = [plt.Rectangle((0, 0), 1, 1, fc=MATERIAL_COLORS[CRAFTER_MATERIALS[c]], ec="0.3", lw=0.3)
               for c in shown]
    fig.legend(handles, [CRAFTER_MATERIALS[c] for c in shown], loc="lower center", ncol=len(shown),
               frameon=False, fontsize=6.5, handlelength=1.0, columnspacing=1.0, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=0.6, rect=(0, 0.1, 1, 1))
    fig.savefig(OUT_FIG / "field_example.pdf")
    plt.close(fig)
    table = CRAFTER_MATERIALS.index("table")
    cells = true == table
    return {
        "post_table": float(post[cells, table].mean()) if cells.any() else float("nan"),
        "prior_table": float(prior[cells, table].mean()) if cells.any() else float("nan"),
        "post_acc": float((post.argmax(-1) == true).mean()),
        "prior_acc": float((prior.argmax(-1) == true).mean()),
    }


def fig_benchmark(bench: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.0), gridspec_kw={"width_ratios": [1.0, 1.15, 1.35]})

    ax = axes[0]
    for i, (slug, _pose, _win) in enumerate(BENCH):
        c = bench[slug]
        k, n = c["teacher_rare"], c["handoffs"]
        lo, hi = wilson(k, n)
        ax.bar(i, k / n, color=COLORS[i], width=0.65)
        ax.errorbar(i, k / n, yerr=[[k / n - lo], [hi - k / n]], color="black", capsize=2, lw=0.8)
        ax.text(i, hi + 0.02, f"{k}/{n}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(3), ["beside\nanch.", "beside\nunif.", "home\nanch."], fontsize=6.5)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("finish rate (95% CI)")
    ax.set_title("(a) Demonstrator after handoff")

    ax = axes[1]
    for i, (slug, _pose, _win) in enumerate(BENCH):
        c = bench[slug]
        xs = [s / 1000 for s, _k in c["held_stone"]]
        ax.plot(xs, [k for _s, k in c["held_stone"]], marker="o", ms=3, lw=1.1, color=COLORS[i],
                label=BENCH_LABEL[slug])
        ax.plot(xs, [k for _s, k in c["held_rare"]], marker="x", ms=4, lw=0, color=COLORS[i])
    ax.text(50, 8.6, "iron pickaxe (x): 0 at every checkpoint", ha="center", va="bottom", fontsize=6.2,
            color="0.3")
    ax.set_ylim(-0.4, 10)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("held-out lives / 10")
    ax.set_title("(b) Actor alone: stone pickaxe (o)")

    ax = axes[2]
    for i, (slug, _pose, _win) in enumerate(BENCH):
        rows, _spanning = load_lives(slug)
        stone = sorted(int(r["env_steps"]) for r in rows if r.get("achievement_make_stone_pickaxe", 0) > 0)
        ax.step([0] + [s / 1000 for s in stone], np.arange(0, len(stone) + 1), where="post", color=COLORS[i],
                lw=1.1, label=BENCH_LABEL[slug])
        rare = sorted(int(r["env_steps"]) for r in rows
                      if not r.get("assisted") and r.get("achievement_make_iron_pickaxe", 0) > 0)
        ax.step([0] + [s / 1000 for s in rare] + [100], [0] + list(range(1, len(rare) + 1)) + [len(rare)],
                where="post", color=COLORS[i], lw=1.1, ls=":")
    for k in (25, 50, 75):
        ax.axvline(k, color="0.85", lw=0.6, zorder=0)
    ax.text(99, 4, "dotted: actor-only rare craft (0 in every run)", fontsize=6, color="0.3", ha="right",
            va="bottom")
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 135)
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("cumulative training lives")
    ax.set_title("(c) Training: stone pickaxe (solid)")
    ax.legend(frameon=False, loc="upper left", fontsize=6.2)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT_FIG / "benchmark_runs.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- tex


def write_tables(bench: dict, room: dict, field: dict, room_probe: dict, start: dict) -> None:
    macros: list[str] = ["% Generated by paper/scripts/build_assets.py. Do not edit."]

    def m(name: str, value) -> None:
        macros.append(f"\\newcommand{{{name}}}{{{value}}}")

    m("\\StartLives", fmt_int(start["lives"]))
    m("\\StartRare", start["rare"])
    m("\\StartSteps", fmt_int(start["last_env"]))
    m("\\StartSpanning", start["spanning"])
    m("\\StartSpanningRare", start["spanning_rare"])
    m("\\BenchSpanning", sum(bench[s]["spanning"] for s, _p, _w in BENCH))
    m("\\BenchSpanningRare", sum(bench[s]["spanning_rare"] for s, _p, _w in BENCH))
    m("\\BenchSpanningShortest", min(bench[s]["spanning_short"] for s, _p, _w in BENCH
                                     if bench[s]["spanning_short"] is not None))
    m("\\FirstHandoffBeside", fmt_int(bench["m22_field_craft"]["first_handoff"]))
    m("\\EarlyStoneBeside", bench["m22_field_craft"]["early_stone"])
    m("\\EarlyRareBeside", bench["m22_field_craft"]["early_rare"])

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
    fig_benchmark(bench)
    example = fig_field_example()
    with open(OUT_GEN / "numbers.tex", "a", encoding="utf-8") as handle:
        handle.write(f"\\newcommand{{\\ExamplePostAcc}}{{{100 * example['post_acc']:.0f}}}\n")
        handle.write(f"\\newcommand{{\\ExamplePriorAcc}}{{{100 * example['prior_acc']:.0f}}}\n")
        handle.write(f"\\newcommand{{\\ExamplePostTable}}{{{example['post_table']:.2f}}}\n")
    summary = {
        "field_example": example,
        "start_checkpoint": start,
        "benchmark": {k: {kk: vv for kk, vv in v.items()} for k, v in bench.items()},
        "room": {k: {kk: vv for kk, vv in v.items() if kk != "lengths"} for k, v in room.items()},
    }
    (OUT_GEN / "numbers.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
