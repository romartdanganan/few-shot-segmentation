"""
scripts/stats_and_figures.py - every table, statistical test and figure in the
report, computed from the JSON files written by scripts/run_experiments.py.
Nothing is typed in by hand.

Unit of analysis: one TRAINING seed. Each final model's test score is its mean
over the evaluation seeds (identical episodes for every model), so methods are
paired by training seed (same training episodes, same test episodes).

Outputs (in <root>/report/):
    tuning_tables.md    parameter-tuning results on the validation classes
    final_table.md      test mIoU / F1, mean +/- std over training seeds
    stats.md, stats.json paired t-test + Wilcoxon, Holm-corrected, effect sizes
    cost_table.md       training time, best epoch, memory, parameters, ms/episode
    fig_val.png         validation curves (mean over training seeds, band = min-max)
    fig_box.png         per-training-seed test mIoU

Usage:
    python -m scripts.stats_and_figures            # reads experiments/
    python -m scripts.stats_and_figures --root experiments_smoke
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# Times-metric serif (template asks for Times New Roman in figure labels)
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8})

# Fixed colour per condition (never reassigned); marker + dash give a second encoding.
STYLE = {
    # Colours differ in brightness and every line has its own dash and marker,
    # so the figures still read correctly when printed in black and white.
    "baseline":           {"label": "Fine-tune", "color": "#1f3f77", "marker": "o", "ls": "-", "mfc": None},
    "prototype":          {"label": "Proto", "color": "#e07b28", "marker": "s", "ls": "--", "mfc": None},
    "prototype_weighted": {"label": "Proto-BW", "color": "#2e8b57", "marker": "^", "ls": ":", "mfc": "white"},
    "baseline_noTTA":     {"label": "Fine-tune, no TTA", "short": "No-TTA", "color": "#d4a017", "marker": "D",
                           "ls": "-.", "mfc": None},
}

# Planned comparisons (A - B). Each metric is its own Holm family.
COMPARISONS = [
    ("k1", "baseline", "prototype"),
    ("k5", "baseline", "prototype"),
    ("k5", "prototype", "prototype_weighted"),
    ("k5", "baseline", "baseline_noTTA"),
]


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def paired_tests(final, metric):
    rows = []
    for k, a, b in COMPARISONS:
        ra, rb = final.get(f"{k}_{a}"), final.get(f"{k}_{b}")
        if not ra or not rb:
            continue
        sa = {r["train_seed"]: r[metric] for r in ra}
        sb = {r["train_seed"]: r[metric] for r in rb}
        seeds = sorted(set(sa) & set(sb))
        if len(seeds) < 2:
            continue
        x = np.array([sa[s] for s in seeds])
        y = np.array([sb[s] for s in seeds])
        d = x - y
        t = stats.ttest_rel(x, y)
        try:
            w = stats.wilcoxon(x, y).pvalue
        except ValueError:
            w = float("nan")
        sd = d.std(ddof=1)
        rows.append({"k": k, "A": a, "B": b, "metric": metric, "n": len(seeds),
                     "mean_diff_A_minus_B": float(d.mean()), "t": float(t.statistic),
                     "p_t": float(t.pvalue), "p_wilcoxon_uncorrected": float(w),
                     "cohen_dz": float(d.mean() / sd) if sd > 0 else float("nan")})
    adj = holm([r["p_t"] for r in rows]) if rows else []
    for r, p in zip(rows, adj):
        r["p_holm"] = p
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments")
    args = ap.parse_args()
    root = Path(args.root)
    summary = json.loads((root / "summary.json").read_text())
    state = json.loads((root / "state.json").read_text())
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- tuning tables
    lines = ["# Parameter tuning (validation classes, training seed 0, k=5)", "",
             "Selection metric: mean validation mIoU over 3 evaluation seeds x 50 episodes.", ""]
    for method, stages in (summary.get("tuning") or {}).items():
        for key, st in stages.items():
            lines += [f"## {method}: {key} (chosen: {st['chosen']})", "",
                      "| value | val mIoU | ± std | val F1 |", "|---|---|---|---|"]
            for r in st["table"]:
                lines.append(f"| {r['value']} | {r['val_mIoU']:.4f} | {r['val_mIoU_std']:.4f} | {r['val_F1']:.4f} |")
            lines.append("")
    lines += ["Chosen settings: `" + json.dumps(summary.get("chosen")) + "`"]
    (out / "tuning_tables.md").write_text("\n".join(lines))

    # ---------------- final test table
    final = summary["final"]
    lines = ["# Test results (200 test classes)", "",
             "Mean ± std over training seeds; each seed's score is the mean over the evaluation seeds.", "",
             "| Condition | k | n seeds | mIoU | F1 | mean within-model eval-seed std (mIoU) |",
             "|---|---|---|---|---|---|"]
    for key in sorted(final):
        k, cond = key.split("_", 1)
        rs = final[key]
        mi = [r["mIoU"] for r in rs]
        f1 = [r["F1"] for r in rs]
        sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
        lines.append(f"| {STYLE.get(cond, {}).get('label', cond)} | {k[1:]} | {len(rs)} | "
                     f"{fmt(statistics.mean(mi), sd(mi))} | {fmt(statistics.mean(f1), sd(f1))} | "
                     f"{statistics.mean(r['eval_seed_mIoU_std'] for r in rs):.3f} |")
    lines += ["", "std = sample standard deviation across training seeds (ddof=1)."]
    (out / "final_table.md").write_text("\n".join(lines))

    # ---------------- statistics
    all_rows = paired_tests(final, "mIoU") + paired_tests(final, "F1")
    (out / "stats.json").write_text(json.dumps(all_rows, indent=2))
    lines = ["# Paired tests across training seeds (A minus B)", "",
             "Paired t-test (primary) and Wilcoxon signed-rank (check, uncorrected); "
             "Holm correction within each metric's family of comparisons; alpha = 0.05; "
             "Cohen's d_z = mean difference / sd of differences.", "",
             "| metric | k | A | B | n | diff (A-B) | t | p (t) | p Holm | p Wilcoxon | d_z |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in all_rows:
        lines.append(f"| {r['metric']} | {r['k'][1:]} | {STYLE[r['A']]['label']} | {STYLE[r['B']]['label']} | {r['n']} | "
                     f"{r['mean_diff_A_minus_B']:+.4f} | {r['t']:.2f} | {r['p_t']:.3g} | {r['p_holm']:.3g} | "
                     f"{r['p_wilcoxon_uncorrected']:.3g} | {r['cohen_dz']:+.2f} |")
    (out / "stats.md").write_text("\n".join(lines))

    # ---------------- cost table
    lines = ["# Computational cost", "",
             "| Condition | train time (min) | best epoch (per seed) | peak GPU MB | total params | "
             "ms/episode k=1 | ms/episode k=5 |", "|---|---|---|---|---|---|---|"]
    for cond, c in summary.get("cost", {}).items():
        tt = [t / 60 for t in c["train_time_s"]]
        mem = [m for m in c["peak_gpu_mem_MB"] if m is not None]
        ms = {k: statistics.mean(r["ms_per_episode"] for r in final.get(f"k{k}_{cond}", [])) if final.get(f"k{k}_{cond}") else float("nan")
              for k in (1, 5)}
        lines.append(f"| {STYLE[cond]['label']} | {statistics.mean(tt):.1f} ± {statistics.stdev(tt) if len(tt) > 1 else 0:.1f} | "
                     f"{c['best_epoch']} | {max(mem):.0f} | {c['total_params']:,} | {ms[1]:.0f} | {ms[5]:.0f} |"
                     if mem else
                     f"| {STYLE[cond]['label']} | {statistics.mean(tt):.1f} | {c['best_epoch']} | n/a | {c['total_params']:,} | {ms[1]:.0f} | {ms[5]:.0f} |")
    nt = final.get("k5_baseline_noTTA")
    if nt:
        lines.append(f"| {STYLE['baseline_noTTA']['label']} | (same model) | | | | "
                     f"{statistics.mean(r['ms_per_episode'] for r in final['k1_baseline_noTTA']):.0f} | "
                     f"{statistics.mean(r['ms_per_episode'] for r in nt):.0f} |")
    lines += ["", f"Hardware: {next(iter(summary.get('cost', {}).values()), {}).get('device')}"]
    (out / "cost_table.md").write_text("\n".join(lines))

    # ---------------- figure: validation curves (final configs, all training seeds)
    fig, ax = plt.subplots(figsize=(3.4, 2.3), dpi=300)
    for cond, dirs in state.get("final_runs", {}).items():
        curves = []
        for d in dirs:
            # Paths are stored as written on the machine that ran them (may use "\\").
            h = Path(str(d).replace("\\", "/")) / "history.json"
            if h.exists():
                curves.append([e["val_mIoU"] for e in json.loads(h.read_text())["epochs"]])
        if not curves:
            continue
        n = min(len(c) for c in curves)
        arr = np.array([c[:n] for c in curves])
        x = np.arange(1, n + 1)
        st = STYLE[cond]
        ax.plot(x, arr.mean(0), marker=st["marker"], ms=3.5, lw=1.2, ls=st["ls"], color=st["color"],
                mfc=st["mfc"] or st["color"], mec=st["color"], label=f"{st['label']} (n={len(curves)})")
        if len(curves) > 1:
            ax.fill_between(x, arr.min(0), arr.max(0), color=st["color"], alpha=0.10, lw=0)
    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel("Validation mIoU")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_val.png")
    plt.close(fig)

    # ---------------- figure: per-training-seed test mIoU
    order = [("k1", "baseline"), ("k1", "prototype"), ("k5", "baseline_noTTA"), ("k5", "baseline"),
             ("k5", "prototype"), ("k5", "prototype_weighted")]
    data, labels, styles = [], [], []
    for k, cond in order:
        rs = final.get(f"{k}_{cond}")
        if rs:
            data.append([r["mIoU"] for r in rs])
            labels.append(f"{STYLE[cond].get('short', STYLE[cond]['label'])}\n{k[1:]}-shot")
            styles.append(STYLE[cond])
    if data:
        fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=300)
        ax.boxplot(data, tick_labels=labels, widths=0.5, showfliers=False, medianprops=dict(color="black", lw=1.2))
        for i, (vals, st) in enumerate(zip(data, styles), start=1):
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=9, marker=st["marker"], zorder=3,
                       facecolors=st["mfc"] or st["color"], edgecolors=st["color"], linewidths=0.6)
        ax.set_ylabel("Test mIoU per training seed")
        ax.tick_params(axis="x", labelsize=6)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "fig_box.png")
        plt.close(fig)

    print(f"Wrote tables, stats and figures to {out}")
    for f in ["tuning_tables.md", "final_table.md", "stats.md", "cost_table.md"]:
        print("\n" + (out / f).read_text())


if __name__ == "__main__":
    main()
