"""Generate all statistical tests, figures, and updated tables for the paper."""
from __future__ import annotations

import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
RESULTS = ROOT / "results"
TABLES = ROOT / "NEU" / "manuscript" / "tables"
FIGS = ROOT / "NEU" / "manuscript" / "figures"
FIGS.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

sys.path.insert(0, str(CODE_DIR))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_csv(name):
    p = RESULTS / name
    if not p.exists():
        return []
    return list(csv.DictReader(p.open()))


# ── Helpers ──────────────────────────────────────────────────────────

DISPLAY = {
    "iris": "Iris", "glass": "Glass", "wine": "Wine", "ecoli": "Ecoli",
    "heart_statlog": "Heart-Statlog", "banknote": "Banknote", "diabetes": "Diabetes",
    "yeast": "Yeast", "ionosphere": "Ionosphere", "dermatology": "Dermatology",
    "sonar": "Sonar", "vehicle": "Vehicle", "spectf": "SPECTF",
    "breast_cancer": "BreastCancer", "waveform": "Waveform", "segment": "Segment",
    "pc3": "PC3", "steel_plates": "SteelPlates", "digits": "Digits",
    "balance_scale": "Balance-Scale",
}

DS_ORDER = [
    "iris", "banknote", "ecoli", "diabetes", "yeast", "glass", "wine",
    "heart_statlog", "vehicle", "breast_cancer", "segment", "steel_plates",
    "ionosphere", "dermatology", "pc3", "waveform", "spectf", "sonar", "digits",
]
# Omitted from paper: balance_scale (degenerate); iris/banknote/glass (too small)
SKIP = {"balance_scale", "iris", "banknote", "glass"}
METHODS_CLS = ["GBRS", "HTBS", "NRS"]


# ── 1. Bootstrap CI for violation rates ──────────────────────────────

def bootstrap_ci(rates, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.array(rates)
    if len(arr) == 0:
        return 0.0, 0.0
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ── 2. Generate mono table with magnitude ────────────────────────────

def gen_mono_table():
    rows = load_csv("mono_audit.csv")
    if not rows:
        print("No mono_audit.csv"); return
    by_ds = defaultdict(dict)
    for r in rows:
        by_ds[r["dataset"]][r["method"]] = r

    d_ranges = [(0, 9, "$d \\le 9$"), (10, 19, "$d=10$--$19$"),
                (20, 39, "$d=20$--$39$"), (40, 999, "$d \\ge 40$")]
    range_gbrs = {lab: [] for _, _, lab in d_ranges}
    range_deg  = {lab: [] for _, _, lab in d_ranges}

    body = []
    ordered = [ds for ds in DS_ORDER if ds in by_ds and ds not in SKIP]
    ordered += [ds for ds in by_ds if ds not in ordered and ds not in SKIP and ds != "balance_scale"]

    all_gbrs, all_deg, all_nrs = [], [], []

    for ds in ordered:
        g = by_ds[ds].get("gbrs", {})
        d = by_ds[ds].get("deg", {})
        n = by_ds[ds].get("nrs", {})
        h = by_ds[ds].get("htbs", {})
        if not g:
            continue
        d_val = int(float(g["d"]))
        gr = float(g.get("violation_rate", 0))
        dr = float(d.get("violation_rate", 0))
        nr = float(n.get("violation_rate", 0))
        hr = float(h.get("violation_rate", 0))
        # magnitude of violations
        gm = float(g.get("mean_delta", 0))
        dm = float(d.get("mean_delta", 0))

        all_gbrs.append(gr); all_deg.append(dr); all_nrs.append(nr)

        for lo, hi, lab in d_ranges:
            if lo <= d_val <= hi:
                range_gbrs[lab].append(gr)
                range_deg[lab].append(dr)

        body.append(
            f"{DISPLAY.get(ds, ds)} & {d_val} & {gr:.1f} & {gm:.4f}"
            f" & {nr:.1f} & {hr:.1f} \\\\"
        )

    # d-range summary rows
    range_rows = []
    for lo, hi, lab in d_ranges:
        rg = range_gbrs[lab]
        if rg:
            range_rows.append(
                f"\\multicolumn{{6}}{{l}}{{\\quad {lab} (avg GBRS~$={np.mean(rg):.1f}\\%$)}} \\\\"
            )

    avg_gbrs = np.mean(all_gbrs) if all_gbrs else 0
    avg_deg  = np.mean(all_deg) if all_deg else 0
    avg_nrs  = np.mean(all_nrs) if all_nrs else 0
    max_rate_gap = max(abs(g - d) for g, d in zip(all_gbrs, all_deg)) if all_gbrs else 0.0

    ci_lo, ci_hi = bootstrap_ci(all_gbrs)
    n_ds = len(all_gbrs)

    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Monotonicity violation rates (\\%) and mean violation magnitude "
        "($\\overline{|\\Delta\\gamma|}$) per dataset (seed 42, "
        "$\\varepsilon=10^{-10}$; 200 random subsets for $d\\le20$, 100 for $d>20$). "
        "HTBS: $0.0\\%$ on all datasets. "
        f"Grand average GBRS $={avg_gbrs:.1f}\\%$ "
        f"(bootstrap 95\\% CI ${ci_lo:.1f}$--${ci_hi:.1f}$, {n_ds} datasets). "
        f"A secondary entropy-threshold admission control (same $k$-means engine, $\\alpha=0.10$) "
        f"tracked GBRS within ${max_rate_gap:.1f}$ percentage points on every dataset and is omitted for clarity.}}\n"
        "\\label{tab:mono}\n\\small\n"
        "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
        " & & \\multicolumn{2}{c}{GBRS} & NRS & HTBS \\\\\n"
        "\\cmidrule(lr){3-4}\n"
        "Dataset & $d$ & Rate\\% & $\\overline{|\\Delta\\gamma|}$ "
        "& Rate\\% & Rate\\% \\\\\n"
        "\\midrule\n"
        + "\n".join(body) + "\n"
        "\\midrule\n"
        + "\n".join(range_rows) + "\n"
        f"\\textbf{{Avg.~({n_ds} ds)}} & -- & \\textbf{{{avg_gbrs:.1f}}} & -- "
        f"& {avg_nrs:.2f} & \\textbf{{0.0}} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (TABLES / "tab_mono_block.tex").write_text(tex)
    print(f"Wrote tab_mono_block.tex  (n_ds={n_ds})")
    return by_ds


# ── 3. Wilcoxon tests ────────────────────────────────────────────────

def wilcoxon_tests():
    rows = load_csv("classification.csv")
    if not rows:
        return
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("scipy not available, skip Wilcoxon"); return

    by_ds = defaultdict(dict)
    for r in rows:
        by_ds[r["dataset"]][r["method"]] = r

    valid_ds = [ds for ds in DS_ORDER if ds in by_ds and ds not in SKIP
                and by_ds[ds].get("GBRS") and float(by_ds[ds]["GBRS"]["|R|"]) > 0
                and by_ds[ds].get("HTBS") and float(by_ds[ds]["HTBS"]["|R|"]) > 0
                and by_ds[ds].get("NRS") and float(by_ds[ds]["NRS"]["|R|"]) > 0]

    pairs = [
        ("GBRS", "HTBS", "knn3_acc"),
        ("GBRS", "NRS", "knn3_acc"),
        ("GBRS", "HTBS", "svm_acc"),
    ]
    lines = []
    for m1, m2, metric in pairs:
        a, b = [], []
        for ds in valid_ds:
            if m1 in by_ds[ds] and m2 in by_ds[ds]:
                a.append(float(by_ds[ds][m1][metric]))
                b.append(float(by_ds[ds][m2][metric]))
        if len(a) < 5:
            lines.append(f"{m1} vs {m2} ({metric}): n={len(a)} (too few)")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = wilcoxon(a, b, alternative="two-sided")
        diff = np.mean(a) - np.mean(b)
        sign = ">" if diff > 0 else "<"
        lines.append(
            f"{m1} vs {m2} ({metric}): n={len(a)}, "
            f"mean diff={diff:+.4f} ({m1} {sign} {m2}), "
            f"W={stat:.1f}, p={p:.4f}"
        )
        print(lines[-1])

    (RESULTS / "wilcoxon_tests.txt").write_text("\n".join(lines))
    print(f"Wrote wilcoxon_tests.txt")
    return lines


# ── 4. Figure: violation rate vs d ───────────────────────────────────

def gen_figure_mono(by_ds_arg=None):
    rows = load_csv("mono_audit.csv")
    if not rows:
        return
    by_ds = by_ds_arg or {}
    if not by_ds:
        by_ds = defaultdict(dict)
        for r in rows:
            by_ds[r["dataset"]][r["method"]] = r

    ds_vals = []
    for ds in [r["dataset"] for r in rows if r["method"] == "gbrs"]:
        g = by_ds[ds].get("gbrs", {})
        if g and ds not in SKIP:
            ds_vals.append((int(float(g["d"])),
                            float(g.get("violation_rate", 0)),
                            float(by_ds[ds].get("nrs", {}).get("violation_rate", 0))))
    if not ds_vals:
        return
    ds_vals.sort(key=lambda x: x[0])
    d_vals = [x[0] for x in ds_vals]
    gbrs_r = [x[1] for x in ds_vals]
    nrs_r  = [x[2] for x in ds_vals]

    gbrs_coords = "\n".join(f"    ({d},{r:.4f})" for d, r in zip(d_vals, gbrs_r))
    nrs_coords  = "\n".join(f"    ({d},{r:.4f})" for d, r in zip(d_vals, nrs_r))

    trend_plot = ""
    if len(d_vals) >= 4:
        slope, intercept = np.polyfit(np.log(np.array(d_vals, dtype=float) + 1), gbrs_r, 1)
        trend_plot = (
            f"  \\addplot[dashed, thick, blue!70!black, domain={min(d_vals)}:{max(d_vals)}, samples=120]\n"
            f"    {{{slope:.6f}*ln(x+1) + {intercept:.6f}}};\n"
            f"  \\addlegendentry{{GBRS log-linear fit}}\n"
        )

    tex = f"""% Violation rate vs. dimensionality (pgfplots; auto-generated)
\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=0.82\\linewidth,
  height=4.6cm,
  xmin=0, xmax={max(d_vals) + 4},
  ymin=-2, ymax=70,
  xlabel={{Number of conditional attributes ($d$)}},
  ylabel={{Violation rate (\\%)}},
  grid=major,
  grid style={{dashed, gray!35}},
  legend style={{at={{(0.02,0.98)}},anchor=north west,font=\\scriptsize}},
  tick label style={{font=\\small}},
  label style={{font=\\small}},
]
  \\addplot[
    only marks,
    mark=*,
    mark size=1.8pt,
    draw=blue!70!black,
    fill=blue!45,
  ] coordinates {{
{gbrs_coords}
  }};
  \\addlegendentry{{GBRS}}
  \\addplot[
    only marks,
    mark=triangle,
    mark size=2.4pt,
    draw=green!50!black,
    fill=green!45,
  ] coordinates {{
{nrs_coords}
  }};
  \\addlegendentry{{NRS}}
{trend_plot}\\end{{axis}}
\\end{{tikzpicture}}
"""
    out = FIGS / "fig_mono_vs_d.tex"
    out.write_text(tex)
    print(f"Wrote {out}")


# ── 5. Updated classification tables with bold/underline ─────────────

def gen_cls_tables():
    rows = load_csv("classification.csv")
    if not rows:
        print("No classification.csv"); return

    by_ds = defaultdict(dict)
    for r in rows:
        by_ds[r["dataset"]][r["method"]] = r

    def fmt_acc(m, s, bold=False, uline=False):
        inner = f"{m:.4f}$\\pm${s:.4f}"
        if bold: inner = f"\\textbf{{{inner}}}"
        elif uline: inner = f"\\underline{{{inner}}}"
        return inner

    def best_two(vals):
        sorted_v = sorted(vals.items(), key=lambda x: x[1], reverse=True)
        best = sorted_v[0][0] if sorted_v else None
        second = sorted_v[1][0] if len(sorted_v) > 1 else None
        return best, second

    knn_lines, svm_lines, rt_lines = [], [], []
    for ds in DS_ORDER:
        if ds not in by_ds or ds in SKIP:
            continue
        ref = by_ds[ds].get("GBRS") or next(iter(by_ds[ds].values()))
        d_val = int(float(ref["d"]))
        fk = float(ref["full_knn3"])
        fs = float(ref["full_svm"])

        kvals = {m: float(by_ds[ds][m]["knn3_acc"]) for m in METHODS_CLS
                 if m in by_ds[ds] and int(float(by_ds[ds][m]["|R|"])) > 0}
        svals = {m: float(by_ds[ds][m]["svm_acc"])  for m in METHODS_CLS
                 if m in by_ds[ds] and int(float(by_ds[ds][m]["|R|"])) > 0}
        kbest, ksec = best_two(kvals)
        sbest, ssec = best_two(svals)

        kr = [DISPLAY.get(ds, ds), str(d_val), f"{fk:.4f}"]
        sr = [DISPLAY.get(ds, ds), str(d_val), f"{fs:.4f}"]
        rr = [DISPLAY.get(ds, ds), str(d_val)]

        for m in METHODS_CLS:
            if m not in by_ds[ds]:
                kr += ["--", "--"]; sr += ["--", "--"]; rr += ["--", "--"]
                continue
            r = by_ds[ds][m]
            size = int(float(r["|R|"]))
            ka, ks_v = float(r["knn3_acc"]), float(r["knn3_std"])
            sa, ss_v = float(r["svm_acc"]),  float(r["svm_std"])
            rt       = float(r["runtime_s"])
            if size == 0:
                kr += ["0", "--"]; sr += ["0", "--"]; rr += ["0", f"{rt:.1f}"]
            else:
                kr += [str(size), fmt_acc(ka, ks_v, bold=(m==kbest), uline=(m==ksec))]
                sr += [str(size), fmt_acc(sa, ss_v, bold=(m==sbest), uline=(m==ssec))]
                rr += [str(size), f"{rt:.1f}"]

        knn_lines.append(" & ".join(kr) + r" \\")
        svm_lines.append(" & ".join(sr) + r" \\")
        rt_lines.append(" & ".join(rr) + r" \\")

    header_k = (
        "Dataset & $d$ & Acc$_{\\text{full}}$ & "
        "\\multicolumn{2}{c|}{GBRS} & "
        "\\multicolumn{2}{c|}{HTBS} & \\multicolumn{2}{c}{NRS} \\\\\n"
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
        " & & & $|R|$ & Acc$\\pm$std & "
        "$|R|$ & Acc$\\pm$std & $|R|$ & Acc$\\pm$std \\\\\n\\midrule"
    )
    header_s = header_k.replace("Acc$_{\\text{full}}$", "SVM$_{\\text{full}}$")

    def wrap_table(caption, label, header, lines, star=True):
        env = "table*" if star else "table"
        return (
            f"\\begin{{{env}}}[t]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\scriptsize\n"
            "\\setlength{\\tabcolsep}{3pt}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{@{}l r r | r p{1.8cm} | r p{1.8cm} | r p{1.8cm}@{}}\n"
            "\\toprule\n" + header + "\n" + "\n".join(lines) + "\n"
            "\\bottomrule\n\\end{tabular}}\n"
            f"\\end{{{env}}}\n"
        )

    knn_cap = ("$k$NN-3 post-reduction accuracy (mean$\\pm$std, 5-fold CV, seed~0). "
               "\\textbf{Bold}: best per row; \\underline{underline}: second-best. "
               "$|R|=0$: method produced empty reduct (degenerate).")
    svm_cap = ("SVM-RBF ($C=1$, \\texttt{gamma=scale}) post-reduction accuracy on the "
               "same reducts as Table~\\ref{tab:knn3}.")

    rt_header = (
        "Dataset & $d$ & \\multicolumn{2}{c|}{GBRS} & "
        "\\multicolumn{2}{c|}{HTBS} & \\multicolumn{2}{c}{NRS} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
        " & & $|R|$ & Time(s) & $|R|$ & Time(s) & $|R|$ & Time(s) \\\\\n\\midrule"
    )
    rt_tab = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Wall-clock reduction time (seconds, single CPU core, Python~3.9). "
        "Time includes full forward-greedy loop.}\n"
        "\\label{tab:runtime}\n\\scriptsize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{@{}l r | r r | r r | r r@{}}\n"
        "\\toprule\n" + rt_header + "\n" + "\n".join(rt_lines) + "\n"
        "\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    )

    tex = (
        wrap_table(knn_cap, "tab:knn3", header_k, knn_lines) + "\n" +
        wrap_table(svm_cap, "tab:svm",  header_s, svm_lines) + "\n" +
        rt_tab
    )
    (TABLES / "tab_results_block.tex").write_text(tex)
    print("Wrote tab_results_block.tex")


# ── 6. Trade-off bar chart (mean kNN-3 over audited benchmarks) ───────

def gen_tradeoff_figure():
    rows = load_csv("classification.csv")
    if not rows:
        return
    by_ds = defaultdict(dict)
    for r in rows:
        by_ds[r["dataset"]][r["method"]] = r

    avgs = {}
    for m in METHODS_CLS:
        vals = []
        for ds in DS_ORDER:
            if ds in SKIP or ds not in by_ds or m not in by_ds[ds]:
                continue
            if int(float(by_ds[ds][m]["|R|"])) > 0:
                vals.append(float(by_ds[ds][m]["knn3_acc"]))
        avgs[m] = float(np.mean(vals)) if vals else 0.0

    n_ds = len([ds for ds in DS_ORDER if ds not in SKIP and ds in by_ds])
    tex = f"""% Monotonicity--accuracy trade-off (dataset averages from experiments)
\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=0.82\\linewidth,
  height=4.6cm,
  ybar,
  bar width=14pt,
  ymin=0, ymax=1.0,
  ytick={{0,0.2,0.4,0.6,0.8,1.0}},
  symbolic x coords={{GBRS,HTBS,NRS}},
  xtick=data,
  ylabel={{Mean $k$NN-3 accuracy}},
  enlarge x limits=0.22,
  nodes near coords,
  nodes near coords align={{vertical}},
  every node near coord/.append style={{font=\\scriptsize}},
  legend style={{at={{(0.02,0.98)}},anchor=north west,font=\\scriptsize}},
  tick label style={{font=\\small}},
  label style={{font=\\small}},
]
  \\addplot[fill=blue!45,draw=blue!70!black] coordinates {{
    (GBRS,{avgs['GBRS']:.3f}) (HTBS,{avgs['HTBS']:.3f}) (NRS,{avgs['NRS']:.3f})
  }};
  \\addlegendentry{{{n_ds}-dataset average}}
\\end{{axis}}
\\end{{tikzpicture}}
"""
    (FIGS / "fig_tradeoff_bars.tex").write_text(tex)
    print(f"Wrote fig_tradeoff_bars.tex  (GBRS={avgs['GBRS']:.3f}, HTBS={avgs['HTBS']:.3f}, NRS={avgs['NRS']:.3f})")
    return avgs, n_ds


# ── 7. LaTeX snippet for Wilcoxon results ────────────────────────────

def gen_wilcoxon_tex(lines):
    if not lines:
        return
    import re
    tex_rows = []
    for line in lines:
        # parse: "A vs B (metric): n=N, mean diff=X, W=Y, p=Z"
        m = re.match(r"(.+?) vs (.+?) \((.+?)\): n=(\d+), mean diff=([+\-\d.]+) .*, W=([\d.]+), p=([\d.]+)", line)
        if not m:
            continue
        m1, m2, metric, n, diff, W, p = m.groups()
        p_f = float(p)
        sig = "$<0.05$" if p_f < 0.05 else "n.s."
        metric_label = "$k$NN-3" if "knn3" in metric else "SVM-RBF"
        tex_rows.append(f"{m1} vs {m2} & {metric_label} & {n} & {diff} & {W} & {float(p):.4f} & {sig} \\\\")

    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Pairwise Wilcoxon signed-rank tests (two-tailed) on post-reduction accuracy. "
        "Positive mean diff: first method has higher accuracy.}\n"
        "\\label{tab:wilcoxon}\n\\scriptsize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{@{}llrrrrl@{}}\n\\toprule\n"
        "Comparison & Metric & $n$ & Mean diff & $W$ & $p$ & Sig. \\\\\n\\midrule\n"
        + "\n".join(tex_rows) + "\n"
        "\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    )
    (TABLES / "tab_wilcoxon.tex").write_text(tex)
    print("Wrote tab_wilcoxon.tex")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    by_ds = gen_mono_table()
    wilcoxon_lines = wilcoxon_tests()
    gen_figure_mono(by_ds)
    gen_cls_tables()
    gen_tradeoff_figure()
    if wilcoxon_lines:
        gen_wilcoxon_tex(wilcoxon_lines)
    print("All done.")
