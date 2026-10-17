"""Generate LaTeX table blocks for main.tex."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "NEU" / "manuscript" / "tables"


def fmt_acc(mean: float, std: float) -> str:
    return f"{mean:.4f}$\\pm${std:.4f}"


def load_csv(name: str):
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


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
    "iris", "banknote", "ecoli", "diabetes", "yeast", "glass", "wine", "vehicle",
    "segment", "breast_cancer", "steel_plates", "ionosphere", "dermatology", "pc3",
    "waveform", "spectf", "sonar", "digits", "balance_scale",
]

METHODS = ["GBRS", "DEG-GB", "HTBS", "NRS"]


def gen_mono_block():
    rows = load_csv("mono_audit.csv")
    if not rows:
        OUT.mkdir(exist_ok=True)
        (OUT / "tab_mono_block.tex").write_text(
            "% Run code/generate_tables.py after mono_audit.csv exists\n"
        )
        return
    by_ds = defaultdict(dict)
    for r in rows:
        by_ds[r["dataset"]][r["method"]] = r

    body = []
    gbrs_rates, deg_rates, nrs_rates = [], [], []
    display_order = [
        "iris", "banknote", "ecoli", "diabetes", "yeast", "glass", "wine",
        "heart_statlog", "vehicle", "segment", "breast_cancer", "steel_plates",
        "ionosphere", "dermatology", "pc3", "waveform", "spectf", "sonar", "digits",
    ]
    ordered = [ds for ds in display_order if ds in by_ds]
    ordered += [ds for ds in by_ds if ds not in ordered and ds != "balance_scale"]

    for ds in ordered:
        g = by_ds[ds].get("gbrs", {})
        d = by_ds[ds].get("deg", {})
        n = by_ds[ds].get("nrs", {})
        if not g:
            continue
        gr = float(g.get("violation_rate_mean", g.get("violation_rate", 0)))
        dr = float(d.get("violation_rate_mean", d.get("violation_rate", 0)))
        nr = float(n.get("violation_rate_mean", n.get("violation_rate", 0)))
        gbrs_rates.append(gr)
        deg_rates.append(dr)
        nrs_rates.append(nr)
        body.append(
            f"{DISPLAY.get(ds, ds)} & {int(float(g['d']))} & {gr:.1f} & {dr:.1f} & {nr:.1f} \\\\"
        )
    if gbrs_rates:
        body.append(
            f"\\midrule\n\\textbf{{Avg. ({len(gbrs_rates)} datasets)}} & -- & "
            f"{sum(gbrs_rates)/len(gbrs_rates):.1f} & "
            f"{sum(deg_rates)/len(deg_rates):.1f} & "
            f"{sum(nrs_rates)/len(nrs_rates):.1f} \\\\"
        )

    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Monotonicity violation rates (\\%). "
        "Two hundred random subsets per dataset ($100$ when $d>20$); all remaining attributes tested; "
        "$\\varepsilon=10^{-10}$. HTBS: $0.0\\%$ on every audited dataset (incremental extension).}\n"
        "\\label{tab:mono}\n\\small\n"
        "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
        "Dataset & $d$ & GBRS & DEG-GB & NRS \\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "tab_mono_block.tex").write_text(tex)
    print("Wrote tab_mono_block.tex")


def gen_results_block():
    rows = load_csv("classification_rerun.csv") or load_csv("classification.csv")
    if not rows:
        OUT.mkdir(exist_ok=True)
        (OUT / "tab_results_block.tex").write_text("% awaiting classification.csv\n")
        return

    by_ds = defaultdict(dict)
    for r in rows:
        by_ds[r["dataset"]][r["method"]] = r

    knn_lines, svm_lines, rt_lines = [], [], []
    for ds in DS_ORDER:
        if ds not in by_ds:
            continue
        ref = by_ds[ds].get("GBRS") or next(iter(by_ds[ds].values()))
        d_val = int(float(ref["d"]))
        fk = float(ref["full_knn3"])
        fs = float(ref["full_svm"])
        kr = [DISPLAY.get(ds, ds), str(d_val), f"{fk:.4f}"]
        sr = [DISPLAY.get(ds, ds), str(d_val), f"{fs:.4f}"]
        rr = [DISPLAY.get(ds, ds), str(d_val)]
        for m in METHODS:
            if m not in by_ds[ds]:
                kr += ["--", "--"]
                sr += ["--", "--"]
                rr += ["--", "--"]
                continue
            r = by_ds[ds][m]
            kr += [str(int(float(r["|R|"]))), fmt_acc(float(r["knn3_acc"]), float(r["knn3_std"]))]
            sr += [str(int(float(r["|R|"]))), fmt_acc(float(r["svm_acc"]), float(r["svm_std"]))]
            rr += [str(int(float(r["|R|"]))), f"{float(r['runtime_s']):.1f}"]
        knn_lines.append(" & ".join(kr) + r" \\")
        svm_lines.append(" & ".join(sr) + r" \\")
        rt_lines.append(" & ".join(rr) + r" \\")

    colspec = "l r | " + " r p{2.1cm} | " * 3 + " r p{2.1cm}"
    header = (
        "Dataset & $d$ & Acc$_{\\text{full}}$ & "
        "\\multicolumn{2}{c|}{GBRS} & \\multicolumn{2}{c|}{DEG-GB} & "
        "\\multicolumn{2}{c|}{HTBS} & \\multicolumn{2}{c}{NRS} \\\\"
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}"
        " & & & $|R|$ & Acc$\\pm$std & $|R|$ & Acc$\\pm$std & $|R|$ & Acc$\\pm$std & $|R|$ & Acc$\\pm$std \\\\"
        "\\midrule"
    )

    svm_header = header.replace("Acc$_{\\text{full}}$", "SVM$_{\\text{full}}$")

    tex = (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{$k$NN-3 post-reduction accuracy (mean$\\pm$std, 5-fold CV).}\n"
        "\\label{tab:knn3}\n\\footnotesize\n"
        "\\begin{tabular}{@{}l r | r | r p{1.9cm} | r p{1.9cm} | r p{1.9cm} | r p{1.9cm}@{}}\n"
        "\\toprule\n" + header + "\n" + "\n".join(knn_lines) + "\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table*}\n\n"
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{SVM-RBF post-reduction accuracy (mean$\\pm$std, 5-fold CV) on the same reducts as Table~\\ref{tab:knn3}.}\n"
        "\\label{tab:svm}\n\\footnotesize\n"
        "\\begin{tabular}{@{}l r | r | r p{1.9cm} | r p{1.9cm} | r p{1.9cm} | r p{1.9cm}@{}}\n"
        "\\toprule\n" + svm_header + "\n" + "\n".join(svm_lines) + "\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table*}\n\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Wall-clock forward-greedy reduction time (seconds, single CPU core).}\n"
        "\\label{tab:runtime}\n\\small\n"
        "\\begin{tabular}{@{}l r | r r | r r | r r | r r@{}}\n"
        "\\toprule\n"
        "Dataset & $d$ & \\multicolumn{2}{c|}{GBRS} & \\multicolumn{2}{c|}{DEG-GB} & "
        "\\multicolumn{2}{c|}{HTBS} & \\multicolumn{2}{c}{NRS} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
        " & & $|R|$ & Time & $|R|$ & Time & $|R|$ & Time & $|R|$ & Time \\\\\n"
        "\\midrule\n" + "\n".join(rt_lines) + "\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}\n"
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "tab_results_block.tex").write_text(tex)
    print("Wrote tab_results_block.tex")


if __name__ == "__main__":
    gen_mono_block()
    gen_results_block()
