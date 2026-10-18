"""Write the LaTeX violation-rate table (Table 5) from the raw audit CSVs.

Usage: python3 make_mono_table.py <output .tex path>
Sources: results/gbrs_official_audit_full.csv (generator of the released GBRS code at the
selected minimum ball size; seed 42), results/mono_audit.csv (our size-adaptive generator
GBRS-A, NRS, HTBS; seed 42) and results/nmin_fixed_audit_16sets.csv (our generator with
n_min fixed at 2 and 5; seed 42).
"""
import sys
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
m = pd.read_csv(R / "mono_audit.csv")
n = pd.read_csv(R / "nmin_fixed_audit_16sets.csv")
m = m[~m.dataset.isin(["iris", "glass", "banknote", "balance_scale"])]
g = m[m.method == "gbrs"].set_index("dataset")
r = m[m.method == "nrs"].set_index("dataset")
nn = n[n.method == "gbrs"]
a2 = nn[nn.config == "nmin2"].set_index("dataset")
a5 = nn[nn.config == "nmin5"].set_index("dataset")
assert m[m.method == "htbs"].violation_rate.abs().max() == 0
of = pd.read_csv(R / "gbrs_official_audit_full.csv")
assert of.error.isna().all()
rel = of.groupby("dataset").agg(pairs=("pairs", "sum"), viol=("violations", "sum"), sdrop=("sum_drop", "sum"))
rel["rate"] = 100 * rel.viol / rel.pairs
rel["mdrop"] = rel.sdrop / rel.viol

NAMES = {"ecoli": "Ecoli", "diabetes": "Diabetes", "yeast": "Yeast", "wine": "Wine",
         "heart_statlog": "Heart-Statlog", "vehicle": "Vehicle", "segment": "Segment",
         "breast_cancer": "BreastCancer", "steel_plates": "SteelPlates",
         "ionosphere": "Ionosphere", "dermatology": "Dermatology", "pc3": "PC3",
         "waveform": "Waveform", "spectf": "SPECTF", "sonar": "Sonar", "digits": "Digits"}
order = sorted(NAMES, key=lambda k: (int(g.loc[k, "d"]), list(NAMES).index(k)))
rows = [f"{NAMES[k]} & {int(g.loc[k, 'd'])} & {rel.loc[k, 'rate']:.1f} & {rel.loc[k, 'mdrop']:.3f} & "
        f"{g.loc[k, 'violation_rate']:.1f} & "
        f"{a2.loc[k, 'violation_rate']:.1f} & {a5.loc[k, 'violation_rate']:.1f} & "
        f"{r.loc[k, 'violation_rate']:.1f}" for k in order]
hdr = (" & & \\multicolumn{2}{c}{GBRS} & \\multicolumn{3}{c}{GBRS-A, $\\nmin$} & NRS & & & \\multicolumn{2}{c}{GBRS} & \\multicolumn{3}{c}{GBRS-A, $\\nmin$} & NRS \\\\\n"
       "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\\cmidrule(lr){11-12}\\cmidrule(lr){13-15}\n"
       "Dataset & $d$ & Rate & $\\overline{|\\Delta\\gamma|}$ & Adapt. & 2 & 5 & Rate"
       " & Dataset & $d$ & Rate & $\\overline{|\\Delta\\gamma|}$ & Adapt. & 2 & 5 & Rate \\\\\n")
body = "\n".join(f"{l} & {rr} \\\\" for l, rr in zip(rows[:8], rows[8:]))
avg = (f"\\multicolumn{{16}}{{c}}{{\\textit{{Average over 16 datasets:}} GBRS {rel.rate.mean():.1f} "
       f"({rel.mdrop.mean():.3f}); GBRS-A {g.violation_rate.mean():.1f}, {a2.violation_rate.mean():.1f}, "
       f"{a5.violation_rate.mean():.1f}; NRS {r.violation_rate.mean():.1f}}} \\\\")
tex = ("\\begin{table*}[pos=!htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{2.6pt}\n"
       "\\caption{Violation rate (\\%) of the random-subset audit, ordered by $d$. GBRS uses the generator of the released code "
       "at the minimum ball size that its pipeline selects on each dataset, and $\\overline{|\\Delta\\gamma|}$ is the mean size of its "
       "violations. GBRS-A uses our generator with the size-adaptive minimum ball size (Adapt.) and with $\\nmin$ fixed at 2 and 5. "
       "HTBS had no violations on any dataset.}\n"
       "\\label{tab:mono}\n"
       "\\begin{tabular*}{\\linewidth}{@{}l @{\\hspace{4pt}\\extracolsep{\\fill}} r r r r r r r @{\\hspace{0.8em}} l @{\\hspace{4pt}} r r r r r r r@{}}\n"
       "\\toprule\n" + hdr + "\\midrule\n" + body + "\n\\midrule\n" + avg + "\n\\bottomrule\n\\end{tabular*}\n"
       "\\tabnote{%\nSeed 42, $\\varepsilon=10^{-10}$; 200 random subsets for $d\\le20$ and 100 for $d>20$, drawn in the same way for every "
       "column. For GBRS the audited quantity is the number of instances in pure balls, as in the released code, and SteelPlates uses the "
       "25 attributes of the classification tables; the other columns keep all 33. The $95\\%$ bootstrap interval of the GBRS-A adaptive "
       "average is $41.1$--$50.5$, and its mean violation size is $" + f"{g.mean_delta.mean():.3f}" + "$. An entropy-threshold admission "
       "rule (same generator, $\\alpha=0.10$) stays within $1.6$ percentage points of GBRS-A on every dataset.}\n"
       "\\end{table*}\n")
Path(sys.argv[1]).write_text(tex)
print("ok released", round(rel.rate.mean(), 3), "adaptive", round(g.violation_rate.mean(), 3), round(a2.violation_rate.mean(), 3), round(a5.violation_rate.mean(), 3))
print("max |GBRS-DEG| adaptive:", round((m[m.method=='gbrs'].set_index('dataset').violation_rate - m[m.method=='deg'].set_index('dataset').violation_rate).abs().max(), 2))
