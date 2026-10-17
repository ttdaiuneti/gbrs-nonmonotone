"""Write the LaTeX violation-rate table (Table 5) from the raw audit CSVs.

Usage: python3 make_mono_table.py <output .tex path>
Sources: results/mono_audit.csv (size-adaptive GBRS, NRS, HTBS; seed 42) and
results/nmin_fixed_audit_16sets.csv (GBRS with n_min fixed at 2 and 5; seed 42).
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

NAMES = {"ecoli": "Ecoli", "diabetes": "Diabetes", "yeast": "Yeast", "wine": "Wine",
         "heart_statlog": "Heart-Statlog", "vehicle": "Vehicle", "segment": "Segment",
         "breast_cancer": "BreastCancer", "steel_plates": "SteelPlates",
         "ionosphere": "Ionosphere", "dermatology": "Dermatology", "pc3": "PC3",
         "waveform": "Waveform", "spectf": "SPECTF", "sonar": "Sonar", "digits": "Digits"}
order = sorted(NAMES, key=lambda k: (int(g.loc[k, "d"]), list(NAMES).index(k)))
rows = [f"{NAMES[k]} & {int(g.loc[k, 'd'])} & {g.loc[k, 'violation_rate']:.1f} & "
        f"{a2.loc[k, 'violation_rate']:.1f} & {a5.loc[k, 'violation_rate']:.1f} & "
        f"{g.loc[k, 'mean_delta']:.4f} & {r.loc[k, 'violation_rate']:.1f}" for k in order]
hdr = (" & & \\multicolumn{3}{c}{GBRS} & & NRS & & & \\multicolumn{3}{c}{GBRS} & & NRS \\\\\n"
       "\\cmidrule(lr){3-5}\\cmidrule(lr){10-12}\n"
       "Dataset & $d$ & Adapt. & $\\nmin\\!=\\!2$ & $\\nmin\\!=\\!5$ & $\\overline{|\\Delta\\gamma|}$ & Rate"
       " & Dataset & $d$ & Adapt. & $\\nmin\\!=\\!2$ & $\\nmin\\!=\\!5$ & $\\overline{|\\Delta\\gamma|}$ & Rate \\\\\n")
body = "\n".join(f"{l} & {rr} \\\\" for l, rr in zip(rows[:8], rows[8:]))
avg = (f"\\multicolumn{{14}}{{c}}{{\\textit{{Average over 16 datasets:}} GBRS {g.violation_rate.mean():.1f}\\% (adaptive), "
       f"{a2.violation_rate.mean():.1f}\\% ($\\nmin=2$), {a5.violation_rate.mean():.1f}\\% ($\\nmin=5$); "
       f"$\\overline{{|\\Delta\\gamma|}}={g.mean_delta.mean():.4f}$; NRS {r.violation_rate.mean():.1f}\\%}} \\\\")
tex = ("\\begin{table*}[pos=!htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3.2pt}\n"
       "\\caption{Violation rate (\\%) of the random-subset audit, ordered by $d$. GBRS is audited with the "
       "size-adaptive minimum ball size of the published protocol (Adapt.) and with $\\nmin$ fixed at 2 and 5; "
       "$\\overline{|\\Delta\\gamma|}$ is the mean size of a violation for the adaptive protocol. HTBS had no violations on any dataset.}\n"
       "\\label{tab:mono}\n"
       "\\begin{tabular*}{\\linewidth}{@{}l @{\\hspace{6pt}\\extracolsep{\\fill}} r r r r r r @{\\hspace{0.8em}} l @{\\hspace{6pt}} r r r r r r@{}}\n"
       "\\toprule\n" + hdr + "\\midrule\n" + body + "\n\\midrule\n" + avg + "\n\\bottomrule\n\\end{tabular*}\n"
       "\\tabnote{%\nSeed 42, $\\varepsilon=10^{-10}$; 200 random subsets for $d\\le20$ and 100 for $d>20$. "
       "The $95\\%$ bootstrap interval of the adaptive average is $41.1$--$50.5$. An entropy-threshold admission rule "
       "(same $k$-means engine, $\\alpha=0.10$) stays within $1.6$ percentage points of GBRS on every dataset.}\n"
       "\\end{table*}\n")
Path(sys.argv[1]).write_text(tex)
print("ok", round(g.violation_rate.mean(), 3), round(a2.violation_rate.mean(), 3), round(a5.violation_rate.mean(), 3))
print("max |GBRS-DEG| adaptive:", round((m[m.method=='gbrs'].set_index('dataset').violation_rate - m[m.method=='deg'].set_index('dataset').violation_rate).abs().max(), 2))
