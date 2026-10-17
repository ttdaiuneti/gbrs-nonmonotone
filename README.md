# Non-Monotonicity in Granular-Ball Attribute Reduction — Reproducibility Package

Supplementary code and data for the manuscript
*Granular-Ball Attribute Reduction Is Not Monotone: Prevalence, an Impossibility
Result, and Downstream Consequences*
(T. D. Tran, T. C. Pham, T. H. L. Luong).

Every number, table, and figure in the manuscript is regenerable from the
scripts here plus the released CSVs. Raw per-dataset results are the single
source of truth: the `.tex` tables and the prose read from them; no number is
typed by hand or rounded backwards.

## Environment

- Python 3.11 (numbers were produced on an Apple-silicon laptop, single core)
- Dependencies and exact pins: see [`code/requirements.txt`](code/requirements.txt)

```bash
python -m venv venv && source venv/bin/activate
pip install -r code/requirements.txt
```

The sixteen benchmarks are fetched automatically via
`sklearn.datasets.fetch_openml` (OpenML v1 task ids) on first run and cached
locally; an internet connection is needed the first time only.

## Layout

```
code/                          Python sources (see "Core" and "Audits" below)
supplementary_data/            released raw CSVs backing every reported number
results/                       additional raw outputs (nested CV, stats reruns)
```

## Core modules

| File | Role |
|---|---|
| `code/gbrs_core.py` | GBRS, DEG-GB (entropy control), HTBS, and NRS reducers; forward-greedy loop; content-addressed γ cache |
| `code/run_experiments.py` | dataset loaders and the shared 5-fold kNN-3 / SVM-RBF evaluation harness |

## Reproducing each claim

Each script writes a CSV; the manuscript element it backs is listed alongside.
Run from inside `code/`.

| Script | Output | Backs |
|---|---|---|
| `run_mono_all.py` | `mono_audit.csv` | Table 1 (reference audit), Fig. 4 |
| `run_mono_multiseed.py` | `mono_audit_multiseed.csv` | Table (multi-seed), §4.4 |
| `run_nmin_fixed_audit.py` | `nmin_fixed_audit_16sets.csv` | Table (n_min confound), §4.4, headline 18.7% |
| `run_nrs_fixed_delta.py` | `nrs_fixed_delta.csv` | Table (fixed-δ), §4.4 |
| `run_variant_audit.py` | `variant_audit.csv` | Table (variants), §4.4 |
| `run_greedy_path_audit.py` | `greedy_path_audit.csv` | §4.5 greedy-path + γ_AT-overshoot |
| `rerun_classification.py` | `classification_rerun.csv` | Tables 5–7, Fig. 8, Acc_full comparison |
| `run_balanced_metrics.py` | `classification_balanced.csv` | balanced-accuracy / F1 / MCC table |
| `run_nested_cv.py` | `nested_cv_16sets.csv` (SteelPlates rows: `--datasets steel_plates`, indicator columns dropped) | Table (nested CV), §5.6 |
| `wine_freeze_audit.py` | `wine_freeze_audit.csv`, `wine_freeze_summary.csv` | §5.2 Wine freeze ceiling |
| `diagnose_leakage.py` | `leakage_diagnosis.md` | §5.1 SteelPlates leakage correction |
| `run_statistics.py` | `stats_*.csv` | Wilcoxon / Friedman–Nemenyi (Tables 7, 10) |
| `generate_cd_diagram.py` | `figures/fig_cd_diagram.pdf` | Fig. 6 (CD diagram) |
| `generate_tables.py` | `tables/*.tex` | regenerates LaTeX table blocks from CSVs |
| `prop1_counterexample.py` | (assertions only) | verifies the Proposition 1 counterexample by running `gbrs_core` directly |
| `overshoot_trace.py` | (prints a trace) | verifies the §4.5 γ-overshoot mechanism (why GBRS reducts stop at \|R\|=1) is a genuine, non-buggy consequence of non-monotone γ |

Verify the headline theoretical claim in one step:

```bash
cd code && python prop1_counterexample.py
```

This runs the paper's own ball-generation routine on the four explicit points
in the proof of Proposition 1 and asserts γ drops from 1.0 to 0.0, then checks
the result is stable under ±0.3 perturbation of every coordinate.

Verify that the small (often |R|=1) reducts are not a stopping-condition bug:

```bash
cd code && python overshoot_trace.py
```

Traces, on Heart-Statlog, exactly why `forward_greedy` halts after one
attribute (its γ already exceeds the full-attribute target computed once at
the start), then repeats the trace with NRS on the same data to show the
early stop is specific to non-monotone γ, not the greedy-loop structure.

## Notes on determinism

- γ is deterministic given the audit seed: fresh-generation k-means uses
  class-centroid initialisation, so re-running with the same seed reproduces
  the rate bit-for-bit. The audit seed (default 42) varies only *which*
  attribute subsets are sampled.
- The reference audit (Table 1) uses 200 random subsets for d ≤ 20 and 100 for
  d > 20. The robustness controls (§4.4) use smaller budgets for tractability
  and are independent re-audits, not recomputations of Table 1's exact entries.

## Excluded datasets in the raw CSVs

`mono_audit.csv` and `classification_rerun.csv` contain rows for four datasets
that are **not** part of the sixteen reported in the paper: `iris`, `glass`,
`banknote`, and `balance_scale`. These are exploratory runs left in the raw
files for completeness; all four are excluded from every table and figure per
the pre-registered criteria stated in the manuscript's Setup section (very
small n for iris/glass/banknote; arithmetic-rule labels for balance_scale).
They are retained here rather than deleted so the raw outputs are unedited, but
no reported number uses them.

## Data provenance

Fifteen benchmarks are from the UCI Machine Learning Repository; PC3 is a
NASA MDP / PROMISE software-defect dataset, loaded via its OpenML mirror.
`Waveform` = OpenML `waveform-5000` (n=5000); `Digits` = OpenML `optdigits`
(n=5620) — not the smaller sklearn/UCI slices.

## License

Released under the MIT License (see [`code/LICENSE`](code/LICENSE)).
