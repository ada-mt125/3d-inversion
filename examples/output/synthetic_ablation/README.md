# Synthetic ablation: priors and regularization in magnetic inversion

The code, inputs, runs and figures of the report `ablation_report.html` (and `.pdf`, in Chinese).

A known model under random ground at Block 8's field (I 19.24°, D -1.32°): a cube (300 m, 100–400 m
below the ground, 0.5 SI) and a dipping intrusion (100 m thick, 50–1,000 m, 45° to the east, 1 SI),
the total-field anomaly 80 m above the ground. Six regularizations (L2, Lp 0,2,2,1, L1, L1–L2,
MGS, TV), each in three groups:

| group | prior |
|---|---|
| A | none: a uniform half-space, bounds 0–3 SI |
| B | the two bodies as the model builder draws them, ~100 m off, 50 m deeper, 10° steeper, κ ±50 % (defaults: ±0.005 SI, weight 5) |
| C | five boreholes logging the true susceptibility (no outlines) |

## Layout

| path | what |
|---|---|
| `scripts/make_synthetic.py` | the model, the data, the priors of B and C and `inputs/runs.json` (the 18 jobs' parameters); byte-identical to the inputs of the runs |
| `scripts/collect.py` | copies the 18 local runs (GUI workspace 69a9bb3bfb59) to `data/runs/`, checking each one's parameters against `runs.json` |
| `scripts/metrics.py` | the true model and the measures (correlation, κ in the bodies, the cube's depth, the intrusion's dip, …) |
| `scripts/make_figures.py` | the figures and `figures/numbers.json` |
| `scripts/build_report.py` | the report; every number is read from `numbers.json` or `inputs/` |
| `inputs/` | DEM, TMI, truth, `spec_B.json`, `boreholes_C.csv` + `spec_C.json`, `runs.json` |
| `data/runs/<group>_<method>/` | each run's `result.zip` and `run.json` |

## Rebuilding

    py examples/output/synthetic_ablation/scripts/make_synthetic.py
    py examples/output/synthetic_ablation/scripts/make_figures.py
    py examples/output/synthetic_ablation/scripts/build_report.py --pdf

To rerun the inversions, submit each job of `inputs/runs.json` (its `params` and `files`) to the
local server, then point `collect.py`'s `JOBS` at the new job ids. Each run takes 1–2 minutes on a
laptop (233,268 cells, 2,601 data).
