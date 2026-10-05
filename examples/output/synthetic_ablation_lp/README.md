# Synthetic ablation 2: Lp, tuned, against a reference model and boreholes

The code, inputs, runs and figures of the report `ablation_report.html` (and `.pdf`), which
replaces the first study (`../synthetic_ablation`): there every method had one setting, the
intrusion was a 100 m sheet, the prior was pinned by the model builder's default range, and the
holes reached no further than their trace. The report is written for readers new to the test
and does not refer to the first study; it is published privately as a claude.ai artifact
(https://claude.ai/artifact/SrqZzEDy5cYpgkdJEJQAfW).

The model: a cube (300 m, 100–400 m below the ground, 0.5 SI) and a stubby intrusion (250 m thick,
1.2 km along strike, 100–600 m, 45° to the east, 1 SI) under random ground at Block 8's field
(I 19.24°, D −1.32°), the total-field anomaly 80 m above the ground. Lp only, 40 settings — norms
(0,0,0,0), (0,1,1,1), (0,2,2,1), (0,2,2,2), (1,1,1,1) × depth weighting β 1, 1.5, 2, 3 × smoothness
length scale 1, 3 — up to 200 iterations (100 IRLS), each in five groups:

| group | prior |
|---|---|
| A | none: the reference 0, bounds 0–3 SI |
| B (w1, w10, w100) | the bodies drawn ~100 m off, 50 m deeper, 10° steeper, κ ±50 %, as a reference model only (bounds 0–3 SI) at weight 1, 10 or 100 |
| C | five boreholes logging the true susceptibility, held at the trace and reaching 150 m around it (`radius_m`, new in the model builder) |

The settings are ranked by the volume-matched overlap with the true bodies (`scripts/metrics.py`).

## Layout

| path | what |
|---|---|
| `scripts/make_synthetic.py` | the model, the data, the priors and `inputs/runs.json` (the 200 jobs) |
| `scripts/metrics.py` | the true model and the measures |
| `scripts/make_figures.py` | the figures, `figures/numbers.json` (every run's measures) and `data/best/` |
| `scripts/build_report.py` | the report; every number is read from `numbers.json` or `inputs/` |
| `data/sweep.log` | the AWS run: 10 × c5.4xlarge in ap-south-1, 4 jobs at a time each |
| `data/runs/` | the 200 results (not in the repository); `data/best/` the ones the report shows |

## Rebuilding

    py examples/output/synthetic_ablation_lp/scripts/make_synthetic.py
    py deploy/ec2_sweep.py examples/output/synthetic_ablation_lp/inputs/runs.json \
        examples/output/synthetic_ablation_lp/inputs examples/output/synthetic_ablation_lp/data/runs \
        --instances 10 --type c5.4xlarge --parallel 4
    py examples/output/synthetic_ablation_lp/scripts/make_figures.py
    py examples/output/synthetic_ablation_lp/scripts/build_report.py --pdf

`deploy/sweep_runner.py` runs the same jobs on one machine instead (`jobs.json` of name → params).
