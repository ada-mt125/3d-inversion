# Karnataka joint gravity–magnetic inversion

The code, runs and figures of report 3, `../karnataka_reports/3_joint_inversion.html` (and `.pdf`).

## Layout

| path | what |
|---|---|
| `scripts/joint_params.py` | the parameters of every run: each model's regularization, the couplings, the rock-sample variants (`VARIANTS`) |
| `scripts/run_lowres.py` | the 2 km runs on this machine (`--bounds KEY` for a rock-sample variant) |
| `scripts/make_figures.py` | figures and `numbers.json`: `--set v2` for the report (into `figures_v2/`), without it the first series (`figures/`) |
| `scripts/constraint_figures.py`, `scripts/compare_bounds.py` | Section 5 (the rock-sample constraints): figures and the comparison tables |
| `scripts/build_report_v2.py`, `scripts/report_text_v2.py` | the report; every number is read from `figures_v2/numbers.json` |
| `scripts/build_workflow.py` | the interactive viewer `karnataka_joint.geoinv3d_viewer.html` (not in the repository) |
| `scripts/build_report.py`, `report_text.py`, `build_fixes_report.py` | the superseded first-series report and the list of fixes, into `archive/` |
| `scripts/fixes_figures.py`, `fixes_summary.py` | the magnetization-vector figure and numbers, used by `make_figures.py --set v2` |
| `data/ec2_runs_fixed/`, `data/lowres_runs_fixed/` | the runs of the report (second series), full resolution and 2 km |
| `data/ec2_runs_bounds/`, `data/lowres_bounds/` | the runs with the rock-sample bounds and depth weighting (Section 5) |
| `data/ec2_runs/`, `data/lowres_runs/` | the first series (archive) |
| `data/rock_properties/` | the measured densities and susceptibilities per rock family |
| `figures_v2/` | the report's figures and `numbers.json`; `figures/` holds those of the first series |
| `archive/` | the first-series report (HTML, PDF) and the list of fixes |

## Rebuilding

    py examples/output/karnataka_joint/scripts/make_figures.py --set v2
    py examples/output/karnataka_joint/scripts/build_report_v2.py --pdf
    py examples/output/karnataka_joint/scripts/build_workflow.py

Runs on EC2: `py deploy/ec2_multi_run.py karnataka-joint --only <coupling>[_<variant>] --collect DIR`
(the variants of `joint_params.py`).
