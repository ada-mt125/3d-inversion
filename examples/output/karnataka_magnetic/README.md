# Karnataka magnetic inversion

The code, runs and figures of report 2, `../karnataka_reports/2_magnetic_inversion.html` (and `.pdf`).

## Layout

| path | what |
|---|---|
| `scripts/make_figures.py` | figures and `figures/numbers.json` |
| `scripts/build_report.py` | the report; every number is read from `numbers.json` |
| `scripts/build_workflow.py` | the interactive viewer `karnataka_magnetic.geoinv3d_viewer.html` (not in the repository) |
| `data/ec2_runs/` | the 13 full-resolution runs of the report, and `beta1_mvi`, the magnetization-vector run of report 3 |
| `figures/` | the report's figures and `numbers.json` |

The inputs (the TMI grid continued upwards, at the 1 km nodes) are in `../karnataka_inputs/`; the run
parameters in `deploy/ec2_multi_run.py` (case `karnataka-magnetic`).

## Rebuilding

    py examples/output/karnataka_magnetic/scripts/make_figures.py
    py examples/output/karnataka_magnetic/scripts/build_report.py --pdf
    py examples/output/karnataka_magnetic/scripts/build_workflow.py
