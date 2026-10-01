# Karnataka gravity inversion with terrain

The code, runs and figures of report 1, `../karnataka_reports/1_gravity_inversion.html` (and `.pdf`).

## Layout

| path | what |
|---|---|
| `scripts/make_figures.py` | figures and `figures/numbers.json` |
| `scripts/build_report.py` | the report; every number is read from `numbers.json` |
| `scripts/run_lowres.py` | the 2 km study on this machine (`data/lowres_runs/`) |
| `scripts/build_workflow.py` | the interactive viewer `karnataka_gravity_terrain.geoinv3d_viewer.html` (not in the repository) |
| `data/ec2_runs/` | the eight full-resolution runs: the six settings, and β = 1 without the terrain correction and on a flat earth |
| `data/lowres_runs/` | the 16 runs of the 2 km study |
| `figures/` | the report's figures and `numbers.json` |

The inputs (DEM, terrain correction, gravity at the 1 km nodes) are in `../karnataka_inputs/`; the run
parameters in `deploy/ec2_multi_run.py` (case `karnataka-gravity-terrain`).

## Rebuilding

    py examples/output/karnataka_gravity_terrain/scripts/make_figures.py
    py examples/output/karnataka_gravity_terrain/scripts/build_report.py --pdf
    py examples/output/karnataka_gravity_terrain/scripts/build_workflow.py
