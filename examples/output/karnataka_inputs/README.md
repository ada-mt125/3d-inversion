# Karnataka with terrain: gravity, magnetics and their joint inversion

Three studies of the same 70 x 70 km area (UTM 43N, easting 641-711 km, northing 1634-1704 km) on
one mesh that follows the ground (1 km x 1 km x 250 m cells, 335,518 below the ground), run on
30 September 2026.  Each has a report (HTML and PDF), its figures, its results and its scripts.

| folder | what | report |
|---|---|---|
| `karnataka_inputs/` (this folder) | the DEM, the terrain correction and the data at the 1 km nodes | - |
| `karnataka_gravity_terrain/` | the six regularizations of the flat-earth comparison, with terrain and terrain correction | `karnataka_gravity_terrain_report_en.pdf` |
| `karnataka_magnetic/` | the magnetic data with seven regularizations and six tests of the set-up | `karnataka_magnetic_report_en.pdf` |
| `karnataka_joint/` | gravity + magnetics with each coupling of the joint inversion | `karnataka_joint_report_en.pdf` |

## Running it on another machine

Everything the inversions need is in this folder (the prepared inputs are in the repository):

    pip install -e .                       # from the repository root; Python 3.13
    pip install scikit-learn rasterio      # PGI; GeoTIFF and projections

    # a full-resolution run on this machine (about 16 GB of memory; 10-30 minutes on 16 cores)
    py deploy/ec2_multi_run.py karnataka-gravity-terrain --only as1_beta1 --local --collect examples/output/karnataka_gravity_terrain/data/ec2_runs
    py deploy/ec2_multi_run.py karnataka-magnetic --only beta1 --local --collect examples/output/karnataka_magnetic/data/ec2_runs
    py deploy/ec2_multi_run.py karnataka-joint --only cross_gradient --local --collect examples/output/karnataka_joint/data/ec2_runs

    # the same on EC2 (needs AWS credentials for EC2; every instance is tagged Owner=<IAM user>
    # and terminated at the end): leave out --local, add --parallel 4
    py deploy/ec2_multi_run.py karnataka-joint --only none,cross_gradient --parallel 4 --collect examples/output/karnataka_joint/data/ec2_runs

    # the coarse 2 km studies (minutes each, on a laptop)
    py examples/output/karnataka_gravity_terrain/scripts/run_lowres.py
    py examples/output/karnataka_joint/scripts/run_lowres.py none cross_gradient

    # figures, report and PDF (headless Chrome or Edge), and the interactive viewer of all runs
    py examples/output/karnataka_joint/scripts/make_figures.py
    py examples/output/karnataka_joint/scripts/build_report.py --pdf
    py examples/output/karnataka_joint/scripts/build_workflow.py

The variants of each case are in `deploy/ec2_multi_run.py` (gravity, magnetics) and in
`karnataka_joint/scripts/joint_params.py` (joint).  The workflow viewers (`*_viewer.html`, 25-30 MB
each) are not in the repository; `build_workflow.py` makes them from the results.

## Preparing the inputs again (`prepare_inputs.py`)

Needs the raw survey data, which are not in the repository, in one folder (default: the Desktop):

- `karnataka_ap_gravity/GEOTIFF/NGPM_BA.tiff`, `karnataka_ap_gravity/ASCII/combined_NGPM_gravity.csv`
- `karnataka_ap_magnetic/GRIDS/GEOTIFF/TAIL_TMI_GE.tiff`
- `karnataka_ap_gravity/DEM/`: six Copernicus GLO-90 tiles, N14-N15 x E075-E077, from
  `https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_N15_00_E076_00_DEM/Copernicus_DSM_COG_30_N15_00_E076_00_DEM.tif`
  (and the five other names)

    py examples/output/karnataka_inputs/prepare_inputs.py [FOLDER]

It writes `dem_utm43n_450m.tif` (the ground for the mesh), `gravity_simple_1km.csv` and
`gravity_complete_1km.csv` (the NGPM grid without and with the terrain correction of
`geoinv3d/methods/terrain.py`), `magnetic_1km*.csv` (the TMI grid continued upwards with
`geoinv3d/methods/continuation.py`), `stations_tc.csv`, and `prep.json` / `prep.npz` for the reports.

## The second joint series (30 September, evening)

`karnataka_joint/karnataka_joint_report_v2_en.html` (and `.pdf`): the couplings again, with the group
lasso weighed like the other runs (cell volume x depth weight, beta = 1), each datum by its error and
the two datasets balanced to chi^2 = N each, plus the magnetic data with a magnetization vector per
cell (MVI).  Every run fits both datasets (chi^2 / N 0.88-1.10); the group lasso keeps 89 % / 75 % of
its models in the core and pairs the supports (81 % of its magnetic cells anomalous in density); MVI
lowers the RMS of the stations south of the Sandur belt from 321 to 129 nT, the strong cells pointing
at I 74, D -79 (remanence).  Runs in `karnataka_joint/data/ec2_runs_fixed`, `data/lowres_runs_fixed`
and `karnataka_magnetic/data/ec2_runs/beta1_mvi`; built by `make_figures.py --set v2` and
`build_report_v2.py`.  `karnataka_joint_fixes_en.html` lists what changed against the first series.

Still open: PGI was not run again at full resolution (its first beta needs tuning per dataset); the
joint inversions couple the induced susceptibility, not the magnetization vector; the group lasso is
the slowest coupling (124 min on c5.18xlarge).
`shared/` holds the code the three reports share (`kmodel.py`: models on the mesh
with terrain and their measures; `style.py`: the report page).
