# Karnataka with terrain: gravity, magnetics and their joint inversion

Studies of the same 70 x 70 km area (UTM 43N, easting 641-711 km, northing 1634-1704 km) on one mesh
that follows the ground (1 km x 1 km x 250 m cells, 335,518 below the ground), run on 30 September -
1 October 2026. **The four reports are in `../karnataka_reports/`** (HTML and PDF, with an index);
each study folder holds the code, the runs and the figures of its report.

| folder | what | report |
|---|---|---|
| `karnataka_inputs/` (this folder) | the DEM, the terrain correction and the data at the 1 km nodes; `scripts/` prepares them and builds the viewers, `shared/` holds the code the reports share | - |
| `karnataka_gravity_terrain/` | the six regularizations of the flat-earth comparison, with terrain and terrain correction | `1_gravity_inversion` |
| `karnataka_magnetic/` | the magnetic data with seven regularizations and six tests of the set-up | `2_magnetic_inversion` |
| `karnataka_joint/` | gravity + magnetics with each coupling of the joint inversion, remanence, the rock-sample constraints | `3_joint_inversion` |
| `karnataka_minerals/` | the mineral prospectivity of the area from the three studies and the published record | `4_mineral_prospectivity` |

`shared/`: `kmodel.py` (models on the mesh with terrain and their measures), `style.py` (the report
page), `literature.py` (the references, the mine and occurrence positions and the schematic map of the
mineral report).

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

    # figures, reports and PDFs (headless Chrome or Edge): see ../karnataka_reports/README.md
    py examples/output/karnataka_joint/scripts/make_figures.py --set v2
    py examples/output/karnataka_joint/scripts/build_report_v2.py --pdf

    # the interactive viewers, one per study, each in its folder (the 1 km runs; --with-2km
    # adds the 2 km studies): all three, or one study's with its scripts/build_workflow.py
    py examples/output/karnataka_inputs/scripts/build_all_workflow.py
    py examples/output/karnataka_joint/scripts/build_workflow.py

The variants of each case are in `deploy/ec2_multi_run.py` (gravity, magnetics) and in
`karnataka_joint/scripts/joint_params.py` (joint).  The workflow viewers
(`karnataka_gravity_terrain/karnataka_gravity_terrain.geoinv3d_viewer.html`, 20 MB;
`karnataka_magnetic/karnataka_magnetic.geoinv3d_viewer.html`, 29 MB;
`karnataka_joint/karnataka_joint.geoinv3d_viewer.html`, 101 MB) are not in the repository; the
scripts above make them from the results.

## Preparing the inputs again (`scripts/prepare_inputs.py`)

Needs the raw survey data, which are not in the repository, in one folder (default: the Desktop):

- `karnataka_ap_gravity/GEOTIFF/NGPM_BA.tiff`, `karnataka_ap_gravity/ASCII/combined_NGPM_gravity.csv`
- `karnataka_ap_magnetic/GRIDS/GEOTIFF/TAIL_TMI_GE.tiff`
- `karnataka_ap_gravity/DEM/`: six Copernicus GLO-90 tiles, N14-N15 x E075-E077, from
  `https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_N15_00_E076_00_DEM/Copernicus_DSM_COG_30_N15_00_E076_00_DEM.tif`
  (and the five other names)

    py examples/output/karnataka_inputs/scripts/prepare_inputs.py [FOLDER]

It writes `dem_utm43n_450m.tif` (the ground for the mesh), `gravity_simple_1km.csv` and
`gravity_complete_1km.csv` (the NGPM grid without and with the terrain correction of
`geoinv3d/methods/terrain.py`), `magnetic_1km*.csv` (the TMI grid continued upwards with
`geoinv3d/methods/continuation.py`), `stations_tc.csv`, and `prep.json` / `prep.npz` for the reports.

## The second joint series (30 September, evening)

`../karnataka_reports/3_joint_inversion.html` (and `.pdf`): the couplings again, with the group
lasso weighed like the other runs (cell volume x depth weight, beta = 1), each datum by its error and
the two datasets balanced to chi^2 = N each, plus the magnetic data with a magnetization vector per
cell (MVI).  Every run fits both datasets (chi^2 / N 0.88-1.10); the group lasso keeps 89 % / 75 % of
its models in the core and pairs the supports (81 % of its magnetic cells anomalous in density); MVI
lowers the RMS of the stations south of the Sandur belt from 321 to 129 nT, the strong cells pointing
at I 74, D -79 (remanence).  Runs in `karnataka_joint/data/ec2_runs_fixed`, `data/lowres_runs_fixed`
and `karnataka_magnetic/data/ec2_runs/beta1_mvi`; built by `make_figures.py --set v2` and
`build_report_v2.py`.  `karnataka_joint/archive/karnataka_joint_fixes_en.html` lists what changed against the first series.

Still open: PGI was not run again at full resolution (its first beta needs tuning per dataset); the
joint inversions couple the induced susceptibility, not the magnetization vector; the group lasso is
the slowest coupling (124 min on c5.18xlarge).
