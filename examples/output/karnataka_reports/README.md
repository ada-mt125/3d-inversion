# Karnataka (Sandur) study: the reports

Four reports on the gravity and aeromagnetic data of a 70 × 70 km area around the Sandur schist belt,
Ballari district, Karnataka (UTM 43N, easting 641–711 km, northing 1634–1704 km). Each is a
self-contained HTML page and a PDF. The first three describe the methods; the fourth is the
industry-facing synthesis.

| # | Report | Reader | What it does | Code and runs |
|---|---|---|---|---|
| 1 | `1_gravity_inversion` | technical | Six regularization settings with the ground in the mesh and a terrain correction, compared with the flat-earth runs; a 2 km study | `../karnataka_gravity_terrain/` |
| 2 | `2_magnetic_inversion` | technical | Seven regularization settings of the aeromagnetic inversion and six tests of the data preparation | `../karnataka_magnetic/` |
| 3 | `3_joint_inversion` | technical | Six couplings of the joint gravity–magnetic inversion, how to choose one, remanence (MVI), the rock-sample constraints | `../karnataka_joint/` |
| 4 | `4_mineral_prospectivity` | industry | Known endowment, ranked priority areas, validation against the mines, phased programme, risks | `../karnataka_minerals/` |

## Main conclusions

- The inversions recover the belt robustly: a dense metavolcanic core with steep magnetic
  iron-formation sheets on both flanks. All four located iron mines lie on or within 1.4 km of
  strongly magnetic rock, which only 7–20% of the area does.
- Measured rock densities used as bounds change the density model more than any coupling: the belt
  reaches the surface and its base lies at about 5.5 km. A structural coupling (joint total
  variation) suits this belt.
- The magnetic data need remanent magnetization at and south of Kumaraswamy (magnetization-vector
  model: RMS 49 → 28 nT).
- Priority iron areas beyond the located mines: R1, the remanent zone (46 km²), and F1, a dense,
  strongly magnetic body (32 km²) in the centre of the belt.
- 1 km cells do not resolve the ore (within 170 m of the surface) or single iron-formation layers
  (100–200 m); manganese, gold and copper need IP/EM and geochemistry.

## Rebuilding the reports

From the repository root (Python of `.venv`; headless Chrome for the PDFs):

    py examples/output/karnataka_gravity_terrain/scripts/make_figures.py
    py examples/output/karnataka_gravity_terrain/scripts/build_report.py --pdf     # 1
    py examples/output/karnataka_magnetic/scripts/make_figures.py
    py examples/output/karnataka_magnetic/scripts/build_report.py --pdf            # 2
    py examples/output/karnataka_joint/scripts/make_figures.py --set v2
    py examples/output/karnataka_joint/scripts/build_report_v2.py --pdf            # 3
    py examples/output/karnataka_minerals/scripts/make_figures.py
    py examples/output/karnataka_minerals/scripts/build_report.py --pdf            # 4

Each builder writes its HTML and PDF into this folder. The interactive viewers of the runs (not in
the repository) are made by `py examples/output/karnataka_inputs/scripts/build_all_workflow.py` and
sit in each study folder (`*.geoinv3d_viewer.html`).

## Superseded material

- `../karnataka_gravity/`: the flat-earth gravity comparison of 28 September (English and Chinese),
  the first stage of the study; report 1 supersedes it.
- `../karnataka_joint/archive/`: the first series of the joint inversion and the list of changes
  against it; report 3 supersedes them.
