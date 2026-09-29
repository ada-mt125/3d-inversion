# Karnataka gravity: comparing regularizations

Everything needed to rebuild the comparison report, independent of any session folder. Run from the
repository root:

    py examples/output/karnataka_gravity/scripts/make_figures.py      # figures/zh, figures/en, figures/numbers.json
    py examples/output/karnataka_gravity/scripts/build_reports.py --pdf   # the two reports and the English PDF
    py examples/output/karnataka_gravity/scripts/build_workflow.py    # all 22 runs in one DAG viewer

## Reports

- `karnataka_gravity_report_zh.html`, `karnataka_gravity_report_en.html`, `karnataka_gravity_report_en.pdf`
  (published at https://claude.ai/artifact/6npeQff1gfaF3HTrq1eST3 and
  https://claude.ai/artifact/32sBUmrzqruCrPsuxGgTdk). The comparison comes first; the original
  α_s = 1e-4 settings are covered briefly in Section 5.
- `depth_study/karnataka_depth_study.geoinv3d_viewer.html` — the six full-resolution runs and the
  16-run 2 km study in one workflow.
- `archive/` — the earlier version of the report (27–28 September), organised around the original
  sparse run. Superseded; kept for reference.

## Data

- `data/ec2_runs/` — six full-resolution SimPEG runs on EC2 (1 km × 1 km × 500 m, 190,512 cells, 5,040 data):
  original sparse (α_s = 1e-4, sensitivity weighting), L1–L2 (IRLS), α_s = 1 with depth weighting
  β = 0.5 / 1 / 1.5, α_s = 0.1 with β = 1.
- `data/lowres_runs/` — the 16-run 2 km study (1,295 data).
- `data/simpeg_8km_mesh_profiles.json` — profiles under the two highs from the SimPEG run on the 8 km deep
  mesh used in the Tomofast-x comparison (24 Sep; full model in Desktop/share_simpeg).
- `data/tomofastx/` — the SimPEG and Tomofast-x section figures from that comparison.
- `data/NGPM_BA.tiff` — the Bouguer grid (for the power spectrum).

## Scripts

- `scripts/common.py` — paths, run registry, depth metrics (centroid depth, not peak depth: models held at
  the density bound have a flat peak).
- `scripts/make_figures.py` — all figures and `figures/numbers.json`.
- `scripts/build_reports.py`, `scripts/report_style.py` — the reports; every number in the text is read from
  `numbers.json`.
- `scripts/build_workflow.py` — the DAG viewer.
