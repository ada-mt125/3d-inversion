# Karnataka mineral prospectivity

The code and figures of report 4, `../karnataka_reports/4_mineral_prospectivity.html` (and `.pdf`): a
synthesis of reports 1–3 and the published record, without new inversions.

## Layout

| path | what |
|---|---|
| `scripts/make_figures.py` | the screens (iron-formation horizons, the remanent zone R1, the segments F1–F7), the figures and `figures/numbers.json` |
| `scripts/validation.py` | the published localities against the models (distances, the schematic map), called by `make_figures.py` |
| `scripts/build_report.py` | the report; model values from `figures/numbers.json` and `../karnataka_joint/figures_v2/numbers.json` |
| `figures/` | the report's figures and `numbers.json` |

The references, the mine and occurrence positions and the schematic geological map drawn from the
DEM are in `../karnataka_inputs/shared/literature.py`.

## Rebuilding

    py examples/output/karnataka_minerals/scripts/make_figures.py
    py examples/output/karnataka_minerals/scripts/build_report.py --pdf
