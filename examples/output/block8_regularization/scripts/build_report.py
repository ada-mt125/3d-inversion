"""Build the report of the regularization trials on the Block-8 window (English HTML, and the PDF).

    py examples/output/block8_regularization/scripts/make_figures.py
    py examples/output/block8_regularization/scripts/build_report.py [--pdf]

Every number comes from data/inputs/*.json and data/runs/*/score.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parent / "karnataka_inputs" / "shared"))
sys.path.insert(0, str(HERE))
from style import Figures, page, to_pdf  # noqa: E402
import report_text  # noqa: E402

OUT = ROOT / "block8_regularization_report_en.html"


def main():
    figs = Figures("en", ROOT / "figures")
    body = report_text.body(figs)
    OUT.write_text(page("en", "Block-8 regularization trials", body), encoding="utf-8")
    print("wrote", OUT)
    if "--pdf" in sys.argv:
        print(to_pdf(OUT))


if __name__ == "__main__":
    main()
