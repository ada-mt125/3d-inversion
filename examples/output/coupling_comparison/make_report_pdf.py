"""Print report/index.html to report/coupling_comparison_report.pdf (A4) with headless Chrome or Edge.

The print copy is written next to the report so the figures resolve; it is removed afterwards.
Block layout in print: Chrome overlaps grid rows that break across pages.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPORT = Path(__file__).resolve().parent / "report"
PDF = REPORT / "coupling_comparison_report.pdf"

PRINT_CSS = """
@page { size: A4; margin: 12mm 12mm 14mm; }
html { zoom: 0.8; }
body { background: #fff; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.wrap, section, header, figure, .findings.why, .summary { display: block; }
.wrap { padding: 0; max-width: none; }
.wrap > * + * { margin-top: 40px; }
section > * + *, .summary > * + * { margin-top: 18px; }
header > * + * { margin-top: 12px; }
figure > figcaption { margin-top: 8px; }
.findings.why > * + * { margin-top: 10px; }
section > .steps, .findings.why > .steps { margin-top: 12px; }
figure, .tablewrap, .findings, .formulas, .setup, .couplings, tr, li { break-inside: avoid; }
h2, h3, .lede { break-after: avoid; }
.plate, .tablewrap, .formulas { overflow: visible; }
.links a::after { content: " (" attr(href) ")"; font-weight: 400; font-size: 12px; }
"""

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser",
]


def browser():
    for b in BROWSERS:
        if os.path.isfile(b) or shutil.which(b):
            return b
    sys.exit("no Chrome, Edge or Chromium found")


def main():
    html = (REPORT / "index.html").read_text(encoding="utf-8")
    html = html.replace(' loading="lazy"', "").replace("</style>", PRINT_CSS + "</style>", 1)
    if not re.match(r"\s*<!DOCTYPE", html, re.I):
        html = '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">\n' + html
    tmp = REPORT / "_print.html"
    tmp.write_text(html, encoding="utf-8")
    try:
        subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=15000", "--run-all-compositor-stages-before-draw",
                        f"--print-to-pdf={PDF}", tmp.as_uri()],
                       check=True, capture_output=True, timeout=180)
    finally:
        tmp.unlink(missing_ok=True)
    print(PDF, f"{PDF.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
