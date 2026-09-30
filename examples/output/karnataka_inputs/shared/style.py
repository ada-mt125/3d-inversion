"""Page shell, stylesheet and small HTML helpers for the Karnataka reports with terrain
(the style of examples/output/karnataka_gravity/scripts/report_style.py)."""

from __future__ import annotations

import base64
import html


LIGHT = """--bg: #f4f6f7; --surface: #ffffff; --ink: #1b232c; --muted: #56626e; --rule: #d6dde3;
  --accent: #1f5f8b; --accent-soft: #e3edf5; --warn-bg: #fbeae7; --warn: #9c3326;
  --good: #1e7a4f; --good-bg: #e2f2ea; --plate: #ffffff;"""
DARK = """color-scheme: dark;
    --bg: #10151b; --surface: #171e26; --ink: #e2e8ee; --muted: #9aa7b3; --rule: #2b3642;
    --accent: #86b8e0; --accent-soft: #1c2a38; --warn-bg: #3a1f1c; --warn: #f0a497;
    --good: #7fd1a6; --good-bg: #173127; --plate: #f7f8f9;"""
FONTS = {
    "zh": ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700'
           '&family=Noto+Serif+SC:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">',
           '"Noto Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif',
           '"Noto Serif SC", "Songti SC", "SimSun", serif', "15px/1.75"),
    "en": ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600'
           '&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap">',
           '"IBM Plex Sans", "Segoe UI", system-ui, sans-serif', '"Source Serif 4", Georgia, serif', "15.5px/1.65"),
}
CSS = """
:root { @@LIGHT@@
  --sans: @@SANS@@; --serif: @@SERIF@@;
  --mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { @@DARK@@ } }
:root[data-theme="dark"] { @@DARK@@ }
body { margin: 0; background: var(--bg); color: var(--ink); font: @@BODY@@ var(--sans); }
.page { max-width: 1120px; margin: 0 auto; padding-inline: 16px; padding-block: 40px 72px; }
.prose { max-width: 760px; }
header { border-bottom: 1px solid var(--rule); padding-bottom: 28px; margin-bottom: 32px; }
.eyebrow { font: 500 12px/1.4 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }
h1 { font: 700 clamp(28px, 4.2vw, 40px)/1.2 var(--serif); margin: 10px 0 10px; text-wrap: balance; }
.lede { color: var(--muted); font-size: 17px; margin: 0 0 22px; max-width: 780px; }
.meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px 24px; margin: 0; }
.meta div { border-top: 2px solid var(--accent); padding-top: 8px; }
.meta dt { font: 500 11px/1.4 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.meta dd { margin: 4px 0 0; font-size: 14px; line-height: 1.5; }
.summary { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 20px 24px; margin: 0 0 40px; max-width: 920px; }
.summary h2 { margin-top: 0; }
.summary ul { margin: 0; padding-left: 1.2em; display: grid; gap: 8px; }
h2 { font: 700 24px/1.3 var(--serif); margin: 52px 0 12px; text-wrap: balance; }
h2 .no { font: 500 14px var(--mono); color: var(--accent); margin-right: 10px; vertical-align: 3px; }
h3 { font-size: 16.5px; font-weight: 600; margin: 30px 0 8px; }
h3 .no { font: 500 13px var(--mono); color: var(--accent); margin-right: 8px; }
p { margin: 0 0 14px; }
.num, code { font-family: var(--mono); font-size: .9em; }
code { background: var(--accent-soft); padding: 1px 5px; border-radius: 3px; overflow-wrap: anywhere; }
td code { overflow-wrap: normal; }
b { font-weight: 600; }
figure { margin: 22px 0 28px; }
figure.narrow { max-width: 680px; }
.plate { background: var(--plate); border: 1px solid var(--rule); border-radius: 4px; padding: 10px; overflow-x: auto; }
.plate img { display: block; width: 100%; height: auto; min-width: 560px; }
figure.narrow .plate img { min-width: 420px; }
figcaption { color: var(--muted); font-size: 13.5px; line-height: 1.55; margin-top: 8px; max-width: 920px; }
.fn { font: 500 12px var(--mono); color: var(--accent); margin-right: 8px; }
.tablewrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 4px; background: var(--surface); margin: 16px 0 8px; }
table { border-collapse: collapse; font-size: 13px; width: 100%; min-width: 900px; }
table.compact { min-width: 560px; }
th, td { padding: 7px 10px; border-bottom: 1px solid var(--rule); text-align: left; white-space: nowrap; vertical-align: top; }
td.wrap { white-space: normal; min-width: 220px; }
th { font-weight: 500; color: var(--muted); background: var(--bg); }
td.n, th.n { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; }
td.warn { color: var(--warn); background: var(--warn-bg); }
td.good { color: var(--good); background: var(--good-bg); }
.note { font-size: 13px; color: var(--muted); }
.callout { border-left: 3px solid var(--accent); background: var(--accent-soft); padding: 12px 16px; border-radius: 0 4px 4px 0; margin: 16px 0 20px; max-width: 760px; }
ol.steps, ul.plain { padding-left: 1.3em; display: grid; gap: 7px; }
footer { border-top: 1px solid var(--rule); margin-top: 56px; padding-top: 16px; font-size: 13px; color: var(--muted); }
@media (max-width: 600px) { body { font-size: 15px; } .summary { padding: 16px; } }
@page { size: A4; margin: 14mm 13mm 15mm; }
@media print {
  :root, :root:not([data-theme="light"]), :root[data-theme="dark"] { color-scheme: light; @@LIGHT@@ }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body { background: #fff; font-size: 10pt; line-height: 1.5; }
  .page { max-width: none; padding: 0; }
  .prose, .lede, figcaption, .summary, .callout { max-width: none; }
  header { padding-bottom: 14pt; margin-bottom: 16pt; }
  h1 { font-size: 24pt; }
  h2 { font-size: 15pt; margin: 20pt 0 6pt; break-after: avoid; }
  h3 { break-after: avoid; }
  .summary { background: #fff; padding: 10pt 14pt; margin-bottom: 16pt; }
  figure, .callout, tr, .meta div { break-inside: avoid; }
  figure { margin: 10pt 0 14pt; }
  figure.narrow { max-width: 130mm; }
  .plate { overflow: visible; padding: 4pt; }
  .plate img, figure.narrow .plate img { min-width: 0; width: auto; max-width: 100%; max-height: 225mm; margin: 0 auto; }
  figcaption { font-size: 8.5pt; }
  .tablewrap { overflow: visible; }
  table, table.compact { min-width: 0; font-size: 7pt; }
  th, td { padding: 3pt 4pt; white-space: normal; }
  footer { margin-top: 20pt; }
}
"""


def page(lang, title, body):
    link, sans, serif, bodyfont = FONTS[lang]
    css = (CSS.replace("@@LIGHT@@", LIGHT).replace("@@DARK@@", DARK).replace("@@SANS@@", sans)
           .replace("@@SERIF@@", serif).replace("@@BODY@@", bodyfont))
    return (f'<!doctype html>\n<html lang="{"zh-CN" if lang == "zh" else "en"}">\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{title}</title>\n"
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            f'{link}\n<style>{css}</style>\n<div class="page">\n{body}\n</div>\n</html>\n')


class Figures:
    """Numbered figures embedded as base64 PNG, one counter per report."""

    def __init__(self, lang, figs):
        self.lang, self.figs, self.n, self.number = lang, figs, 0, {}

    def ref(self, name):
        """Number of a figure, allocated on first mention so text can refer ahead."""
        if name not in self.number:
            self.n += 1
            self.number[name] = self.n
        return self.number[name]

    def __call__(self, name, caption, narrow=False):
        data = base64.b64encode((self.figs / f"{name}.png").read_bytes()).decode()
        word = "图" if self.lang == "zh" else "Figure"
        k = self.ref(name)
        return (f'<figure{" class=narrow" if narrow else ""}><div class="plate">'
                f'<img src="data:image/png;base64,{data}" alt="{word} {k}"></div>'
                f'<figcaption><span class="fn">{word} {k}</span>{caption}</figcaption></figure>')


def table(head, rows, compact=False, numeric_from=1):
    """head: column titles; rows: lists of cells, each a string or (text, css class)."""
    th = "".join(f'<th{" class=n" if i >= numeric_from else ""}>{h}</th>' for i, h in enumerate(head))
    body = []
    for row in rows:
        tds = []
        for i, c in enumerate(row):
            text, cls = c if isinstance(c, tuple) else (c, "n" if i >= numeric_from else "")
            attr = ' class="%s"' % cls if cls else ""
            tds.append(f"<td{attr}>{text}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="tablewrap"><table{" class=compact" if compact else ""}><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def esc(s):
    return html.escape(s, quote=False)


def find_chrome():
    import shutil
    from pathlib import Path
    for p in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chrome")):
        if p and Path(p).exists():
            return p
    return None


def to_pdf(html_path):
    """Print a report to PDF next to it with headless Chrome; returns a line for the log.

    Chrome on macOS can stay open after writing the PDF, so it is stopped once the file
    exists and its size has not changed for two seconds (at most 240 s in all)."""
    import subprocess
    import sys
    import tempfile
    import time
    chrome = find_chrome()
    if not chrome:
        sys.exit("Chrome not found; the PDF needs headless Chrome")
    pdf = html_path.with_suffix(".pdf")
    if pdf.exists():
        pdf.unlink()
    with tempfile.TemporaryDirectory() as profile:
        proc = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--no-first-run",
                                 "--no-default-browser-check", "--no-pdf-header-footer",
                                 "--virtual-time-budget=15000", f"--user-data-dir={profile}",
                                 f"--print-to-pdf={pdf}", html_path.resolve().as_uri()],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0, last, stable = time.time(), -1, 0
        while time.time() - t0 < 240:
            if proc.poll() is not None:
                break
            size = pdf.stat().st_size if pdf.exists() else -1
            stable = stable + 1 if size > 0 and size == last else 0
            if stable >= 4:          # 2 s unchanged: written
                break
            last = size
            time.sleep(0.5)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    if not pdf.exists():
        sys.exit("Chrome did not write the PDF")
    return f"{pdf} {pdf.stat().st_size / 1e6:.2f} MB"
