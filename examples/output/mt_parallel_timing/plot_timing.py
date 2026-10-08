"""MT's frequencies on 1 and on 3 processes (LOGBOOK 2026-10-08): run times and memory.

    py examples/output/mt_parallel_timing/plot_timing.py

Reads results_1process.json and results_3processes.json (examples/synthetic_builder.py --only mt
--quick --mt-workers 1 | 3) and memory_samples.csv (the Python processes' working sets every 2 s),
writes timing.png.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = Path(__file__).resolve().parent
ONE, THREE = "#9aa7b3", "#1f5f8b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def main():
    r1 = json.loads((HERE / "results_1process.json").read_text())["runs"]
    r3 = json.loads((HERE / "results_3processes.json").read_text())["runs"]
    runs = [k for k in r1 if k.startswith("mt")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1, 1.6]})
    a = ax[0]
    for i, k in enumerate(runs):
        s1, s3 = r1[k]["seconds"], r3[k]["seconds"]
        a.barh(i + 0.2, s1, height=0.38, color=ONE, label="1 process" if i == 0 else None)
        a.barh(i - 0.2, s3, height=0.38, color=THREE, label="3 processes (4 threads each)" if i == 0 else None)
        a.text(s1 + 10, i + 0.2, f"{s1:.0f} s", va="center", fontsize=8)
        a.text(s3 + 10, i - 0.2, f"{s3:.0f} s  ({s1 / s3:.1f}x)", va="center", fontsize=8, color=THREE)
    a.set_yticks(range(len(runs)), [k.replace("mt ", "") for k in runs])
    a.set_xlim(0, 950)
    a.set_xlabel("seconds (2 Gauss-Newton iterations)")
    a.set_title("(a) time: 28,392 cells, 3 frequencies, PARDISO")
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    rows = list(csv.DictReader(open(HERE / "memory_samples.csv", encoding="utf-8-sig")))
    t = [datetime.strptime(r["time"], "%H:%M:%S") for r in rows]
    t0 = t[0]
    m = [(x - t0).total_seconds() / 60 for x in t]
    tot = [float(r["total_gb"]) for r in rows]
    n = [int(r["n_python"]) for r in rows]
    a = ax[1]
    split = next(i for i, k in enumerate(n) if k > 2)          # the processes start
    a.plot(m[:split], tot[:split], color=ONE, lw=1.2, label=f"1 process (peak {max(tot[:split]):.2f} GB)")
    a.plot(m[split:], tot[split:], color=THREE, lw=1.2,
           label=f"3 processes, all together (peak {max(tot[split:]):.2f} GB)")
    a.set_xlabel("minutes")
    a.set_ylabel("memory of the Python processes (GB)")
    a.set_title("(b) memory, sampled every 2 s (data made first, then two inversions)")
    a.legend(frameon=False, loc="upper left", fontsize=8)
    a.set_ylim(0, 11)
    fig.savefig(HERE / "timing.png", dpi=150)
    print(HERE / "timing.png")


if __name__ == "__main__":
    main()
