# Block-8 window: which regularization, and which settings?

Trials of the magnetic inversion on the user's 5 km window of Block-8 (TAIL_TMI_GE, 37.5 m,
flown 80 m above the ground), to help choose a regularization and its parameters.

Window (UTM 43N): E 659,650–664,650, N 1,664,550–1,669,550, found again from the extremes the
LOGBOOK gives for the user's job of 2 October (−4,132 to +8,864 nT; the east edge −4,132 to −647 nT).

## How each trial is judged

* **Cross-validation.** Every trial inverts every second node (75 m, 4,489 data) and predicts the
  other 13,467 nodes from its model. χ²/N on both, with the same errors (2 % + 1.5 % of the 5–95 %
  spread = 85 nT): a model that fits noise fits the nodes it saw better than those it did not.
* **Where the model is.** The share of |χ| × volume by depth below the ground under the window,
  and outside the core (beside the window, below 4 km).
* **A synthetic test** (`syn-*` trials): a known model on the same mesh and stations (two NW–SE
  iron-formation bands dipping 70° NE from the ground to 400–600 m, a deeper block at
  700–1,200 m), its data with the same errors; scored against the truth.

## Scripts

    py scripts/prepare_window.py          # data/inputs: TMI around the window, GLO-90 ground
    py scripts/trials.py run [NAME ...]   # the inversions (data/runs/NAME)
    py scripts/synthetic.py               # the synthetic model and data (after the trial default)
    py scripts/trials.py run syn-NAME ... # the synthetic trials
    py scripts/trials.py score [NAME ...] # cross-validation and depth (score.json)
    py scripts/make_figures.py            # figures/
    py scripts/build_report.py [--pdf]    # the report

`trials.py list` lists the trials. Two series: `data/runs/` (2 October, the former defaults:
sensitivity weighting, length scales 1, bounds [0, 2], no trend) and `data/runs_new_defaults/`
(8 October, the defaults tuned on 5 October: Lp (1,1,1,1), length scale 3, depth weighting
beta 1.5, an order-2 trend removed, bounds [0, 3]); `figures_2026-10-02_old_defaults/` and
`figures/` are theirs.

    py scripts/build_workflow.py          # the DAG viewer of the runs (not in git: > 100 MB)

Each run keeps result.zip (the worker's result; its result.json inside), params.json, run.json,
score.json and model_grid.npz; the loose result.json and the viewers are left out of git.

The sensitivity matrix is computed once, in 12 processes
(without choclo SimPEG uses one), and cached in `data/cache/` (0.9 GB; not kept in git).
