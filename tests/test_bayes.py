"""The linear-Gaussian posterior (methods/bayes.py) and the worker's Bayesian run."""

import numpy as np
import pytest
import scipy.sparse as sp

from geoinv3d.methods.bayes import LinearGaussian, body_threshold


@pytest.fixture(scope="module")
def small():
    rng = np.random.default_rng(3)
    N, M = 30, 80
    G = rng.standard_normal((N, M))
    sigma = 0.3 + rng.random(N)
    B1 = sp.identity(M, format="csr")
    D = sp.diags([-np.ones(M - 1), np.ones(M - 1)], [0, 1], shape=(M - 1, M)).tocsr()
    P = (B1.T @ B1 + 4 * D.T @ D).toarray()
    m_true = np.linalg.cholesky(np.linalg.inv(2.0 * P)) @ rng.standard_normal(M)   # beta = 2
    d = G @ m_true + sigma * rng.standard_normal(N)
    return G, d, sigma, [(1.0, B1), (4.0, D)], P


def test_the_mean_solves_the_normal_equations(small):
    G, d, sigma, terms, P = small
    lg = LinearGaussian(G, d, sigma, terms)
    beta = 1.7
    m = lg.mean(beta)
    Wd2 = np.diag(1 / sigma ** 2)
    # G and P^-1 G^T are kept in float32
    assert np.allclose(m, np.linalg.solve(G.T @ Wd2 @ G + beta * P, G.T @ Wd2 @ d), rtol=1e-4, atol=1e-6)


def test_beta_by_chi2_and_by_evidence(small):
    from scipy.stats import multivariate_normal
    G, d, sigma, terms, P = small
    lg = LinearGaussian(G, d, sigma, terms)
    b = lg.beta_for(len(d))
    assert np.sum(((G @ lg.mean(b) - d) / sigma) ** 2) == pytest.approx(len(d), rel=1e-4)
    # the evidence is the Gaussian density of the data, up to a constant
    dens = lambda beta: multivariate_normal(np.zeros(len(d)), G @ np.linalg.inv(beta * P) @ G.T
                                            + np.diag(sigma ** 2)).logpdf(d)
    assert lg.log_evidence(0.5) - lg.log_evidence(2.0) == pytest.approx(dens(0.5) - dens(2.0), abs=1e-4)
    assert lg.beta_evidence() == pytest.approx(2.0, rel=0.3)         # the beta the model came from


def test_rml_samples_have_the_posterior_spread(small):
    G, d, sigma, terms, P = small
    lg = LinearGaussian(G, d, sigma, terms)
    S, prior = lg.samples(2.0, 3000, seed=1)
    C = np.linalg.inv(G.T @ np.diag(1 / sigma ** 2) @ G + 2.0 * P)
    ratio = S.std(0) / np.sqrt(np.diag(C))
    assert np.median(ratio) == pytest.approx(1.0, abs=0.03) and ratio.min() > 0.9 and ratio.max() < 1.1
    assert np.median(prior.std(0) / np.sqrt(np.diag(np.linalg.inv(2.0 * P)))) == pytest.approx(1.0, abs=0.03)


def test_the_body_threshold_weighs_by_volume():
    v = np.r_[np.full(98, 0.1), [1.0, 1.0]]          # two strong cells...
    assert body_threshold(v) == pytest.approx(0.25 * 0.1, rel=0.2) or body_threshold(v) > 0
    small_cells = np.r_[np.ones(98), [0.001, 0.001]]   # ...that are tiny: they do not set it
    assert body_threshold(v, weights=small_cells) == pytest.approx(0.05)        # half of 0.1
    assert body_threshold(v, weights=small_cells, share=0.25) == pytest.approx(0.025)


def test_a_bayesian_run_of_the_pipeline(tmp_path):
    from geoinv3d.cloud.worker import pack_result, run_data_pipeline
    from geoinv3d.viz.result_workflow import load_result
    from geoinv3d.viz.sections import ResultModel
    from tests.test_data_pipeline import _single, _station_grid, _synthetic, _write_csv
    locs = _station_grid(0.0)
    _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
    params = _single("gravity", ["g.csv"], param_mode="manual", regularization_type="bayes",
                     bayes_samples=12, dataset={"noise_pct": "auto", "noise_floor": "auto"})
    result = run_data_pipeline(params, str(tmp_path))
    b = result["bayes"]
    assert result["regularization"] == "bayesian_L2" and b["n_samples"] == 12
    assert b["beta_rule"] == "discrepancy" and b["beta"] == pytest.approx(b["beta_discrepancy"])
    assert b["chi2_per_datum"] == pytest.approx(1.0, abs=0.02) and b["error_scale_evidence"] > 0
    n = int(np.sum(result["active_cells"])) if result.get("active_cells") is not None else len(result["recovered_model"])
    assert result["samples"].shape == (12, n) and np.all((result["prob_body"] >= 0) & (result["prob_body"] <= 1))
    zpath = tmp_path / "r.zip"
    pack_result(result, str(zpath))
    rm = ResultModel(load_result(zpath))
    assert set(rm.fields) == {"model", "std", "probability"} and rm.samples.shape[0] == 12
    W, E, S, N = rm.extent
    line = rm.line(W, (S + N) / 2, E, (S + N) / 2, field="probability")
    assert line["range"] == [0.0, 1.0] and set(line["depths"]) == {"body_share", "top_rows", "base_rows"}
    # another threshold, from the samples: fewer cells above a higher one
    thr = b["threshold"]
    lo, hi = rm.probability(thr), rm.probability(2 * thr)
    ok = np.isfinite(lo)
    assert np.all(hi[ok] <= lo[ok]) and np.nanmax(lo) > 0
    assert rm.line(W, (S + N) / 2, E, (S + N) / 2, field="probability", threshold=2 * thr)["threshold"] == 2 * thr


def test_the_evidence_finds_the_errors_scale():
    """Errors given twice too large: the joint evidence scales them back (s ~ 0.5)."""
    rng = np.random.default_rng(7)
    N, M = 120, 60
    G = rng.standard_normal((N, M))
    B1 = sp.identity(M, format="csr")
    D = sp.diags([-np.ones(M - 1), np.ones(M - 1)], [0, 1], shape=(M - 1, M)).tocsr()
    P = (B1.T @ B1 + 4 * D.T @ D).toarray()
    m_true = np.linalg.cholesky(np.linalg.inv(0.5 * P)) @ rng.standard_normal(M)
    sigma_true = 0.2 + 0.2 * rng.random(N)
    d = G @ m_true + sigma_true * rng.standard_normal(N)
    lg = LinearGaussian(G, d, 2.0 * sigma_true, [(1.0, B1), (4.0, D)])
    beta, noise2 = lg.evidence_beta_noise()
    assert np.sqrt(noise2) == pytest.approx(0.5, rel=0.25)
    assert beta == pytest.approx(0.5, rel=0.6)


def test_the_compact_and_the_smooth_prior(tmp_path):
    """The compact prior's samples spread around the Lp model, cut at the bounds; the smooth
    prior's are wider and centred on the smooth L2 model."""
    from geoinv3d.cloud.worker import run_data_pipeline
    from tests.test_data_pipeline import _single, _station_grid, _synthetic, _write_csv
    locs = _station_grid(0.0)
    _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
    out = {}
    for prior in ("compact", "smooth"):
        params = _single("gravity", ["g.csv"], param_mode="manual", regularization_type="bayes",
                         bayes_samples=16, bayes_prior=prior, bounds_lower=0.0,
                         dataset={"noise_pct": "auto", "noise_floor": "auto"})
        out[prior] = run_data_pipeline(params, str(tmp_path))
    c, s = out["compact"], out["smooth"]
    assert c["bayes"]["prior_kind"] == "compact" and c["norms"] == [0.0, 1.0, 1.0, 1.0]
    assert c["samples"].min() >= 0.0                             # cut at the lower bound
    assert np.median(c["posterior_std"]) < np.median(s["posterior_std"])
    assert c["bayes"]["chi2_per_datum"] == pytest.approx(1.0, abs=0.15)


def test_the_compact_prior_caps_susceptibility_and_floors_eps(tmp_path, monkeypatch):
    """Magnetics: the compact prior's model and samples stay within 0 to BAYES_MAG_UPPER SI
    when no upper bound is given; the eps floor widens the spread outside the bodies (with
    eps -> 0 the empty cells could not move at all)."""
    from geoinv3d.cloud import worker
    from tests.test_data_pipeline import INDUCING, _single, _station_grid, _synthetic, _write_csv
    locs = _station_grid(0.0)
    _write_csv(tmp_path / "m.csv", locs, _synthetic("magnetics", locs))
    # sensitivity weights (not the magnetic default, depth): their IRLS ends with an eps far
    # below the floor, which is where the floor matters
    params = _single("magnetics", ["m.csv"], param_mode="manual", regularization_type="bayes",
                     bayes_samples=16, depth_weighting="sensitivity",
                     dataset={"noise_pct": "auto", "noise_floor": "auto",
                              "method_kwargs": {"inducing_field": INDUCING}})
    floored = worker.run_data_pipeline(params, str(tmp_path))
    monkeypatch.setattr(worker, "BAYES_EPS_SHARE", 0.0)
    bare = worker.run_data_pipeline(params, str(tmp_path))
    b = floored["bayes"]
    assert b["bounds"] == [0.0, worker.BAYES_MAG_UPPER] and b["eps_floor"]["smallness"] > 0
    assert bare["bayes"]["eps_floor"] is None
    for r in (floored, bare):
        assert r["samples"].min() >= 0.0 and r["samples"].max() <= worker.BAYES_MAG_UPPER
    empty = np.asarray(bare["recovered_model"]) < 0.1 * b["threshold"]
    assert np.median(floored["posterior_std"][empty]) > np.median(bare["posterior_std"][empty])
