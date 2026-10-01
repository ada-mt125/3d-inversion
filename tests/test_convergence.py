"""Whether a run converged: chi^2 = N within the tolerance, before its iteration limit."""

import pytest

from geoinv3d.cloud.worker import MISFIT_TOLERANCE, assess_convergence


def _result(phi_ds, label="sparse_IRLS", **extra):
    return {"regularization": label, "converged": True,
            "iterations": [{"iteration": i + 1, "phi_d": v} for i, v in enumerate(phi_ds)], **extra}


class TestAssessConvergence:
    def test_converged(self):
        r = _result([5000.0, 1200.0, 1040.0])
        c = assess_convergence(r, 1000, 30)
        assert c["status"] == "converged" and r["converged"] is True and not c["advice"]
        assert c["chi2_per_datum"] == pytest.approx(1.04) and c["target"] == 1000

    def test_the_iteration_limit_before_the_target(self):
        # the Karnataka auto run: 30 iterations, phi_d 7087 for 5040 data
        r = _result([9000.0] * 29 + [7087.0])
        c = assess_convergence(r, 5040, 30)
        assert c["status"] == "not_converged" and r["converged"] is False
        assert c["at_iteration_limit"] and "max_iter 60" in c["advice"][0]

    def test_fits_but_at_the_limit(self):
        r = _result([2000.0] * 9 + [990.0])
        c = assess_convergence(r, 1000, 10)
        assert c["status"] == "at_limit" and r["converged"] is False

    def test_the_irls_ended_above_the_target(self):
        r = _result([3000.0, 2000.0, 1500.0])
        c = assess_convergence(r, 1000, 30)
        assert c["status"] == "not_converged" and not c["at_iteration_limit"]
        assert "max_irls_iterations" in c["advice"][0]

    def test_tolerance(self):
        assert assess_convergence(_result([1000 * (1 + MISFIT_TOLERANCE)]), 1000, 30)["status"] == "converged"
        assert assess_convergence(_result([1000 * (1 + MISFIT_TOLERANCE) + 1]), 1000, 30)["status"] != "converged"

    def test_stopped_by_the_user(self):
        r = _result([1000.0], stopped_early={"reason": "stopped by the user", "at_iteration": 1})
        assert assess_convergence(r, 1000, 30)["status"] == "stopped" and r["converged"] is False

    @pytest.mark.parametrize("label", ["joint_sparse_IRLS", "joint_mixed", "joint_L2", "mvi_sparse_IRLS",
                                       "smooth_L2", "total_variation"])
    def test_discrepancy_principle_labels(self, label):
        assert assess_convergence(_result([1000.0], label), 1000, 30) is not None

    @pytest.mark.parametrize("label", ["elastic_net_CDA", "group_lasso_ADMM", "joint_pgi"])
    def test_runs_that_judge_themselves(self, label):
        r = _result([5000.0], label)
        assert assess_convergence(r, 1000, 30) is None and r["converged"] is True

    def test_beta_chosen_by_the_l_curve(self):
        r = _result([5000.0], beta_selection={"criterion": "lcurve"})
        assert assess_convergence(r, 1000, 30) is None
