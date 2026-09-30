"""Parameters of the Karnataka joint gravity-magnetic inversions with terrain.

Shared by the local 2 km study (run_lowres.py) and the EC2 runs (deploy/ec2_multi_run.py,
case "karnataka-joint").  Both datasets, their errors, the terrain and each model's own
regularization are those of the single inversions (karnataka_gravity_terrain and
karnataka_magnetic, sparse with alpha_s = 1 and depth weighting beta = 1); only the coupling
differs between the runs.
"""

from __future__ import annotations

FIELD = [42100.0, 19.3, -1.4]       # IGRF 2020 at the centre; declination to grid north

# each model's regularization: as in its single inversion
GRAVITY_REG = {"regularization_type": "sparse", "norms": [0, 2, 2, 2], "alpha_s": 1.0,
               "depth_weighting": "depth", "depth_weighting_exponent": 1.0,
               "bounds_lower": -0.2, "bounds_upper": 0.5}
MAGNETIC_REG = {"regularization_type": "sparse", "norms": [0, 2, 2, 2], "alpha_s": 1.0,
                "depth_weighting": "depth", "depth_weighting_exponent": 1.0,
                "bounds_lower": 0.0, "bounds_upper": 1.0}


def base(core_cell_m=1000.0, core_cell_z_m=250.0, decimate=None):
    thin = {"decimate_spacing_m": decimate} if decimate else {}
    return {
        "method_type": "joint", "inversion_mode": "joint", "crs": "EPSG:32643",
        "datasets": [
            {"method": "gravity", "files": ["gravity_complete_1km.csv"], "component": "gz",
             "noise_pct": 0.0, "noise_floor": 0.5, "regional": {"method": "polynomial", "order": 2},
             "regularization": GRAVITY_REG, **thin},
            {"method": "magnetics", "files": ["magnetic_1km.csv"], "component": "tmi",
             "method_kwargs": {"inducing_field": FIELD}, "noise_pct": 0.05, "noise_floor": 10.0,
             "regional": {"method": "polynomial", "order": 2}, "regularization": MAGNETIC_REG, **thin}],
        "joint_weights": [1.0, 1.0],
        "topography": {"file": "dem_utm43n_450m.tif"}, "mesh_type": "tensor",
        "core_cell_m": core_cell_m, "core_cell_z_m": core_cell_z_m, "depth_core_m": 10000.0,
        "pad_distance_m": 20000.0,
        "param_mode": "manual", "regularization_type": "sparse", "norms": [0, 2, 2, 2],
        "alpha_s": 1.0, "alpha_x": 1.0, "alpha_y": 1.0, "alpha_z": 1.0,
        "max_iter": 60, "max_irls_iterations": 40,
    }


# Rock units for PGI (density contrast to 2.67 g/cc, susceptibility SI).  Densities from the
# rock samples of the area; the susceptibility of the iron formation is what the magnetic
# data need at the scale of a 1 km cell (the hand samples give 0.05 SI, the single
# inversions 0.3-1 SI), not a measured value.
PGI_UNITS = [
    {"name": "iron formation", "means": {"gravity": 0.5, "magnetics": 0.5}, "proportion": 0.01},
    {"name": "greenstone", "means": {"gravity": 0.3, "magnetics": 0.005}, "proportion": 0.05},
    {"name": "light granite", "means": {"gravity": -0.15, "magnetics": 0.002}, "proportion": 0.04},
]

COUPLINGS = {
    "none": {"coupling": "none"},
    "cross_gradient": {"coupling": "cross_gradient", "coupling_weight": 1.0},
    "joint_total_variation": {"coupling": "joint_total_variation", "coupling_weight": 1.0},
    # one relation for every cell, the ratio of the two upper bounds: density = 0.5 x susceptibility
    "linear_correspondence": {"coupling": "linear_correspondence", "coupling_weight": 1.0,
                              "coupling_options": {"slope": 0.5, "intercept": 0.0}},
    "pgi": {"coupling": "pgi", "coupling_options": {"units": PGI_UNITS}},
    "group_lasso": {"coupling": "group_lasso", "gl_lambda1_selection": "lcurve",
                    "gl_lambda2": 0.3, "gl_data_scaling": "max_ratio"},
    # the group lasso's control: L1 + L2 per value by ADMM, no pairing
    "group_lasso_uncoupled": {"coupling": "group_lasso", "gl_coupling": "none",
                              "gl_lambda1_selection": "lcurve", "gl_lambda2": 0.3,
                              "gl_data_scaling": "max_ratio"},
    # the same two with the data weighted by their errors and lambda1 chosen for chi^2 = N,
    # as the other couplings are: the settings above (the paper's) balance the two datasets
    # by their largest amplitudes, which here overfits the gravity and underfits the magnetics
    "group_lasso_std": {"coupling": "group_lasso", "gl_lambda1_selection": "discrepancy",
                        "gl_lambda2": 0.3, "gl_data_scaling": "std"},
    "group_lasso_std_uncoupled": {"coupling": "group_lasso", "gl_coupling": "none",
                                  "gl_lambda1_selection": "discrepancy", "gl_lambda2": 0.3,
                                  "gl_data_scaling": "std"},
}
# The group lasso weights every cell by 1 / ||its sensitivity column||^(gamma / 2); gamma = 2 (the
# paper) makes all columns equal, which leaves nothing to keep the model out of the padding.
# Smaller exponents favour the cells the data see best (tests on the 2 km mesh).
for _gamma in (0.5, 1.0, 1.5):
    COUPLINGS[f"group_lasso_std_g{_gamma:g}"] = {**COUPLINGS["group_lasso_std"], "gl_gamma": _gamma}
