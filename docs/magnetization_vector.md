# Magnetization-vector inversion (MVI)

A susceptibility inversion assumes that every cell is magnetized along the present field.
Rock with remanent magnetization (banded iron formation, dykes, basalts) is not, and its
anomaly then has a shape — the ratio of its high to its low, their positions — that no
induced model can make.  On the Karnataka magnetic data every susceptibility model left the
positive anomaly on the southern side of the Sandur belt underfitted by more than 150 nT
(magnetic report, Section 5; joint report, Section 5).

`MagneticsMethod(magnetization="vector")` inverts for a magnetization vector per cell instead:
SimPEG's Cartesian formulation (`Simulation3DIntegral(model_type="vector")`), the model
`[m_x | m_y | m_z]` in effective susceptibility (SI) along east, north and up, so that
`M = |m| H₀` points along `m`, in any direction.

## The inversion (`worker.run_mvi_inversion`)

| | |
|---|---|
| regularization | SimPEG `VectorAmplitude`: the task's norms (`sparse`) or 2 (`l2`) on the amplitude \|m\| and its gradients, alphas and length scales as for the scalar runs, reference 0 |
| weighting | the task's depth weighting (Li & Oldenburg, `depth_weighting_exponent`) or sensitivity weights |
| bounds | `bounds_upper` bounds each component, \|m_i\| ≤ upper (so \|m\| ≤ √3 upper); `bounds_lower` does not apply |
| β | cooled to χ² = N, then steered there during IRLS (`DampedUpdateIRLS`), as for the scalar sparse inversion |
| result | `recovered_model` = the amplitude \|m\| per cell (what the viewer and the core-share measures use), `magnetization_vector` (n × 3, packed as `magnetization_vector.npy`), `magnetization`: inclination and declination of the amplitude-weighted resultant of the cells above 10 % of the largest amplitude, their coherence (1 = all parallel) and the cells' median direction, next to the inducing field's |

Pipeline: a magnetic dataset with `"method_kwargs": {"inducing_field": [...], "magnetization":
"vector"}`, regularization `sparse` or `l2`, β by χ² = N.  Upload page: the magnetic card's
*Magnetization* select.  Single magnetic inversions only: the joint inversions and the group
lasso couple the induced susceptibility and refuse a vector model.

## Checks

- `tests/test_mvi.py`: a block magnetized at I −30°, D 120° under a field of I 60°, D 0°.  The
  induced inversion (χ ≥ 0) cannot fit it (χ²/N > 3); MVI fits it (χ²/N < 1.5), finds its
  direction within 30° (−33°, 122° on the run), and centres its amplitude on the block (30 % of
  the amplitude in 5.6 % of the cells; about 1.5 cells too deep, as MVI models tend to be).
- Karnataka, 2 km mesh (1,296 data, β = 1, \|m_i\| ≤ 1 SI): RMS 47 → 26 nT against the induced
  inversion; at the 29 stations the induced model underfits by more than 150 nT (easting
  655–707 km, northing 1655–1675 km: the south of the Sandur belt) the RMS residual falls
  from 280 to 123 nT, and 5 of them stay off by more than 150 nT (the signed mean, 260 → 70 nT,
  hides residuals of both signs).  The strong cells point at I 74°, D −49° (coherence 0.73), far
  from the field (I 19°, D −1°): the data ask for remanence.  9 % of the amplitude lies outside
  the core (47 % of the induced model).
- Karnataka, full resolution (5,041 data, 335,518 cells × 3, c5.9xlarge, 27 min): RMS 49 → 28 nT,
  largest residual 680 → 309 nT; at the 92 stations the induced model underfits by more than
  150 nT the RMS falls from 321 to 129 nT and 22 stay off by more than 150 nT (16 above, 6 below).
  The strong cells point at I 74°, D −79° (coherence 0.75); 79 % of the amplitude lies in the core
  (71 % of the induced model; the joint report's measure).

## Not in this version

The spherical (amplitude, inclination, declination) stage that SimPEG's examples run after the
Cartesian one, amplitude-data inversion, and a vector model in the joint inversions.
