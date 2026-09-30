"""The interpretive text of the joint report: written from the numbers of the runs.

Each function gets the helpers of build_report.py (F: the runs' numbers; LOW: the 2 km runs;
g, m, c: the gravity, magnetic and coupling measures of a run; has; pct; rng) and returns HTML.
"""

from __future__ import annotations


def _stopped(F, k):
    s = F[k].get("stopped") or {}
    return s.get("at_iteration"), s.get("of")


def coupling_notes(F, g, m, c, has, pct, rng, **_):
    base, out = c("none"), []
    if has("cross_gradient"):
        k = "cross_gradient"
        out.append(f"""<li><b>Cross-gradient.</b> It does what it is built for: the cross-gradient measure of Section 2.3 falls from {base['cross_gradient']:.2f} to {c(k)['cross_gradient']:.2f}, with both datasets fitted (χ²/N {g(k)['chi2']:.2f} and {m(k)['chi2']:.2f}). But it gets there mostly by the second way of making a cross product zero. The susceptibility model gives up its sharp sheets for a smooth shell that follows the upper boundary of the dense body, and only {pct(c(k)['edges_shared'])} of its edges now lie on an edge of the density model ({pct(base['edges_shared'])} uncoupled): where one model has an edge, the other is flat. The density model hardly changes (correlation {g(k)['corr_with_none']:.2f} with the uncoupled one); the susceptibility model changes a great deal ({m(k)['corr_with_none']:.2f}), and its centroid in the belt moves from {m('none')['box_sandur']['centroid_km']:.1f} to {m(k)['box_sandur']['centroid_km']:.1f} km, towards the dense body.</li>""")
    if has("joint_total_variation"):
        k = "joint_total_variation"
        out.append(f"""<li><b>Joint total variation.</b> It rewards edges in the same cells, and the share of susceptibility edges on a density edge rises from {pct(base['edges_shared'])} to {pct(c(k)['edges_shared'])}; the directions of the gradients are not constrained (measure {c(k)['cross_gradient']:.2f}). The magnetic rock is gathered into fewer, larger blocks inside the dense body; the density model stays as it was ({g(k)['corr_with_none']:.2f}). It costs data fit: χ²/N {g(k)['chi2']:.2f} for gravity and {m(k)['chi2']:.2f} for magnetics after {F[k]['n_iterations']} iterations.</li>""")
    if has("linear_correspondence"):
        k = "linear_correspondence"
        out.append(f"""<li><b>Linear correspondence.</b> The two models become one: {pct(c(k)['support']['magnetic_in_dense'])} of the magnetic cells are anomalous in density too, and |density| correlates with susceptibility at {c(k)['corr_abs_cells']:.2f}. The relation is wrong for this belt and the density model pays for it: the dense body is pulled up to the magnetic rock (half-maximum range {rng(g(k)['main'])} km under the main high, against {rng(g('none')['main'])} km), and what a shallow body cannot explain is put below the core ({pct(g(k)['below'])} of the density model, against {pct(g('none')['below'])}). Both datasets are still fitted, to χ²/N {g(k)['chi2']:.2f} and {m(k)['chi2']:.2f}: the fit does not warn.</li>""")
    if has("pgi"):
        k = "pgi"
        at, _ = _stopped(F, k)
        early = (f" The run was stopped by us after {at} of 60 iterations (68 minutes), before PGI had finished its schedule, so this is an intermediate model." if at else "")
        out.append(f"""<li><b>PGI.</b>{early} With the three rock units given, {pct(c(k)['support']['magnetic_in_dense'])} of the magnetic cells are dense or light, and the cross-plot (Section 2.3) shows the clusters of the units. χ²/N is {g(k)['chi2']:.2f} for gravity and {m(k)['chi2']:.2f} for magnetics. The models are much less compact than the sparse ones: only {pct(g(k)['core'])} of the density model and {pct(m(k)['core'])} of the susceptibility model lie in the core, the dense body under the main high spans {rng(g(k)['main'])} km, and the magnetic rock of the belt is centred at {m(k)['box_sandur']['centroid_km']:.1f} km, {m(k)['box_sandur']['centroid_km'] - m('none')['box_sandur']['centroid_km']:.1f} km deeper than uncoupled. PGI weights its cells by their sensitivity, the weighting that filled the bottom of the mesh in the single magnetic inversions, not by the depth weighting (β = 1) of the other runs; and its units were chosen by us from the rock samples and the single inversions, not measured.</li>""")
    if has("group_lasso"):
        k, ctrl = "group_lasso", "group_lasso_uncoupled"
        at, of = _stopped(F, k)
        early = (f" Stopped by us at the {at}th of {of} values of λ₁, after an hour on 72 cores: the model shown is the one of the smallest λ₁ reached, and it does not fit the data yet (χ²/N {g(k)['chi2']:.1f} for gravity, {m(k)['chi2']:.1f} for magnetics)." if at else f" χ²/N {g(k)['chi2']:.2f} for gravity and {m(k)['chi2']:.2f} for magnetics.")
        txt = f"""<li><b>Group lasso.</b>{early} Its coupling is visible all the same: {pct(c(k)['support']['magnetic_in_dense'])} of the magnetic cells are anomalous in density too"""
        if has(ctrl):
            txt += f""", against {pct(c(ctrl)['support']['magnetic_in_dense'])} in its control without the pairing"""
        txt += f""", and |density| correlates with susceptibility at {c(k)['corr_abs_cells']:.2f}. But {pct(m(k)['below'])} of the susceptibility model lies below the core and {pct(1 - g(k)['core'])} of the density model outside it: its weighting does not keep the model where the data resolve it (Section 5).</li>"""
        out.append(txt)
    return "\n  ".join(out)


def summary(F, LOW, S, g, m, c, has, pct, rng, **_):
    base = c("none")
    out = [f"""    <li><b>Without a coupling the joint inversion returns the two single inversions</b> (cell-by-cell correlation {S['gravity_corr']:.2f} and {S['magnetics_corr']:.2f}). The two uncoupled models have little in common cell by cell: {pct(base['support']['magnetic_in_dense'])} of the magnetic cells are dense or light, and {pct(base['edges_shared'])} of the susceptibility model's edges lie on an edge of the density model.</li>"""]
    if has("cross_gradient") and has("joint_total_variation"):
        out.append(f"""    <li><b>The two structural couplings act on the susceptibility model and leave the density model alone</b> (correlation with the uncoupled density {g('cross_gradient')['corr_with_none']:.2f} and {g('joint_total_variation')['corr_with_none']:.2f}; susceptibility {m('cross_gradient')['corr_with_none']:.2f} and {m('joint_total_variation')['corr_with_none']:.2f}). The cross-gradient makes the gradients parallel (measure {base['cross_gradient']:.2f} → {c('cross_gradient')['cross_gradient']:.2f}), but mostly by keeping the edges of the two models apart ({pct(c('cross_gradient')['edges_shared'])} shared); the joint total variation puts more of them in the same cells ({pct(c('joint_total_variation')['edges_shared'])}) at some cost of fit (gravity χ²/N {g('joint_total_variation')['chi2']:.2f}).</li>""")
    if has("linear_correspondence"):
        k = "linear_correspondence"
        out.append(f"""    <li><b>The linear correspondence forces one relation on the belt and spoils the density model:</b> the dense body is pulled up to the magnetic rock ({rng(g(k)['main'])} km instead of {rng(g('none')['main'])} km under the main high) and {pct(g(k)['below'])} of the density model ends below the core, while both datasets are still fitted (χ²/N {g(k)['chi2']:.2f} and {m(k)['chi2']:.2f}).</li>""")
    if has("pgi"):
        k = "pgi"
        at, _ = _stopped(F, k)
        out.append(f"""    <li><b>PGI gives models made of the rock units it is told about</b> ({pct(c(k)['support']['magnetic_in_dense'])} of the magnetic cells dense or light; χ²/N {g(k)['chi2']:.2f} and {m(k)['chi2']:.2f}), but spreads them: only {pct(g(k)['core'])} and {pct(m(k)['core'])} of the two models lie in the core. It is the slowest of the SimPEG couplings{f' and was stopped after {at} of 60 iterations' if at else ''}; its units are our assumptions.</li>""")
    if has("group_lasso"):
        k = "group_lasso"
        at, of = _stopped(F, k)
        u = LOW.get("group_lasso_unbounded")
        first = (f"Without bounds it put {u['gravity']['max']:.1f} g/cc into padding cells on the 2 km mesh; bounds were added to the solver for this test. " if u else "")
        lowfit = (f" On the 2 km mesh, run to the end, it overfits gravity (χ²/N {LOW['group_lasso']['gravity']['chi2']:.2f}) and underfits magnetics ({LOW['group_lasso']['magnetics']['chi2']:.2f})." if "group_lasso" in LOW else "")
        out.append(f"""    <li><b>The group lasso is not ready for field data of this size.</b> {first}With bounds it couples as intended ({pct(c(k)['support']['magnetic_in_dense'])} of the magnetic cells anomalous in density, {pct(c('group_lasso_uncoupled')['support']['magnetic_in_dense']) if has('group_lasso_uncoupled') else '—'} in its control), but {pct(m(k)['below'])} of its susceptibility model lies below the core, and the full-resolution run needed an hour on 72 cores{f' for {at} of its {of} values of λ₁, where we stopped it' if at else ''}.{lowfit}</li>""")
    out.append("""    <li><b>For this belt the structural assumption is the right one.</b> The magnetic rock (iron formation) lies on the margins of the dense rock (the greenstone pile), so the two models share boundaries, not values: a coupling that ties the values together, by one relation or by one support, moves one model to where the other is. The data fit does not tell the couplings apart, so the choice has to come from the geology.</li>""")
    out.append("""    <li><b>No coupling fixes the depth.</b> The depth of each model still follows its own regularization (β); a coupling moves one model towards the other, which helps only where they really are the same rock.</li>""")
    return "\n".join(out)


def discussion(F, g, m, c, has, pct, rng, **_):
    base = c("none")
    out = ["""<p>The single-method reports showed what the two datasets see in the Sandur belt: a thick body of dense, weakly magnetic rock (the greenstone pile; amphibolite and metabasalt samples of 2.9–3.0 g/cc and 0.0002–0.004 SI), and thin sheets of strongly magnetic rock along its margins and its south-eastern closure (the banded iron formation), which are dense too (3.4 g/cc) but too thin to dominate the gravity. A coupling is an assumption about how the two properties are related; on this belt the assumptions fare as follows.</p>""",
           "<ul class=\"plain\">"]
    out.append("""  <li><b>Same boundaries (structural couplings): consistent with the geology.</b> The iron formation bounds the greenstone, so edges of the susceptibility model should lie on edges of the density model. The joint total variation asks for exactly that and delivers it in part; the cross-gradient asks only for parallel gradients and is satisfied, more cheaply, by a smooth susceptibility model whose edges avoid those of the density model. Both leave the density model unchanged: the gravity data and their regularization determine it, and the magnetic sheets are too thin to reshape it.</li>""")
    if has("linear_correspondence"):
        out.append("""  <li><b>One relation between the values (linear correspondence): contradicted.</b> The densest rock here is not the most magnetic. Forcing density = 0.5 × susceptibility moves the dense body into the magnetic sheets and leaves the rest of the gravity anomaly to the padding. A relation fitted to samples would fail in the same way, because two rock types with different relations make up the belt.</li>""")
    if has("pgi"):
        out.append("""  <li><b>A few rock units (PGI): the right idea, given the right units.</b> Two units, a dense magnetic one and a dense non-magnetic one, describe this belt, and PGI is the only coupling that can hold both. It needs their values: here the susceptibility of the iron formation at the scale of a cell is unknown (0.05 SI in the hand samples, 0.3–1 SI in the inversions), so the run shows the consequences of our choice. It also needs the depth weighting of the other runs, in place of its sensitivity weighting, to keep the units in the core.</li>""")
    if has("group_lasso"):
        k = "group_lasso"
        out.append(f"""  <li><b>The same cells are anomalous (group lasso): partly true.</b> The iron formation is both dense and magnetic, the greenstone is dense only. The group lasso lets a cell be anomalous in one property and not the other, at a price, so it does not force the greenstone to be magnetic; but it pulls the magnetic rock into the dense cells ({pct(c(k)['support']['magnetic_in_dense'])} of the magnetic cells are anomalous in density, against {pct(base['support']['magnetic_in_dense'])} uncoupled). On the synthetic test of 30 September it was the best coupling because there every magnetic body was a dense body; here most of the dense rock is not magnetic.</li>""")
    out.append("</ul>")
    out.append("""<p>The common point: a coupling improves a model where its assumption holds and damages it where it does not, and the data fit says little about which is the case. The measures of Section 2.3 say what a coupling did, not whether it was right.</p>""")
    return "\n".join(out)


def low_text(LOW, pct, **_):
    out = ["<ul class=\"plain\">"]
    if "pgi" in LOW and "none" in LOW:
        out.append(f"""  <li><b>The coarse runs behave like the full-resolution ones</b> for the four SimPEG couplings that finished at both resolutions: the cross-gradient lowers the cross-gradient measure, the joint total variation raises the share of common edges, the linear correspondence ties the values. On a laptop the uncoupled run took {LOW['none']['seconds'] / 60:.0f} minutes, the cross-gradient {LOW['cross_gradient']['seconds'] / 60:.0f} and PGI {LOW['pgi']['seconds'] / 60:.0f}: enough to choose a coupling before paying for the full runs.</li>""")
        out.append(f"""  <li><b>PGI on the coarse mesh ran to the end</b> (60 iterations, χ²/N {LOW['pgi']['gravity']['chi2']:.2f} and {LOW['pgi']['magnetics']['chi2']:.2f}); like the full-resolution run it leaves much of the susceptibility model outside the core ({pct(LOW['pgi']['magnetics']['core'])} inside).</li>""")
    if "group_lasso_unbounded" in LOW:
        u = LOW["group_lasso_unbounded"]
        out.append(f"""  <li><b>Without bounds the group lasso is not usable on these data.</b> Its first version had none, and on the 2 km mesh it returned density contrasts from {u['gravity']['min']:.2f} to {u['gravity']['max']:.1f} g/cc and negative susceptibilities (to {u['magnetics']['min']:.2f} SI), with {pct(1 - u['gravity']['core'])} of the density model outside the core: the large padding cells, which the data barely see, take whatever values fit. On the synthetic tests the padding was small and this did not show.</li>""")
        out.append("""  <li><b>Bounds were added to the solver</b> (<code>GroupLassoProblem(bounds=...)</code>, used by the pipeline with each model's bounds). In ADMM's shrinkage step a sign constraint, such as susceptibility ≥ 0, is projected before the group shrink, which is the exact proximal step; other bounds are clipped after it. The group lasso runs of this report use the bounds of the other couplings.</li>""")
    if "group_lasso" in LOW:
        a = LOW["group_lasso"]
        txt = f"""  <li><b>With bounds, two problems remain.</b> The fit is unbalanced: gravity χ²/N {a['gravity']['chi2']:.2f}, magnetics {a['magnetics']['chi2']:.2f} (the paper's balance of the datasets by their largest amplitudes, λ₁ from the L-curve)"""
        if "group_lasso_std" in LOW:
            b = LOW["group_lasso_std"]
            txt += f"""; weighting every datum by its error and choosing λ₁ for χ² = N does not repair it ({b['gravity']['chi2']:.2f} and {b['magnetics']['chi2']:.2f}: even the smallest λ₁ of the path leaves the magnetic data underfitted)"""
        txt += f""". And the models leave the core: {pct(a['gravity']['core'])} of the density model and {pct(a['magnetics']['core'])} of the susceptibility model are inside it. The group lasso scales every cell by the norm of its sensitivity column, which makes all cells equally easy to use; the sparse inversions of this report needed a much weaker depth weighting (β = 1) to stay in the core.</li>"""
        out.append(txt)
    out.append("</ul>")
    return "\n".join(out)


def problems(F, LOW, g, m, has, pct, **_):
    """What went wrong during this work, what was fixed and what is left open."""
    rows = [
        ("Group lasso: no bounds", "Unphysical values in the padding (21.9 g/cc, negative susceptibility) on the 2 km mesh.",
         "Fixed: bounds in the ADMM solver and in the pipeline; tests added (TestBounds)."),
        ("Group lasso: unbalanced fit", "Gravity overfitted, magnetics underfitted, with the paper's amplitude scaling and with error weighting (2 km mesh).",
         "Open. Needs a separate weight or target per dataset (as the SimPEG path has), or a wider λ₁ path."),
        ("Group lasso: model outside the core", "More than 80% of the susceptibility model outside the core, at both resolutions; the sensitivity weighting (γ = 2) has no preference for the cells the data resolve.",
         "Open. Tests of γ = 0.5, 1 and 1.5 on the 2 km mesh were started and stopped unfinished; the settings are in <code>joint_params.py</code>."),
        ("Group lasso: cost at full resolution", "28 minutes of set-up and about 5 minutes per value of λ₁ on 72 cores; stopped at the 9th of 13 values. Two further runs with error weighting were stopped after 35 minutes, at the 1st and 2nd value, and are not reported.",
         "Open. A shorter λ₁ path (or a fixed λ₁ from the 2 km run) and the error weighting would make it affordable."),
        ("PGI: slow, and not compact", "1.4–1.6 minutes per iteration on 36 cores; stopped after 43 of 60 iterations. Half of both models outside the core.",
         "Open. Run it to the end; weight it by depth (β = 1) instead of by sensitivity; get measured units."),
        ("Joint total variation: fit", "Gravity χ²/N 1.26 after 44 iterations: the target was not reached within the iteration limit.",
         "Open. More iterations or a smaller coupling weight."),
        ("Magnetic data: systematic residual", "The high south of the Sandur belt is underfitted in every single and joint run.",
         "Open. Points to remanent magnetization; needs a magnetization-vector inversion."),
        ("EC2: instances lost at launch", "With eight launches at once, four instances were reported as terminated seconds after launch and left running untracked.",
         "Fixed in <code>cloud/ec2.py</code> (a just-launched instance unknown to describe_instances is 'launching'); the four were terminated by hand and rerun."),
        ("Stations inside the ground", "With the ground as a staircase of cells, 1,900 of 5,040 gravity stations lay inside the top cell of their column.",
         "Fixed: such stations are moved to the top of their column (72 m on average)."),
    ]
    head = "<div class=\"tablewrap\"><table><thead><tr><th>Problem</th><th>What happened</th><th>State</th></tr></thead><tbody>"
    body = "".join(f"<tr><td class=\"wrap\"><b>{a}</b></td><td class=\"wrap\">{b}</td><td class=\"wrap\">{c_}</td></tr>" for a, b, c_ in rows)
    return head + body + "</tbody></table></div>"


def recommend(has, **_):
    out = ["""  <li><b>Choose the coupling from the geology, not from the fit.</b> Same rock in both models (one body, dense and magnetic): group lasso once it is fixed, or PGI with measured units. Different rock sharing boundaries, as in this belt: a structural coupling. Unknown: invert separately first and compare the two models, as in Section 5 of the magnetic report.</li>""",
           """  <li><b>For this area: the joint total variation or the cross-gradient,</b> read together with the uncoupled models. Report what is common to them; do not read the coupled susceptibility model as better resolved.</li>""",
           """  <li><b>Do not use the linear correspondence</b> unless one rock type carries both anomalies.</li>""",
           """  <li><b>PGI needs petrophysics and a depth weighting:</b> susceptibility measurements of the fresh iron formation and of the greenstone at the scale of a cell (logs, not hand samples) would turn its units from an assumption into a constraint.</li>""",
           """  <li><b>Group lasso: finish it on the 2 km mesh before running it at full resolution again.</b> Bounds are in; the balance between the two datasets and the weighting that keeps the model in the core are not (Section 5).</li>""",
           """  <li><b>Depth still needs independent information.</b> The couplings do not supply it; the mapped dips of the iron formation, boreholes, or a magnetization-vector inversion for the remanence would.</li>"""]
    return "\n".join(out)
