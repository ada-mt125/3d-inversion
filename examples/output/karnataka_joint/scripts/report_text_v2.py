"""The interpretive text of the second joint report, written from the numbers of the runs.

Each function gets the helpers of build_report_v2.py (F: the full-resolution runs; LOW: the 2 km
runs; S: the single inversions; MAG: the magnetic runs with and without a magnetization vector;
GL: the group lasso's settings; g, m, c: the gravity, magnetic and coupling measures of a run;
has; pct; rng; GLK / CTRL: the keys of the group lasso and its control) and returns HTML.
"""

from __future__ import annotations


def _within(x):
    return 0.7 <= x <= 1.3


def _dir(a, b, up="rises", down="falls"):
    return up if b > a else down


def fit_text(F, S, KEYS, g, m, c, **_):
    out = []
    bad = [k for k in KEYS if not (_within(g(k)["chi2"]) and _within(m(k)["chi2"]))]
    lo = min(min(g(k)["chi2"], m(k)["chi2"]) for k in KEYS)
    hi = max(max(g(k)["chi2"], m(k)["chi2"]) for k in KEYS)
    if not bad:
        out.append(f"<p>Every run fits both datasets to their errors: χ²/N lies between {lo:.2f} and {hi:.2f} for "
                   "gravity and magnetics alike. The data fit therefore does not rank the couplings; what they do "
                   "differently shows in the models (Sections 2.2 and 2.3).</p>")
    else:
        out.append(f"<p>χ²/N lies between {lo:.2f} and {hi:.2f}; outside 0.7–1.3: "
                   + ", ".join(bad) + ".</p>")
    out.append(f"<p>Inverted together without a coupling, the two models are those of the single inversions "
               f"(cell-by-cell correlation {S['gravity_corr']:.2f} for density and {S['magnetics_corr']:.2f} for "
               "susceptibility): the joint set-up itself changes nothing. The positive magnetic residual south of the "
               "Sandur belt is there in every run: no coupling repairs what a magnetization along the present field "
               "cannot explain (Section 3).</p>")
    return "\n".join(out)


def coupling_notes(F, g, m, c, has, pct, rng, GL, GLK, CTRL, **_):
    base, out = c("none"), []
    if has("cross_gradient"):
        k = "cross_gradient"
        out.append(f"""<li><b>Cross-gradient.</b> It does what it is built for: the cross-gradient measure of Section 2.3 falls from {base['cross_gradient']:.2f} to {c(k)['cross_gradient']:.2f}, with both datasets fitted (χ²/N {g(k)['chi2']:.2f} and {m(k)['chi2']:.2f}). It gets there mostly by the second way of making a cross product zero: only {pct(c(k)['edges_shared'])} of the susceptibility model's edges lie on an edge of the density model ({pct(base['edges_shared'])} uncoupled), so where one model has an edge the other is flat. The density model hardly changes (correlation {g(k)['corr_with_none']:.2f} with the uncoupled one); the susceptibility model changes more ({m(k)['corr_with_none']:.2f}), and its centroid in the belt moves from {m('none')['box_sandur']['centroid_km']:.1f} to {m(k)['box_sandur']['centroid_km']:.1f} km.</li>""")
    if has("joint_total_variation"):
        k = "joint_total_variation"
        out.append(f"""<li><b>Joint total variation.</b> It rewards edges in the same cells, and the share of susceptibility edges on a density edge {_dir(base['edges_shared'], c(k)['edges_shared'])} from {pct(base['edges_shared'])} to {pct(c(k)['edges_shared'])}; the directions of the gradients are not constrained (measure {c(k)['cross_gradient']:.2f}). {pct(c(k)['support']['magnetic_in_dense'])} of the magnetic cells are anomalous in density ({pct(base['support']['magnetic_in_dense'])} uncoupled). The density model stays as it was ({g(k)['corr_with_none']:.2f}); both datasets are fitted (χ²/N {g(k)['chi2']:.2f} and {m(k)['chi2']:.2f}) in {F[k]['n_iterations']} iterations.</li>""")
    if has("linear_correspondence"):
        k = "linear_correspondence"
        out.append(f"""<li><b>Linear correspondence.</b> The two models become one: {pct(c(k)['support']['magnetic_in_dense'])} of the magnetic cells are anomalous in density too, and |density| correlates with susceptibility at {c(k)['corr_abs_cells']:.2f}. The relation is wrong for this belt and the density model pays for it: the dense body under the main high has its half-maximum at {rng(g(k)['main'])} km, against {rng(g('none')['main'])} km uncoupled, and what the relation cannot place is put below the core ({pct(g(k)['below'])} of the density model, against {pct(g('none')['below'])}). Both datasets are still fitted, to χ²/N {g(k)['chi2']:.2f} and {m(k)['chi2']:.2f}: the fit does not warn.</li>""")
    if has(GLK):
        k = GLK
        info = GL.get(k) or {}
        w = info.get("data_weights") or []
        rounds = max(len(info.get("balance") or []) - 1, 0)
        txt = (f"""<li><b>Group lasso.</b> Both datasets are fitted (χ²/N {g(k)['chi2']:.2f} and {m(k)['chi2']:.2f})"""
               + (f""", after {rounds} round{'s' if rounds != 1 else ''} of reweighting the datasets (weights {w[0]:.2f} for gravity and {w[1]:.2f} for magnetics: along the λ₁ sweep the gravity data are fitted sooner)""" if w and rounds else "")
               + f""", and the models stay where the data resolve them ({pct(g(k)['core'])} of the density and {pct(m(k)['core'])} of the susceptibility model in the core). The pairing shows: {pct(c(k)['support']['magnetic_in_dense'])} of the magnetic cells are anomalous in density""")
        if has(CTRL):
            txt += f""", against {pct(c(CTRL)['support']['magnetic_in_dense'])} in its control without the pairing and {pct(base['support']['magnetic_in_dense'])} in the uncoupled SimPEG run"""
        txt += (f""", and |density| correlates with susceptibility at {c(k)['corr_abs_cells']:.2f} ({c('none')['corr_abs_cells']:.2f} uncoupled). It also makes the two structures alike: {pct(c(k)['edges_shared'])} of the susceptibility edges lie on a density edge ({pct(base['edges_shared'])} uncoupled), more than with any coupling but the linear correspondence, and the cross-gradient measure falls to {c(k)['cross_gradient']:.2f} ({base['cross_gradient']:.2f} uncoupled""" + (f"""; {c('cross_gradient')['cross_gradient']:.2f} for the cross-gradient itself""" if has("cross_gradient") else "") + """). """)
        if g(k)["corr_with_none"] < 0.85:
            ctrl_main = f"; its control {rng(g(CTRL)['main'])} km" if has(CTRL) else ""
            txt += (f"""The pairing reshapes the density model too (correlation {g(k)['corr_with_none']:.2f} with the uncoupled one): the dense body under the main high rises to a half-maximum at {rng(g(k)['main'])} km (uncoupled {rng(g('none')['main'])} km{ctrl_main}; centroid {g(k)['main']['centroid_km']:.1f} km against {g('none')['main']['centroid_km']:.1f} km), towards the magnetic rock, and {pct(g(k)['below'])} of the density model lies below the core ({pct(g('none')['below'])} uncoupled). """)
        else:
            txt += f"""The density model follows the uncoupled one (correlation {g(k)['corr_with_none']:.2f}). """
        txt += (f"""The susceptibility model spreads through the dense body (correlation {m(k)['corr_with_none']:.2f} with the uncoupled one; centroid in the belt {m(k)['box_sandur']['centroid_km']:.1f} km, against {m('none')['box_sandur']['centroid_km']:.1f} km).</li>""")
        out.append(txt)
    if has(CTRL):
        k = CTRL
        out.append(f"""<li><b>The group lasso's control</b> (the same solver, weighting and balance, each model on its own) fits the data to χ²/N {g(k)['chi2']:.2f} and {m(k)['chi2']:.2f} and keeps {pct(g(k)['core'])} and {pct(m(k)['core'])} of its models in the core; {pct(c(k)['support']['magnetic_in_dense'])} of its magnetic cells are anomalous in density. What the group lasso adds to it is the pairing of the two supports.</li>""")
    return "\n  ".join(out)


def mvi_text(MAG, pct, **_):
    if not MAG:
        return "<p>(The magnetization-vector run is not available.)</p>"
    a, b, u = MAG["induced"], MAG["mvi"], MAG["underfit"]
    z = b.get("magnetization") or {}
    J = MAG.get("joint_underfit") or {}
    return "\n".join([
        "<p>A susceptibility model magnetizes every cell along the present field, and a body so magnetized gives a fixed "
        "shape to its anomaly: the ratio of its high to its low, and where they lie. The positive anomaly on the southern "
        f"side of the Sandur belt is fitted worse than the rest: the single magnetic inversion underfits {u['n']} stations "
        f"by more than {u['threshold_nT']:.0f} nT (easting {u['extent_km'][0]:.0f}–{u['extent_km'][1]:.0f} km, northing "
        f"{u['extent_km'][2]:.0f}–{u['extent_km'][3]:.0f} km; RMS {u['induced_rms_nT']:.0f} nT there)"
        + (f", and every joint run leaves {min(v['n_over'] for v in J.values())}–{max(v['n_over'] for v in J.values())} "
           f"such stations, with an RMS of {min(v['rms_at_single'] for v in J.values()):.0f}–"
           f"{max(v['rms_at_single'] for v in J.values()):.0f} nT at the same {u['n']}" if J else "")
        + ".</p>",
        "<p>The magnetic data were therefore inverted once more with a magnetization vector in every cell (SimPEG's "
        "magnetization-vector inversion: three components of effective susceptibility per cell, a sparse regularization of "
        "their amplitude with the same norms, depth weighting and bounds, |m<sub>i</sub>| ≤ 1 SI), which allows a "
        f"magnetization in any direction. It fits the data to χ²/N {b['chi2']:.2f} with an RMS of {b['rms']:.0f} nT, against "
        f"{a['rms']:.0f} nT for the susceptibility model, and its largest residual is {b['max_abs']:.0f} nT against "
        f"{a['max_abs']:.0f} nT. At the {u['n']} underfitted stations the RMS residual falls from {u['induced_rms_nT']:.0f} "
        f"to {u['mvi_rms_nT']:.0f} nT (mean |residual| {u['induced_mean_abs_nT']:.0f} → {u['mvi_mean_abs_nT']:.0f} nT); "
        f"{u['mvi_over']} of them are still off by more than {u['threshold_nT']:.0f} nT ({u['mvi_over_pos']} above, "
        f"{u['mvi_over_neg']} below the model), so the vector model explains most, not all, of the anomaly.</p>",
        f"<p>The strongly magnetized cells (the {z.get('n_cells', 0):,} above a tenth of the largest amplitude) point at an "
        f"inclination of {z.get('resultant_inclination', 0):.0f}° and a declination of {z.get('resultant_declination', 0):.0f}° "
        f"(their resultant; coherence {z.get('resultant_coherence', 0):.2f}, where 1 means all parallel), far from the "
        f"present field (I {z.get('inducing_inclination', 0):.0f}°, D {z.get('inducing_declination', 0):.0f}°). The data "
        "ask for a magnetization that is not along the present field: remanent magnetization, common in banded iron "
        f"formation. The amplitude model is compact: {pct(b['core'])} of it lies in the core ({pct(a['core'])} of the "
        "susceptibility model).</p>",
        "<p>For the joint inversions this means that the susceptibility models of the belt are an approximation where the "
        "rock is remanent: a coupling can move the susceptibility towards the density model, but it cannot give it the "
        "direction the data ask for. The couplings of this report act on the induced susceptibility.</p>",
    ])


def summary(F, LOW, S, MAG, GL, g, m, c, has, pct, rng, GLK, CTRL, **_):
    base = c("none")
    out = [f"""    <li><b>Without a coupling the joint inversion returns the two single inversions</b> (cell-by-cell correlation {S['gravity_corr']:.2f} and {S['magnetics_corr']:.2f}). The two uncoupled models have little in common cell by cell: {pct(base['support']['magnetic_in_dense'])} of the magnetic cells are dense or light, and {pct(base['edges_shared'])} of the susceptibility model's edges lie on an edge of the density model.</li>"""]
    if has("cross_gradient") and has("joint_total_variation"):
        cg, jtv = "cross_gradient", "joint_total_variation"
        out.append(f"""    <li><b>The two structural couplings act on the susceptibility model and leave the density model alone</b> (correlation with the uncoupled density {g(cg)['corr_with_none']:.2f} and {g(jtv)['corr_with_none']:.2f}; susceptibility {m(cg)['corr_with_none']:.2f} and {m(jtv)['corr_with_none']:.2f}). The cross-gradient makes the gradients parallel (measure {base['cross_gradient']:.2f} → {c(cg)['cross_gradient']:.2f}), but mostly by keeping the edges of the two models apart ({pct(c(cg)['edges_shared'])} shared); the joint total variation puts more of them in the same cells ({pct(c(jtv)['edges_shared'])}), with both datasets fitted (χ²/N {g(jtv)['chi2']:.2f} and {m(jtv)['chi2']:.2f}).</li>""")
    if has("linear_correspondence"):
        k = "linear_correspondence"
        out.append(f"""    <li><b>The linear correspondence forces one relation on the belt and spoils the density model:</b> the dense body under the main high moves to {rng(g(k)['main'])} km (uncoupled {rng(g('none')['main'])} km) and {pct(g(k)['below'])} of the density model ends below the core, while both datasets are still fitted (χ²/N {g(k)['chi2']:.2f} and {m(k)['chi2']:.2f}).</li>""")
    if has(GLK):
        k = GLK
        ctrl = f", {pct(c(CTRL)['support']['magnetic_in_dense'])} in its control" if has(CTRL) else ""
        moved = (f"Both models move: the dense body under the main high rises to {rng(g(k)['main'])} km (uncoupled {rng(g('none')['main'])} km), and the susceptibility spreads through the dense body."
                 if g(k)["corr_with_none"] < 0.85 else
                 f"The density model follows the uncoupled one ({g(k)['corr_with_none']:.2f}); the susceptibility spreads through the dense body.")
        out.append(f"""    <li><b>The group lasso fits both datasets and keeps its models in the core</b> (χ²/N {g(k)['chi2']:.2f} and {m(k)['chi2']:.2f}; {pct(g(k)['core'])} and {pct(m(k)['core'])} in the core), <b>and it pairs the two supports:</b> {pct(c(k)['support']['magnetic_in_dense'])} of its magnetic cells are anomalous in density ({pct(base['support']['magnetic_in_dense'])} uncoupled{ctrl}), and it lines up the two structures ({pct(c(k)['edges_shared'])} of the susceptibility edges on a density edge, cross-gradient measure {c(k)['cross_gradient']:.2f}). {moved}</li>""")
    if MAG:
        a, b, u = MAG["induced"], MAG["mvi"], MAG["underfit"]
        z = b.get("magnetization") or {}
        out.append(f"""    <li><b>The magnetic data ask for remanent magnetization.</b> With a magnetization vector per cell the RMS residual falls from {a['rms']:.0f} to {b['rms']:.0f} nT, and at the {u['n']} stations south of the Sandur belt that every susceptibility model underfits by more than {u['threshold_nT']:.0f} nT from {u['induced_rms_nT']:.0f} to {u['mvi_rms_nT']:.0f} nT (RMS; {u['mvi_over']} of the {u['n']} are still off by more than {u['threshold_nT']:.0f} nT); the strongly magnetized cells point at I {z.get('resultant_inclination', 0):.0f}°, D {z.get('resultant_declination', 0):.0f}°, not along the present field (I {z.get('inducing_inclination', 0):.0f}°, D {z.get('inducing_declination', 0):.0f}°).</li>""")
    out.append("""    <li><b>For this belt the structural assumption is the right one.</b> The magnetic rock (iron formation) lies on the margins of the dense rock (the greenstone pile), so the two models share boundaries, not values: a coupling that ties the values together, by one relation or by one support, moves one model to where the other is. The data fit does not tell the couplings apart, so the choice has to come from the geology.</li>""")
    out.append("""    <li><b>No coupling fixes the depth.</b> The depth of each model still follows its own regularization (β); a coupling moves one model towards the other, which helps only where they really are the same rock.</li>""")
    return "\n".join(out)


def discussion(F, g, m, c, has, pct, rng, GLK, **_):
    base = c("none")
    out = ["""<p>The single-method reports showed what the two datasets see in the Sandur belt: a thick body of dense, weakly magnetic rock (the greenstone pile; amphibolite and metabasalt samples of 2.9–3.0 g/cc and 0.0002–0.004 SI), and thin sheets of strongly magnetic rock along its margins and its south-eastern closure (the banded iron formation), which are dense too (3.4 g/cc) but too thin to dominate the gravity. A coupling is an assumption about how the two properties are related; on this belt the assumptions fare as follows.</p>""",
           "<ul class=\"plain\">"]
    out.append("""  <li><b>Same boundaries (structural couplings): consistent with the geology.</b> The iron formation bounds the greenstone, so edges of the susceptibility model should lie on edges of the density model. The joint total variation asks for exactly that and delivers it in part; the cross-gradient asks only for parallel gradients and is satisfied, more cheaply, by a smooth susceptibility model whose edges avoid those of the density model. Both leave the density model unchanged: the gravity data and their regularization determine it, and the magnetic sheets are too thin to reshape it.</li>""")
    if has("linear_correspondence"):
        out.append("""  <li><b>One relation between the values (linear correspondence): contradicted.</b> The densest rock here is not the most magnetic. Forcing density = 0.5 × susceptibility moves the dense body into the magnetic sheets and leaves the rest of the gravity anomaly to the padding. A relation fitted to samples would fail in the same way, because two rock types with different relations make up the belt.</li>""")
    if has(GLK):
        k = GLK
        out.append(f"""  <li><b>The same cells are anomalous (group lasso): partly true.</b> The iron formation is both dense and magnetic, the greenstone is dense only. The group lasso lets a cell be anomalous in one property and not the other, at a price, so it does not force the greenstone to be magnetic; but it pulls the magnetic rock into the dense cells ({pct(c(k)['support']['magnetic_in_dense'])} of the magnetic cells are anomalous in density, against {pct(base['support']['magnetic_in_dense'])} uncoupled)""" + (f""", and it lifts the dense body towards the magnetic rock (half-maximum {rng(g(k)['main'])} km under the main high, against {rng(g('none')['main'])} km), as the linear correspondence does, only less""" if g(k)["corr_with_none"] < 0.85 else "") + f""". Where a magnetic body is also a dense body, as in the synthetic test of 30 September, this is the assumption to make; here most of the dense rock is not magnetic, and the pairing moves magnetic rock to where there is none.</li>""")
    out.append("""  <li><b>Remanence.</b> The strongest magnetic rock of the belt is not magnetized along the present field (Section 3). Every coupling here acts on the induced susceptibility, so in the south of the belt the coupled susceptibility models inherit an approximation that no coupling can remove.</li>""")
    out.append("</ul>")
    out.append("""<p>The common point: a coupling improves a model where its assumption holds and damages it where it does not, and the data fit says little about which is the case. The measures of Section 2.3 say what a coupling did, not whether it was right.</p>""")
    return "\n".join(out)


def low_text(LOW, F, pct, GLK, **_):
    out = ["<ul class=\"plain\">"]
    keys = [k for k in F if k in LOW]
    if keys:
        slowest = max(keys, key=lambda k: LOW[k].get("seconds") or 0)
        names = {"none": "no coupling", "cross_gradient": "cross-gradient", "joint_total_variation": "joint total variation",
                 "linear_correspondence": "linear correspondence", GLK: "group lasso", "group_lasso_depth_uncoupled": "its control"}
        out.append(f"""  <li><b>The coarse runs behave like the full-resolution ones</b>: the cross-gradient lowers the cross-gradient measure, the joint total variation raises the share of common edges, the linear correspondence ties the values, and the group lasso pairs the supports. Each coupling took at most {LOW[slowest]['seconds'] / 60:.1f} minutes on a laptop ({names.get(slowest, slowest)}): enough to choose a coupling before paying for the full runs.</li>""")
    if GLK in LOW and GLK in F:
        a, b = LOW[GLK], F[GLK]
        out.append(f"""  <li><b>The group lasso on the 2 km mesh</b> fits the data to χ²/N {a['gravity']['chi2']:.2f} and {a['magnetics']['chi2']:.2f} and keeps {pct(a['gravity']['core'])} and {pct(a['magnetics']['core'])} of its models in the core ({pct(b['gravity']['core'])} and {pct(b['magnetics']['core'])} at full resolution); {pct(a['coupling']['support']['magnetic_in_dense'])} of its magnetic cells are anomalous in density ({pct(b['coupling']['support']['magnetic_in_dense'])} at full resolution).</li>""")
    out.append("</ul>")
    return "\n".join(out)


def recommend(has, MAG, GLK, **_):
    out = ["""  <li><b>Choose the coupling from the geology, not from the fit.</b> Same rock in both models (one body, dense and magnetic): the group lasso. Different rock sharing boundaries, as in this belt: a structural coupling. Unknown: invert separately first and compare the two models.</li>""",
           """  <li><b>For this area: the joint total variation or the cross-gradient,</b> read together with the uncoupled models. Report what is common to them; do not read the coupled susceptibility model as better resolved.</li>""",
           """  <li><b>Use the group lasso where the magnetic bodies are also the dense ones</b> (the iron formation alone, or a mafic intrusion); over the whole belt it draws magnetic rock into the greenstone.</li>""",
           """  <li><b>Do not use the linear correspondence</b> unless one rock type carries both anomalies.</li>"""]
    if MAG:
        out.append("""  <li><b>Invert the magnetic data of the belt for a magnetization vector.</b> The susceptibility models are an approximation where the iron formation is remanent; a joint inversion that couples the amplitude of the magnetization vector, rather than the susceptibility, would carry the remanence into the joint model. Oriented samples of the iron formation would tell whether the direction found (Section 3) is its remanence.</li>""")
    out.append("""  <li><b>Depth still needs independent information.</b> The couplings do not supply it; the mapped dips of the iron formation or boreholes would.</li>""")
    return "\n".join(out)
