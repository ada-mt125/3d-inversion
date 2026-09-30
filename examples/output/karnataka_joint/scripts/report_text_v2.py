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
        out.append(f"""<li><b>Linear correspondence.</b> The two models become one: {pct(c(k)['support']['magnetic_in_dense'])} of the magnetic cells are anomalous in density too, and |density| correlates with susceptibility at {c(k)['corr_abs_cells']:.2f}. The relation does not describe this belt's rocks (Section 4.3), and the density model pays for it: the dense body under the main high has its half-maximum at {rng(g(k)['main'])} km, against {rng(g('none')['main'])} km uncoupled, and what the relation cannot place is put below the core ({pct(g(k)['below'])} of the density model, against {pct(g('none')['below'])}). Both datasets are still fitted, to χ²/N {g(k)['chi2']:.2f} and {m(k)['chi2']:.2f}: the fit does not warn.</li>""")
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
    lo = min(min(g(k)["chi2"], m(k)["chi2"]) for k in F)
    hi = max(max(g(k)["chi2"], m(k)["chi2"]) for k in F)
    out = [f"""    <li><b>The data do not rank the couplings.</b> Every run fits both datasets (χ²/N {lo:.2f}–{hi:.2f}), and there is no known model of the area to compare with. The conclusions are of three kinds: what each coupling does (certain: a property of the method), what the magnetic data show (the remanence), and which coupling suits this belt (a judgement from the geology).</li>"""]
    out.append(f"""    <li><b>Without a coupling the joint inversion returns the two single inversions</b> (cell-by-cell correlation {S['gravity_corr']:.2f} and {S['magnetics_corr']:.2f}); the two uncoupled models have little in common cell by cell ({pct(base['support']['magnetic_in_dense'])} of the magnetic cells are dense or light, {pct(base['edges_shared'])} of the susceptibility edges lie on a density edge).</li>""")
    parts = []
    if has("cross_gradient"):
        k = "cross_gradient"
        parts.append(f"the cross-gradient makes the gradients parallel ({base['cross_gradient']:.2f} → {c(k)['cross_gradient']:.2f}) mostly by keeping the edges apart ({pct(c(k)['edges_shared'])} shared)")
    if has("joint_total_variation"):
        k = "joint_total_variation"
        parts.append(f"the joint total variation lines the edges up ({pct(c(k)['edges_shared'])}) and leaves the density model as it was ({g(k)['corr_with_none']:.2f})")
    if has("linear_correspondence"):
        k = "linear_correspondence"
        parts.append(f"the linear correspondence lifts the dense body to {rng(g(k)['main'])} km (uncoupled {rng(g('none')['main'])} km) and puts {pct(g(k)['below'])} of the density model below the core")
    if has(GLK):
        k = GLK
        parts.append(f"the group lasso pairs the magnetic cells with dense ones ({pct(c(k)['support']['magnetic_in_dense'])}, {pct(base['support']['magnetic_in_dense'])} uncoupled) and lifts the dense body part of the way, to {rng(g(k)['main'])} km")
    out.append("""    <li><b>What each coupling does</b> (Section 4.1): """ + "; ".join(parts) + """. The couplings that tie the values move the dense body up because the magnetic data pin the magnetic rock near the surface far more firmly than the gravity data pin the dense rock at depth (Section 4.2).</li>""")
    if MAG:
        a, b, u = MAG["induced"], MAG["mvi"], MAG["underfit"]
        z = b.get("magnetization") or {}
        out.append(f"""    <li><b>The magnetic data ask for remanent magnetization.</b> With a magnetization vector per cell the RMS residual falls from {a['rms']:.0f} to {b['rms']:.0f} nT, and at the {u['n']} stations south of the Sandur belt that every susceptibility model underfits by more than {u['threshold_nT']:.0f} nT from {u['induced_rms_nT']:.0f} to {u['mvi_rms_nT']:.0f} nT ({u['mvi_over']} of them still off by more than {u['threshold_nT']:.0f} nT); the strongly magnetized cells point at I {z.get('resultant_inclination', 0):.0f}°, D {z.get('resultant_declination', 0):.0f}°, not along the present field (I {z.get('inducing_inclination', 0):.0f}°, D {z.get('inducing_declination', 0):.0f}°).</li>""")
    out.append("""    <li><b>For this belt a structural coupling is the one consistent with the geology</b> (Section 4.3), a judgement from the rock samples and the mapping, not from the fit: the magnetic iron formation lies on the margins of the dense, weakly magnetic greenstone, so the two models share boundaries, not values. The linear correspondence's premise is contradicted by the samples; the group lasso's holds for the iron formation only. Section 4.4 gives the first choice for other settings.</li>""")
    out.append("""    <li><b>No coupling fixes the depth.</b> Where the couplings disagree, as on the depth of the dense body, the data do not decide; the depth follows the assumption and the regularization.</li>""")
    return "\n".join(out)


def discussion(F, g, m, c, has, pct, rng, GLK, **_):
    base = c("none")
    out = ["""<p>This is a judgement, not a result of the inversions: with no known model and every coupling fitting the data equally, the choice rests on what is known of the rocks. The single-method reports and the rock samples describe the Sandur belt as a thick body of dense, weakly magnetic rock (the greenstone pile; amphibolite and metabasalt samples of 2.9–3.0 g/cc and 0.0002–0.004 SI) with thin sheets of strongly magnetic rock along its margins and its south-eastern closure (the banded iron formation), which are dense too (3.4 g/cc) but too thin to dominate the gravity. Against this, the assumptions of the couplings fare as follows.</p>""",
           "<ul class=\"plain\">"]
    if has("linear_correspondence"):
        k = "linear_correspondence"
        out.append(f"""  <li><b>One relation between the values (linear correspondence): contradicted by the samples.</b> Its premise, that the densest rock is the most magnetic, does not hold here: the greenstone is dense and hardly magnetic, the iron formation dense and strongly magnetic, two rocks with different relations. It can be set aside with some confidence, and its result shows the cost of the wrong premise: the dense body moved to {rng(g(k)['main'])} km and {pct(g(k)['below'])} of the density model below the core, with the data fitted as well as by any other coupling.</li>""")
    if has(GLK):
        k = GLK
        out.append(f"""  <li><b>The same cells are anomalous (group lasso): true for the iron formation, not for the greenstone.</b> The iron formation is both dense and magnetic, the greenstone dense only. The group lasso does not force the greenstone to be magnetic, but it draws the magnetic rock into the dense cells ({pct(c(k)['support']['magnetic_in_dense'])} of the magnetic cells anomalous in density, against {pct(base['support']['magnetic_in_dense'])} uncoupled) and the dense body towards the magnetic rock ({rng(g(k)['main'])} km). It fits where the magnetic rock is the dense rock; over the whole belt its premise holds only in part.</li>""")
    cg, jtv = "cross_gradient", "joint_total_variation"
    txt = """  <li><b>Same boundaries, free values (structural couplings): consistent with the geology.</b> The iron formation bounds the greenstone, so edges of the susceptibility model should lie on edges of the density model, while their values stay unrelated."""
    if has(jtv) and has(cg):
        txt += f""" The joint total variation asks for exactly that and delivers it in part ({pct(c(jtv)['edges_shared'])} of the susceptibility edges on a density edge, {pct(base['edges_shared'])} uncoupled); the cross-gradient asks only for parallel gradients and is satisfied, more cheaply, by keeping the edges apart ({pct(c(cg)['edges_shared'])}). Both leave the density model as the gravity data and its regularization make it (correlation {g(jtv)['corr_with_none']:.2f} and {g(cg)['corr_with_none']:.2f} with the uncoupled one). Of the two, the joint total variation is the one whose result matches its premise here."""
    out.append(txt + "</li>")
    out.append("""  <li><b>Remanence.</b> The strongest magnetic rock of the belt is not magnetized along the present field (Section 3). Every coupling here acts on the induced susceptibility, so in the south of the belt the coupled susceptibility models inherit an approximation that no coupling removes.</li>""")
    out.append("</ul>")
    out.append("""<p>The judgement could be tested: on a synthetic model built from this geometry (thin magnetic sheets on the margins of a thick dense body), where the truth is known, the couplings can be compared directly; density and susceptibility logs, or measured samples at the scale of a cell, would turn the premises into constraints; oriented samples of the iron formation would tell whether the direction of Section 3 is its remanence.</p>""")
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
    out = ["""  <li><b>Certain: what each coupling does</b> (Section 4.1). The cross-gradient keeps the edges of the two models apart, the joint total variation lines them up, the linear correspondence and the group lasso move the dense body towards the magnetic rock. This is the behaviour of the methods on these data; it does not depend on what the rock really is.</li>""",
           """  <li><b>Supported by the data: remanence</b> in the south of the Sandur belt (Section 3). The magnetic data of the belt should be inverted for a magnetization vector; a joint inversion that couples its amplitude, rather than the susceptibility, would carry the remanence into the joint model.</li>""",
           """  <li><b>A judgement: for this belt, the joint total variation</b> (or the cross-gradient), read together with the uncoupled models (Section 4.3). Report what the couplings agree on; do not read the coupled susceptibility model as better resolved.</li>""",
           """  <li><b>Elsewhere, choose the coupling from what is known of the rocks</b> (Section 4.4): the group lasso where the magnetic bodies are the dense ones, a structural coupling where different rocks share boundaries, the linear correspondence only for one rock type with a known relation, and no coupling first when the relation is unknown.</li>""",
           """  <li><b>To test the judgement,</b> run the couplings on a synthetic model of this belt's geometry, where the truth is known, and constrain the premises with logs or measured samples; oriented samples of the iron formation for the direction of its magnetization.</li>""",
           """  <li><b>Depth needs independent information.</b> The couplings do not supply it; the mapped dips of the iron formation or boreholes would.</li>"""]
    return "\n".join(out)


def why(F, DS, g, m, c, has, pct, rng, GLK, CTRL, **_):
    """Why the linear correspondence (and, less, the group lasso) moves the dense body up."""
    if not DS:
        return ""
    top = DS[0]
    mid = next((d for d in DS if d["from_km"] == 4), DS[min(2, len(DS) - 1)])
    deep = DS[-1]
    out = [f"<p>The two datasets do not see depth alike. Per unit volume, the magnetic data lose sight of a cell much "
           f"faster than the gravity data: at 4–8 km below the ground a cell counts {mid['gravity']:.2f} of what it counts "
           f"at the surface for gravity but {mid['magnetics']:.2f} for the magnetic data, and below the core "
           f"{deep['gravity']:.2f} against {deep['magnetics']:.3f} (the field of a magnetic dipole falls off one power of the "
           "distance faster than that of a mass). The magnetic data therefore fix where the magnetic rock is, near the "
           "surface, much more firmly than the gravity data fix the depth of the dense rock.</p>"]
    if has("linear_correspondence"):
        k = "linear_correspondence"
        out.append(f"<p><b>Linear correspondence.</b> With density = 0.5 × susceptibility in every cell, a cell can be dense "
                   "only if it is magnetic too: the relation asks for 2 SI of susceptibility per g/cc of density, 0.6 SI for a "
                   f"contrast of 0.3 g/cc. The dense body at {rng(g('none')['main'])} km of the uncoupled run would need that "
                   "much susceptibility at depth, whose magnetic anomaly the data do not have; so density is allowed only where "
                   "the magnetic rock is — the "
                   f"shallow sheets, which carry the dense body up to {rng(g(k)['main'])} km under the main high — and where "
                   f"the magnetic data can no longer object: below the core, which receives {pct(g(k)['below'])} of the "
                   f"density model ({pct(g('none')['below'])} uncoupled). The susceptibility model gets shallower too "
                   f"(centroid in the belt {m(k)['box_sandur']['centroid_km']:.1f} km, against "
                   f"{m('none')['box_sandur']['centroid_km']:.1f} km uncoupled).</p>")
    if has(GLK):
        k = GLK
        ctrl = (f" Its control, the same solver without the pairing, keeps the dense body at {rng(g(CTRL)['main'])} km: the "
                "lift comes from the pairing." if has(CTRL) else "")
        out.append(f"<p><b>Group lasso.</b> The group norm of a cell, ‖(ρ, χ)‖, is less than |ρ| + |χ|: density costs less in a "
                   "cell that already holds susceptibility than in an empty one. So the dense rock is drawn towards the shallow "
                   "magnetic cells, but, with no fixed ratio, it may still stay in non-magnetic cells at the full price: the "
                   f"dense body rises only part of the way, to {rng(g(k)['main'])} km, and little goes below the core "
                   f"({pct(g(k)['below'])}).{ctrl}</p>")
    out.append("<p>The couplings that leave the values free (the cross-gradient and the joint total variation) place no such "
               "condition on a cell, and the dense body stays where the gravity data and its own regularization put it. None "
               "of the depths is confirmed by the data — every run fits them equally — so the lift shows what the "
               "assumption does, not where the rock is.</p>")
    return "\n".join(out)
