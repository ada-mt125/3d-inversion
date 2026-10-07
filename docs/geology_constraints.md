# Geology constraints: a reference model from rock units

Code: `geoinv3d/methods/geology.py` (building the constraints), `geoinv3d/io/vector.py`
(shapefiles and GeoJSON without GIS libraries), the pipeline's `params["geology"]`
(`cloud/worker.py`, `_apply_geology`), tests: `tests/test_geology.py`.
Upload page: the *Model* step (4 of 7: after Mesh, before Inversion): the 3D model builder
(`geoinv3d/viz/model_builder.html`) in the page, and/or a dropped spec with its files. The
builder also has a page of its own (`/model`), whose models are kept on the server.

## Building the model (like MARE2DEM's regions, in 3D)

In the *Model* step the builder takes the page's data window as its area, the ground and the
coordinate system, and shows the data as points on the ground (**Data**), so the bodies can be
drawn over their anomalies; every change goes back to the page, and each job takes the part
of the model its data are about.

- **Bodies**: **+ Box**, **+ Cylinder**, or **Polygon body** (click the ground in plan view,
  double-click or Enter to close). A body has an outline, a top and bottom — metres **below
  the ground**, or **elevations** — and a dip (degrees from horizontal, 90 = vertical) and dip
  direction (the outline moves that way as it deepens). In the view: drag the gizmo's arrows
  to move it (the vertical one moves it up and down), **S** to resize a box or cylinder,
  **V** to edit a polygon's vertices (drag a corner; drag an edge's midpoint to add one;
  right-click to delete one).
- **Layer stacks** (**≡ Layers**): everywhere, or in a box or polygon; a top (below the ground
  or an elevation); the interfaces flat or **tilted** (tilt from horizontal, the azimuth they
  sink towards, about a pivot where the top holds); thicknesses are vertical, and an empty
  last thickness reaches the bottom of the mesh.
- **Boreholes** (**⊙ Boreholes**, **⤒ Boreholes CSV**, or a CSV pasted under the holes table):
  each hole (collar, azimuth, dip) is logged in depth intervals along it, each with a
  lithology and its values; a CSV has one row per interval (`hole,x,y,from,to,…`) or per
  measurement (`hole,x,y,depth,…`, each value standing for the depths half-way to its
  neighbours). The cells a hole passes through take its intervals by their length in them;
  a **reach** also puts the log partly (1 − distance / reach) into the cells around.
- **Values per property.** Every body, layer and interval carries a value (and optionally a
  lowest and highest value) for each property: **density** (g/cc, a rock density; sent as
  the contrast to the reduction density), **susceptibility** (SI) and **resistivity** (Ω·m;
  inverted as log conductivity). An empty value leaves that part free for that property,
  so one model serves a gravity and a DC/MT job at once. Default ranges: ± 0.05 g/cc,
  ± 0.005 SI, a factor 2 in resistivity.
- **Weight** (how strongly the inversion is pulled to the value), **Fixed** (a tenth of the
  range and weight ≥ 100: it hardly moves) and **Sharp** (little smoothing across the
  part's boundaries: the model may jump there, as across ModEM's covariance "tears").
- Later parts win where they overlap (↑ ↓ reorder the list). **Everywhere else**: the
  reduction density; bounds for the free cells; for resistivity the free cells' value
  (empty = the method's background) and how a cell holding several layers averages them
  (below).
- The view: colour by unit or by a property (with its colour bar), plan / south / east views,
  a west–east or south–north **section**, **Zoom to selected** (or a double-click), undo / redo.
- **Save model** (in the step) downloads the spec of the first property with the others under
  `other_properties` and the model under `builder`; dropping it back in the files area, or
  importing it in the builder, restores everything. Bodies and layer stacks with given values
  from any spec come into the builder too. The builder's own **Save / Open** keep models on
  the server; its page's **Use in an inversion** opens the upload page with the model in this
  step.

The builder's parts are added to a dropped file spec of the same property (samples,
boreholes, maps): their units join its units and their sources come after its sources.
Files are **optional**: a spec whose samples or borehole files are missing is still
submitted, and the worker leaves out what needs them, with a note in the result.

## What it does

For gravity (a linear problem) with a convex regularization, the starting model does not
change the answer. What does is

- the **reference model** the smallness term pulls towards,
- the **bounds** of each cell, and
- **how strongly** each cell is pulled (the smallness term's weights).

So the constraints set all three per active cell, and the inversion also starts from the
reference:

| | where a unit is placed | elsewhere |
|---|---|---|
| reference | the unit's value | `unconstrained.value`, else the method's background (0; log σ_background for MT/DC) |
| bounds | the unit's range | `unconstrained.lower/upper`, else the job's bounds |
| smallness weight | the unit's `weight` (default 5) | `unconstrained.weight` (1) |

**Cells take the volume share of what they hold.** Bodies, map polygons and layers give
each cell the fraction of its volume they cover (depths exactly; the footprint sampled on a
4 × 4 grid where an outline crosses it), and a later source takes its share of a cell from
what was there. A cell's reference, bounds and weight are then the volume averages of its
contents, so a layer or body thinner than the cells is mixed into them instead of lost —
for gravity the mass is kept exactly (contrast × thickness down a column). Resistivity
averages in log conductivity (the geometric mean) by default; `"mixing": "conductance"`
averages conductivity (a thin conductor keeps its conductance, as MT's horizontal currents
see it) and `"resistance"` averages resistivity (a thin resistor keeps its transverse
resistance, as vertical DC currents see it). Samples and boreholes mark whole cells, the
depth a sample stands for being a guess. The summary gives, per unit, the cells it
dominates (`n_cells`), the cells it touched (`n_touched`) and its volume in cells
(`volume_cells`), and notes the layers and bodies thinner than their cells.

**Sharp boundaries.** A unit with `"sharp": true` labels the cells it dominates; the
first-order smoothness terms weigh the faces between differently labelled cells by
`sharp_factor` (0.01), so the model may jump across the unit's boundaries (ModEM's
covariance does this with a mask and "tear" rules between region codes). SimPEG 0.25 takes
these face weights directly (`set_weights` on `SmoothnessFirstOrder`).

For density the values are **contrasts relative to the Bouguer reduction density**
(`background`, 2.67 by default), the quantity the inversion solves for. The constraints
are a hypothesis: the data still decide, so compare the misfit with and without them
(the viewer shows the reference model as the "Geology reference" layer in 3D; runs with and
without constraints sit in separate branches of the workflow tree).

The L1–L2 coordinate-descent solver uses the reference and the bounds (it inverts for
the deviation from the reference) but not the weights; SimPEG's solvers (smooth L2,
sparse, L1–L2 IRLS, MGS, TV) use all three. Single MT and DC inversions take
resistivity constraints; there the start at the reference is the starting model proper
(the problem is non-linear). Joint inversions and the group lasso do not take geology
constraints yet.

## The spec (JSON)

```json
{
  "name": "samples + iron holes",
  "property": "density",
  "background": 2.67,
  "samples": {"file": "Physical_properties_rock_samples.csv", "window_pad_m": 20000},
  "units": {
    "BIF": {"rock_types": ["banded iron formation"], "weight": 10},
    "mafic": {"rock_types": ["amphibolite", "dolerite*", "gabbro", "metabasalt"], "weight": 3},
    "granitoid": {"rock_types": ["*granite*", "granodiorite*", "*gneiss*"], "weight": 2},
    "cover": {"value": -0.1, "lower": -0.3, "upper": 0.0}
  },
  "sources": [
    {"type": "samples", "depth_m": 500},
    {"type": "boreholes", "file": "holes.shp",
     "unit_by": {"field": "commodity_", "values": {"IRON*": "BIF"}}, "extend_m": 400},
    {"type": "body", "unit": "BIF", "polygon": [[x1, y1], [x2, y2], [x3, y3]],
     "top_m": 0, "bottom_m": 1500, "dip": 70, "dip_direction": 90},
    {"type": "map", "file": "geology.shp", "bottom_m": 1000,
     "unit_by": {"field": "LITHOLOGY", "values": {"*iron*": "BIF", "*basalt*": "mafic"}}}
  ],
  "unconstrained": {"lower": -0.3, "upper": 0.8}
}
```

**Units.** Either `rock_types` (case-insensitive patterns with `*`, matched against the
samples' rock types): value = mean sample density − background, range = sample min…max
widened by `margin` (0.05 g/cc; 0.005 SI) — or `value` with `lower` / `upper`. `weight`
multiplies the smallness term in the unit's cells (1 = no stronger than elsewhere).
`property: "susceptibility"` reads a susceptibility column instead (a header like
`X 10^(-6) CGS units` is converted to SI) with background 0.

**Samples file.** CSV with position (`Latitude`/`Longitude` or `x`/`y`), rock type and
density (or susceptibility) columns, matched by name (as the NGPM
`Physical_properties_rock_samples.csv`). `window_pad_m` keeps the samples within that
distance of the mesh.

**Sources**, applied in order (a later source overrides an earlier one in shared cells):

- `samples`: the column of cells under each sample whose rock type belongs to a unit,
  every cell overlapping 0…`depth_m` below the ground (`radius_m` widens the column).
- `boreholes`: cells along each hole from its collar (on the mesh's ground by default,
  `"collar": "file"` uses `collar_field`), along `azimuth_field` (bearing, degrees from
  north) and `inclination_field` (degrees below horizontal; 0 or missing = vertical) for
  `length_field` metres plus `extend_m`. Field names default to the GSI drilling table
  (`bearing`, `cl_inclina`, `length_m`, `rl_collar_`). The unit is `unit`, or from
  `unit_by` {`field`, `values`: {pattern: unit}}. A shapefile, GeoJSON or CSV (`x_field`,
  `y_field`). Alone, a hole constrains only the cells its trace passes through, and the
  regularization carries that little further than a cell or two (about 100 m with 50 m
  cells in `examples/output/synthetic_ablation`). `radius_m` reaches further: a cell whose
  centre lies a distance d from the nearest point of a trace takes the share 1 − d/radius
  of that point's unit, so its reference, bounds and weight go linearly from the log's at
  the hole to the unconstrained ones (or an earlier source's) at the radius.
- `body`: a `polygon` (vertices in the job's CRS, or longitude/latitude — detected, or
  `"crs": "EPSG:4326"`) or `box` [west, east, south, north], from `top_m` to `bottom_m`
  below the ground (`"elevations": true` for absolute z), optionally dipping (`dip` from
  horizontal, `dip_direction` azimuth): the outline moves down-dip by
  (depth − top) / tan(dip).
- `map`: the polygons of a shapefile or GeoJSON, each extruded like a body; the unit from
  `unit` or `unit_by`.

- `layers`: a stack, everywhere or inside a `polygon` / `box`; `top_m` below the ground,
  or an elevation with `"reference": "elevation"`; `layers` [{`unit`, `thickness_m`}]
  top to bottom (vertical thicknesses; the last may omit `thickness_m` to reach the
  bottom; `"unit": null` leaves a layer free, its thickness still counting); interfaces
  tilted by `dip` (degrees) sinking towards `dip_direction` about `origin` [x, y] (default:
  the outline's or the mesh's centre), where the top holds.

Coordinates in longitude/latitude are projected to the job's CRS (see `io/crs.py`);
`"crs": "job"` on a source says its coordinates are metres in the job's CRS (the upload
page sends it; otherwise small numbers could pass for degrees).

**Resistivity.** `"property": "resistivity"`: unit values, ranges, `unconstrained` values
and samples in Ω·m (a samples column `Resistivity…` in Ω·m or `Conductivity…` in S/m or
mS/m; units from samples take the geometric mean and the min…max widened by the factor
`margin`, default 2). The model is log conductivity = −ln ρ, so a unit's lowest
resistivity is its highest model value. A layered prior for MT or DC:

```json
{"property": "resistivity", "mixing": "conductance",
 "units": {"cover": {"value": 30}, "clay": {"value": 3, "sharp": true},
           "basement": {"value": 3000, "lower": 1000, "upper": 30000}},
 "sources": [{"type": "layers", "crs": "job", "top_m": 0, "dip": 5, "dip_direction": 90,
              "layers": [{"unit": "cover", "thickness_m": 40},
                         {"unit": "clay", "thickness_m": 15},
                         {"unit": "basement"}]}]}
```

## Ideas taken from ModEM and mtpy-v2

- **mtpy-v2** (`modeling/structured_mesh_3d.py`): `assign_resistivity_from_surface_data`
  puts a value between two gridded surfaces by cell centre — the `layers` source here does
  planes, by volume share, so thin layers are not lost. Its meshes grade the vertical
  cells (`z1_layer` growing logarithmically to `z_target_depth`,
  `mesh_tools.make_log_increasing_array`): thin shallow cells are what resolve thin
  shallow layers; the core cells here are uniform (a possible next step).
- **ModEM** (`3D_MT/modelParam/modelCov`): the model covariance takes an integer mask per
  cell (0 air, 9 ocean — both fixed — 1–8 free regions) and exception rules that turn off
  smoothing across boundaries between chosen codes: here `fixed` and `sharp`.

## Karnataka, 2 km (examples/output/karnataka_gravity/geology/)

`run_geology_lowres.py` inverts the NGPM Bouguer grid (2 km thinning, 2 km × 1 km cells,
terrain from the station elevations, sparse p = [0, 2, 2, 2], α_s = 1, depth weighting
β = 1, bounds −0.3…+0.8) without and with `karnataka_geology.json` (rock samples and the
iron-ore boreholes; no geological map is available):

| | unconstrained | geology |
|---|---|---|
| χ²/N | 1.134 | 1.131 |
| RMS | 0.532 mGal | 0.532 mGal |
| mean density contrast in the 68 constrained cells | −0.003 | +0.103 (reference +0.113) |

Units from the samples within 20 km: BIF +0.60 (3 samples), mafic +0.32, granitoid −0.02.
The constraints placed BIF on 5 cells, mafic on 18 and granitoid on 45 (of 30,381).
**The data fit equally well with density at the surface under the outcrops**: that the
earlier models held almost none in the top kilometre was a choice of the regularization,
not a requirement of the data. With so few constrained cells this is a local statement;
a geological map (`map` source) would constrain the outcrop belts as a whole.
