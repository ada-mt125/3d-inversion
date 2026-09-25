"""Generate a Plotly-based 3D model viewer HTML page.

Creates an interactive HTML page with:
  - 3D isosurface model comparison (true vs recovered)
  - Surface data maps (observed vs predicted)
  - Convergence charts
  - Controls: opacity, threshold, layer toggles

Inspired by Tomofast-x viewer template.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..datamodel.mesh import Mesh3D
from ..datamodel.model import PhysicalModel


def _mesh_to_grid_coords(mesh: Mesh3D):
    """Build flattened x/y/z coordinates for every cell center."""
    hx, hy, hz = mesh.hx, mesh.hy, mesh.hz
    nx, ny, nz = len(hx), len(hy), len(hz)

    cx = np.cumsum(hx) - hx / 2 + mesh.origin[0]
    cy = np.cumsum(hy) - hy / 2 + mesh.origin[1]
    cz = -(np.cumsum(hz) - hz / 2)  # depth positive downward → negative z

    xx, yy, zz = np.meshgrid(cx, cy, cz, indexing="ij")
    return {
        "x": cx.tolist(),
        "y": cy.tolist(),
        "z": cz.tolist(),
        "x_flat": xx.ravel(order="F").tolist(),
        "y_flat": yy.ravel(order="F").tolist(),
        "z_flat": zz.ravel(order="F").tolist(),
        "nx": nx, "ny": ny, "nz": nz,
    }


def _surface_data_json(
    locations: NDArray,
    values: NDArray,
    label: str,
) -> dict:
    """Pack surface observation data for Plotly scatter."""
    return {
        "x": locations[:, 0].tolist(),
        "y": locations[:, 1].tolist(),
        "z": values.tolist(),
        "label": label,
    }


def generate_model3d_viewer(
    mesh: Mesh3D,
    true_model: Optional[NDArray] = None,
    recovered_model: Optional[NDArray] = None,
    true_label: str = "True Model",
    recovered_label: str = "Recovered",
    property_name: str = "density",
    property_unit: str = "g/cm³",
    surface_data: Optional[list[dict]] = None,
    convergence: Optional[dict] = None,
    title: str = "GeoInv3D — 3D Model Viewer",
    output_path: str = "model3d_viewer.html",
) -> str:
    """Generate a self-contained HTML 3D model viewer.

    Args:
        mesh: The 3D mesh.
        true_model: True model values (n_cells,). Can be None.
        recovered_model: Recovered model values (n_cells,). Can be None.
        true_label: Display label for true model.
        recovered_label: Display label for recovered model.
        property_name: Physical property name.
        property_unit: Unit string.
        surface_data: List of dicts with keys:
            - locations (N,3), obs_values (N,), pred_values (N,),
            - method (str), unit (str)
        convergence: Dict with keys: iterations, phi_d, phi_m, phi_total
        title: Page title.
        output_path: Where to write the HTML.

    Returns:
        Path to the generated HTML file.
    """
    grid = _mesh_to_grid_coords(mesh)

    data = {"grid": grid, "property": property_name, "unit": property_unit}

    if true_model is not None:
        lim = float(np.max(np.abs(true_model)))
        data["true"] = {
            "values": true_model.tolist(),
            "lim": lim,
            "label": true_label,
        }

    if recovered_model is not None:
        lim = float(np.max(np.abs(recovered_model)))
        data["recovered"] = {
            "values": recovered_model.tolist(),
            "lim": lim,
            "label": recovered_label,
        }

    if surface_data:
        data["surfaces"] = surface_data

    if convergence:
        data["convergence"] = convergence

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            return super().default(obj)

    data_json = json.dumps(data, separators=(",", ":"), cls=_NumpyEncoder)

    html = _MODEL3D_TEMPLATE.replace("__DATA_JSON__", data_json)
    html = html.replace("__TITLE__", title)
    html = html.replace("__PROPERTY__", property_name)
    html = html.replace("__UNIT__", property_unit)

    out = Path(output_path)
    out.write_text(html, encoding="utf-8")
    return str(out)


def generate_comparison_viewer(
    mesh: Mesh3D,
    models: dict[str, NDArray],
    labels: dict[str, str],
    property_name: str = "density",
    property_unit: str = "g/cm³",
    surface_data: Optional[list[dict]] = None,
    convergence_data: Optional[dict[str, dict]] = None,
    title: str = "GeoInv3D — Inversion Results",
    output_path: str = "comparison_viewer.html",
) -> str:
    """Generate a multi-model comparison viewer.

    Args:
        mesh: The 3D mesh (shared by all models).
        models: Dict mapping model_id -> values array.
        labels: Dict mapping model_id -> display name.
        surface_data: List of surface data dicts per method.
        convergence_data: Dict mapping inv_id -> {iterations, phi_d, ...}
        title: Page title.
        output_path: Output HTML path.
    """
    grid = _mesh_to_grid_coords(mesh)

    data = {
        "grid": grid,
        "property": property_name,
        "unit": property_unit,
        "models": {},
    }

    for mid, vals in models.items():
        lim = float(np.max(np.abs(vals))) if np.any(vals != 0) else 1.0
        data["models"][mid] = {
            "values": vals.tolist(),
            "lim": lim,
            "label": labels.get(mid, mid),
        }

    if surface_data:
        data["surfaces"] = surface_data

    if convergence_data:
        data["convergence"] = convergence_data

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            return super().default(obj)

    data_json = json.dumps(data, separators=(",", ":"), cls=_NumpyEncoder)

    html = _COMPARISON_TEMPLATE.replace("__DATA_JSON__", data_json)
    html = html.replace("__TITLE__", title)

    out = Path(output_path)
    out.write_text(html, encoding="utf-8")
    return str(out)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML Template — 3D Model Viewer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_MODEL3D_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.3/plotly.min.js"></script>
<style>
:root{
  --bg:#0e131a;--panel:#161d27;--panel-2:#1c2530;--border:#2a3644;
  --border-soft:#212b37;--text:#e7ecf2;--text-muted:#8b97a6;--text-faint:#57677a;
  --accent:#d98e4a;--accent-soft:#4a3a28;--pos:#c0483d;--neg:#3d6fc0;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:"IBM Plex Sans",system-ui,sans-serif;
    height:100vh;overflow:hidden;display:flex;flex-direction:column;}
.mono{font-family:"IBM Plex Mono",monospace;}
header{display:flex;align-items:center;gap:16px;padding:10px 18px;border-bottom:1px solid var(--border);
      background:var(--panel);flex:none;}
.brand .name{font-family:"IBM Plex Mono";font-weight:600;font-size:14px;}
.brand .sub{font-size:11px;color:var(--text-muted);}
.tabs{display:flex;gap:0;margin-left:24px;}
.tab{font-family:"IBM Plex Mono";font-size:11.5px;padding:8px 16px;cursor:pointer;
    color:var(--text-muted);border-bottom:2px solid transparent;background:none;border-top:none;border-left:none;border-right:none;}
.tab:hover{color:var(--text);}
.tab.active{color:var(--accent);border-bottom-color:var(--accent);}
.main{flex:1;display:flex;min-height:0;overflow:hidden;}
.view{display:none;flex:1;min-height:0;}
.view.active{display:flex;}
#view-3d{flex-direction:row;}
#view-surface{flex-direction:column;}
#view-convergence{flex-direction:column;}
#plot3d{flex:1;min-width:0;}
aside{width:260px;flex:none;background:var(--panel);border-left:1px solid var(--border);
     padding:16px;display:flex;flex-direction:column;gap:14px;overflow-y:auto;}
.rail-group{display:flex;flex-direction:column;gap:6px;}
.rail-label{font-family:"IBM Plex Mono";font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
           color:var(--text-faint);display:flex;justify-content:space-between;align-items:baseline;}
.rail-label .val{color:var(--accent);font-size:11px;}
input[type="range"]{-webkit-appearance:none;width:100%;height:3px;border-radius:2px;background:var(--border);outline:none;margin:4px 0;}
input[type="range"]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;
  background:var(--accent);border:2px solid var(--panel);cursor:pointer;margin-top:-5px;}
input[type="checkbox"]{accent-color:var(--accent);}
.check-row{display:flex;align-items:center;gap:6px;font-family:"IBM Plex Mono";font-size:11px;color:var(--text-muted);}
.divider{height:1px;background:var(--border-soft);margin:2px 0;}
.btn{font-family:"IBM Plex Mono";font-size:11px;font-weight:500;color:var(--text);
    background:var(--panel-2);border:1px solid var(--border);border-radius:5px;padding:7px 10px;cursor:pointer;}
.btn:hover{border-color:var(--accent);color:var(--accent);}
.cbar{height:10px;border-radius:3px;}
.cbar-ticks{display:flex;justify-content:space-between;font-family:"IBM Plex Mono";font-size:9px;color:var(--text-faint);margin-top:2px;}
.surface-plots{display:flex;flex-wrap:wrap;flex:1;overflow-y:auto;padding:8px;}
.surface-plots .plot-cell{flex:1 1 48%;min-width:380px;min-height:350px;}
.conv-container{flex:1;display:flex;padding:8px;}
.conv-container .plot-cell{flex:1;min-height:300px;}
.hint{position:absolute;left:16px;bottom:12px;font-family:"IBM Plex Mono";font-size:10px;color:var(--text-faint);
     background:rgba(22,29,39,.85);border:1px solid var(--border-soft);padding:5px 9px;border-radius:5px;pointer-events:none;}
.stats{display:flex;gap:18px;margin-left:auto;font-family:"IBM Plex Mono";font-size:11px;color:var(--text-muted);}
.stats .stat b{color:var(--text);font-size:12px;font-weight:500;}
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="name">GeoInv3D</div>
    <div class="sub">3D Model Viewer</div>
  </div>
  <div class="tabs">
    <button class="tab active" data-view="view-3d">3D Model</button>
    <button class="tab" data-view="view-surface">Surface Data</button>
    <button class="tab" data-view="view-convergence">Convergence</button>
  </div>
  <div class="stats">
    <div class="stat" id="stat-info"></div>
  </div>
</header>
<div class="main">
  <!-- 3D Model View -->
  <div class="view active" id="view-3d" style="position:relative;">
    <div id="plot3d"></div>
    <div class="hint">drag to orbit · scroll to zoom</div>
    <aside>
      <div class="rail-group">
        <div class="rail-label">Iso threshold (% of max) <span class="val" id="thresh-val">25%</span></div>
        <input type="range" id="threshold" min="5" max="95" value="25" step="1">
      </div>
      <div class="rail-group">
        <div class="rail-label">Opacity <span class="val" id="op-val">0.60</span></div>
        <input type="range" id="opacity" min="10" max="95" value="60" step="1">
      </div>
      <div class="divider"></div>
      <div id="layer-toggles"></div>
      <div class="divider"></div>
      <div id="legends"></div>
      <div class="divider"></div>
      <button class="btn" id="reset-cam">&#8635; RESET VIEW</button>
      <div class="divider"></div>
      <div id="model-stats" style="font-size:11px;color:var(--text-faint);"></div>
    </aside>
  </div>
  <!-- Surface Data View -->
  <div class="view" id="view-surface">
    <div class="surface-plots" id="surface-plots"></div>
  </div>
  <!-- Convergence View -->
  <div class="view" id="view-convergence">
    <div class="conv-container" id="conv-container"></div>
  </div>
</div>
<script>
const D = __DATA_JSON__;

// Tab switching
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    t.classList.add('active');
    document.getElementById(t.dataset.view).classList.add('active');
  });
});

// ── 3D Model ────────────────────────────────────────────
const PALETTES = {
  true: {pos:[[0,"#1a5276"],[0.5,"#2980b9"],[1,"#85c1e9"]], neg:[[0,"#85c1e9"],[0.5,"#2980b9"],[1,"#1a5276"]]},
  recovered: {pos:[[0,"#7a3630"],[0.5,"#c0483d"],[1,"#e8836f"]], neg:[[0,"#5c96e8"],[0.5,"#3d6fc0"],[1,"#274c8a"]]},
};
const PAL_KEYS = ['true','recovered'];
let threshPct = 0.25, opVal = 0.60;

function buildIsosurface(key, vals, lim) {
  const g = D.grid;
  const thr = lim * threshPct;
  const common = {
    type:"isosurface", x:g.x_flat, y:g.y_flat, z:g.z_flat, value:vals,
    opacity:opVal, caps:{x:{show:false},y:{show:false},z:{show:false}},
    showscale:false, lighting:{ambient:0.55,diffuse:0.7,specular:0.15,roughness:0.9},
  };
  const palKey = PAL_KEYS.includes(key) ? key : 'recovered';
  const pal = PALETTES[palKey];
  const traces = [];
  if (lim > 0) {
    traces.push(Object.assign({}, common, {
      isomin:thr, isomax:lim, surface:{count:8,fill:0.9}, colorscale:pal.pos,
    }));
  }
  const negLim = Math.min(...vals);
  if (negLim < -thr) {
    traces.push(Object.assign({}, common, {
      isomin:negLim, isomax:-thr, surface:{count:8,fill:0.9}, colorscale:pal.neg,
    }));
  }
  return traces;
}

function axStyle(t){
  return {title:{text:t,font:{family:"IBM Plex Mono",size:11,color:"#8b97a6"}},color:"#57677a",
    gridcolor:"#212b37",zerolinecolor:"#2a3644",showbackground:true,backgroundcolor:"#11161e",
    tickfont:{family:"IBM Plex Mono",size:9,color:"#57677a"}};
}

const activeModels = {};
function initToggles() {
  const container = document.getElementById('layer-toggles');
  const legends = document.getElementById('legends');
  const statsDiv = document.getElementById('model-stats');
  let html = '', legHtml = '', statHtml = '';

  if (D.true) {
    activeModels['true'] = true;
    html += '<div class="check-row"><input type="checkbox" id="chk-true" checked><label for="chk-true">' + D.true.label + '</label></div>';
    legHtml += '<div class="rail-group"><div class="rail-label">' + D.true.label + ' ' + D.unit + '</div>' +
      '<div class="cbar" style="background:linear-gradient(90deg,#1a5276,#2980b9,#85c1e9);"></div>' +
      '<div class="cbar-ticks"><span>0</span><span>' + D.true.lim.toFixed(4) + '</span></div></div>';
    statHtml += '<div>True range: [' + Math.min(...D.true.values).toFixed(4) + ', ' + D.true.lim.toFixed(4) + ']</div>';
  }
  if (D.recovered) {
    activeModels['recovered'] = true;
    html += '<div class="check-row"><input type="checkbox" id="chk-recovered" checked><label for="chk-recovered">' + D.recovered.label + '</label></div>';
    legHtml += '<div class="rail-group"><div class="rail-label">' + D.recovered.label + ' ' + D.unit + '</div>' +
      '<div class="cbar" style="background:linear-gradient(90deg,#274c8a,#3d6fc0,#e8836f,#c0483d,#7a3630);"></div>' +
      '<div class="cbar-ticks"><span>-' + D.recovered.lim.toFixed(4) + '</span><span>0</span><span>' + D.recovered.lim.toFixed(4) + '</span></div></div>';
    statHtml += '<div>Recovered range: [' + Math.min(...D.recovered.values).toFixed(4) + ', ' + D.recovered.lim.toFixed(4) + ']</div>';
  }
  container.innerHTML = html;
  legends.innerHTML = legHtml;
  statsDiv.innerHTML = statHtml;

  ['true','recovered'].forEach(k => {
    const el = document.getElementById('chk-' + k);
    if (el) el.addEventListener('change', e => { activeModels[k] = e.target.checked; render3D(false); });
  });
}

function render3D(first) {
  const traces = [];
  if (activeModels['true'] && D.true) traces.push(...buildIsosurface('true', D.true.values, D.true.lim));
  if (activeModels['recovered'] && D.recovered) traces.push(...buildIsosurface('recovered', D.recovered.values, D.recovered.lim));

  const g = D.grid;
  const xSpan = g.x[g.x.length-1]-g.x[0], ySpan = g.y[g.y.length-1]-g.y[0], zSpan = Math.abs(g.z[g.z.length-1]-g.z[0]);
  const m = Math.max(xSpan, ySpan);

  const layout = {
    paper_bgcolor:"#0e131a", plot_bgcolor:"#0e131a", margin:{l:0,r:0,t:0,b:0},
    scene:{
      xaxis:axStyle("Easting (m)"), yaxis:axStyle("Northing (m)"), zaxis:axStyle("Depth (m)"),
      aspectmode:"manual",
      aspectratio:{x:xSpan/m, y:ySpan/m, z:Math.max(zSpan/m, 0.35)},
      camera:{eye:{x:1.5,y:-1.5,z:0.9}},
    },
    showlegend:false,
  };

  if (first) Plotly.newPlot("plot3d", traces, layout, {displayModeBar:false, responsive:true});
  else Plotly.react("plot3d", traces, layout, {displayModeBar:false, responsive:true});
}

// Controls
document.getElementById('threshold').addEventListener('input', e => {
  threshPct = (+e.target.value)/100;
  document.getElementById('thresh-val').textContent = e.target.value + '%';
  render3D(false);
});
document.getElementById('opacity').addEventListener('input', e => {
  opVal = (+e.target.value)/100;
  document.getElementById('op-val').textContent = opVal.toFixed(2);
  render3D(false);
});
document.getElementById('reset-cam').addEventListener('click', () => {
  Plotly.relayout("plot3d", {"scene.camera":{eye:{x:1.5,y:-1.5,z:0.9}}});
});

// ── Surface Data ────────────────────────────────────────
function renderSurface() {
  const container = document.getElementById('surface-plots');
  if (!D.surfaces || !D.surfaces.length) {
    container.innerHTML = '<div style="padding:40px;color:var(--text-faint);text-align:center;">No surface data available</div>';
    return;
  }

  container.innerHTML = '';
  D.surfaces.forEach((sd, idx) => {
    // Observed data
    const obsDiv = document.createElement('div');
    obsDiv.className = 'plot-cell';
    obsDiv.id = 'surf-obs-' + idx;
    container.appendChild(obsDiv);

    // Predicted data
    const predDiv = document.createElement('div');
    predDiv.className = 'plot-cell';
    predDiv.id = 'surf-pred-' + idx;
    container.appendChild(predDiv);

    // Residual
    const resDiv = document.createElement('div');
    resDiv.className = 'plot-cell';
    resDiv.id = 'surf-res-' + idx;
    container.appendChild(resDiv);

    // Scatter comparison
    const scatDiv = document.createElement('div');
    scatDiv.className = 'plot-cell';
    scatDiv.id = 'surf-scat-' + idx;
    container.appendChild(scatDiv);

    const unit = sd.unit || '';
    const method = sd.method || 'Data';
    const cs = 'RdBu_r';
    const surfLayout = (title) => ({
      paper_bgcolor:"#0e131a", plot_bgcolor:"#161d27", margin:{l:60,r:20,t:40,b:50},
      title:{text:title, font:{family:"IBM Plex Sans",size:13,color:"#e7ecf2"}},
      xaxis:{title:"Easting (m)", color:"#57677a", gridcolor:"#212b37",
            tickfont:{family:"IBM Plex Mono",size:9,color:"#57677a"}},
      yaxis:{title:"Northing (m)", color:"#57677a", gridcolor:"#212b37",
            tickfont:{family:"IBM Plex Mono",size:9,color:"#57677a"}, scaleanchor:"x"},
      font:{family:"IBM Plex Sans", color:"#8b97a6"},
    });

    const obsTrace = {
      type:'scatter', mode:'markers', x:sd.obs_x, y:sd.obs_y,
      marker:{size:10, color:sd.obs_values, colorscale:cs, showscale:true,
             colorbar:{title:{text:unit,font:{size:10}},tickfont:{size:9}}},
      text:sd.obs_values.map(v => v.toFixed(4)),
      hovertemplate:'%{x:.0f}, %{y:.0f}<br>%{marker.color:.4f} ' + unit + '<extra></extra>',
    };
    Plotly.newPlot('surf-obs-'+idx, [obsTrace], surfLayout(method+' — Observed'), {displayModeBar:false,responsive:true});

    const predTrace = {
      type:'scatter', mode:'markers', x:sd.obs_x, y:sd.obs_y,
      marker:{size:10, color:sd.pred_values, colorscale:cs, showscale:true,
             colorbar:{title:{text:unit,font:{size:10}},tickfont:{size:9}},
             cmin:Math.min(...sd.obs_values), cmax:Math.max(...sd.obs_values)},
      text:sd.pred_values.map(v => v.toFixed(4)),
      hovertemplate:'%{x:.0f}, %{y:.0f}<br>%{marker.color:.4f} ' + unit + '<extra></extra>',
    };
    Plotly.newPlot('surf-pred-'+idx, [predTrace], surfLayout(method+' — Predicted'), {displayModeBar:false,responsive:true});

    const residuals = sd.obs_values.map((v,i) => v - sd.pred_values[i]);
    const resMax = Math.max(...residuals.map(Math.abs));
    const resTrace = {
      type:'scatter', mode:'markers', x:sd.obs_x, y:sd.obs_y,
      marker:{size:10, color:residuals, colorscale:'RdBu', showscale:true, cmid:0,
             colorbar:{title:{text:'Δ'+unit,font:{size:10}},tickfont:{size:9}}},
      hovertemplate:'%{x:.0f}, %{y:.0f}<br>residual: %{marker.color:.4f}<extra></extra>',
    };
    Plotly.newPlot('surf-res-'+idx, [resTrace], surfLayout(method+' — Residual (obs-pred)'), {displayModeBar:false,responsive:true});

    // 1:1 scatter
    const vmin = Math.min(...sd.obs_values, ...sd.pred_values);
    const vmax = Math.max(...sd.obs_values, ...sd.pred_values);
    const scatTrace = {
      type:'scatter', mode:'markers', x:sd.obs_values, y:sd.pred_values,
      marker:{size:8, color:'#d98e4a', opacity:0.8},
      hovertemplate:'obs: %{x:.4f}<br>pred: %{y:.4f}<extra></extra>',
    };
    const lineTrace = {
      type:'scatter', mode:'lines', x:[vmin,vmax], y:[vmin,vmax],
      line:{color:'#57677a', dash:'dash', width:1}, hoverinfo:'skip',
    };
    const scatLayout = Object.assign(surfLayout(method+' — Obs vs Pred'), {
      xaxis:{title:'Observed ('+unit+')', color:"#57677a", gridcolor:"#212b37",
            tickfont:{family:"IBM Plex Mono",size:9,color:"#57677a"}},
      yaxis:{title:'Predicted ('+unit+')', color:"#57677a", gridcolor:"#212b37",
            tickfont:{family:"IBM Plex Mono",size:9,color:"#57677a"}},
    });
    delete scatLayout.yaxis.scaleanchor;
    Plotly.newPlot('surf-scat-'+idx, [scatTrace, lineTrace], scatLayout, {displayModeBar:false,responsive:true});
  });
}

// ── Convergence ─────────────────────────────────────────
function renderConvergence() {
  const container = document.getElementById('conv-container');
  if (!D.convergence) {
    container.innerHTML = '<div style="padding:40px;color:var(--text-faint);text-align:center;">No convergence data</div>';
    return;
  }

  container.innerHTML = '';
  const convLayout = (title) => ({
    paper_bgcolor:"#0e131a", plot_bgcolor:"#161d27", margin:{l:70,r:30,t:40,b:50},
    title:{text:title, font:{family:"IBM Plex Sans",size:14,color:"#e7ecf2"}},
    xaxis:{title:"Iteration", color:"#57677a", gridcolor:"#212b37", dtick:1,
          tickfont:{family:"IBM Plex Mono",size:10,color:"#57677a"}},
    yaxis:{title:"Value", type:"log", color:"#57677a", gridcolor:"#212b37",
          tickfont:{family:"IBM Plex Mono",size:10,color:"#57677a"}},
    font:{family:"IBM Plex Sans", color:"#8b97a6"},
    showlegend:true, legend:{font:{size:10,color:"#8b97a6"},bgcolor:"rgba(0,0,0,0)"},
  });

  const phidDiv = document.createElement('div');
  phidDiv.className = 'plot-cell';
  phidDiv.id = 'conv-phid';
  container.appendChild(phidDiv);

  const phimDiv = document.createElement('div');
  phimDiv.className = 'plot-cell';
  phimDiv.id = 'conv-phim';
  container.appendChild(phimDiv);

  const colors = ['#d98e4a','#3d6fc0','#c0483d','#2f9c8f','#e8c04f','#9b59b6'];
  const conv = D.convergence;
  const phidTraces = [], phimTraces = [];
  let ci = 0;

  Object.keys(conv).forEach(k => {
    const c = conv[k];
    const iters = c.iterations || c.phi_d.map((_,i) => i);
    phidTraces.push({
      type:'scatter', mode:'lines+markers', x:iters, y:c.phi_d,
      name:k, line:{color:colors[ci%colors.length],width:2},
      marker:{size:5},
    });
    if (c.phi_m) {
      phimTraces.push({
        type:'scatter', mode:'lines+markers', x:iters, y:c.phi_m,
        name:k, line:{color:colors[ci%colors.length],width:2},
        marker:{size:5},
      });
    }
    ci++;
  });

  // Target line
  if (D.surfaces && D.surfaces.length > 0) {
    const nData = D.surfaces.reduce((s,sd) => s + sd.obs_values.length, 0);
    phidTraces.push({
      type:'scatter', mode:'lines', x:[0, Math.max(...phidTraces.flatMap(t=>t.x))],
      y:[nData, nData], name:'Target (N_data)', line:{color:'#57677a',dash:'dash',width:1},
    });
  }

  Plotly.newPlot('conv-phid', phidTraces, convLayout('Data Misfit φ_d'), {displayModeBar:false,responsive:true});
  if (phimTraces.length > 0) {
    Plotly.newPlot('conv-phim', phimTraces, convLayout('Model Norm φ_m'), {displayModeBar:false,responsive:true});
  }
}

// Initialize
initToggles();
render3D(true);
setTimeout(() => { renderSurface(); renderConvergence(); }, 100);
window.addEventListener("resize", () => {
  Plotly.Plots.resize("plot3d");
  document.querySelectorAll('.plot-cell').forEach(el => Plotly.Plots.resize(el));
});
</script>
</body>
</html>
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML Template — Multi-model Comparison (placeholder, same structure)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_COMPARISON_TEMPLATE = _MODEL3D_TEMPLATE
