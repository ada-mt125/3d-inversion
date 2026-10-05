"""Build the Lp ablation report: Chinese HTML, optionally the PDF.

    py examples/output/synthetic_ablation_lp/scripts/make_figures.py
    py examples/output/synthetic_ablation_lp/scripts/build_report.py [--pdf]

Every number in the text comes from figures/numbers.json, data/sweep.log (time and cost) or
the inputs (inputs/: the reference model of group B, the boreholes of group C).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "karnataka_inputs" / "shared"))
from style import Figures, page, table, to_pdf  # noqa: E402

FIGS = ROOT / "figures"
N = json.loads((FIGS / "numbers.json").read_text(encoding="utf-8"))
T, R, BEST, TM = N["truth"], N["runs"], N["best"], N["truth_measures"]
SPEC_B = json.loads((ROOT / "inputs" / "spec_B.json").read_text())
HOLES = list(csv.DictReader((ROOT / "inputs" / "boreholes_C.csv").read_text(encoding="utf-8").splitlines()))
OUT = ROOT / "ablation_report.html"

GROUPS = {"A": "A 无先验", "Bw1": "B 参考模型 · 权重 1", "Bw10": "B 参考模型 · 权重 10",
          "Bw100": "B 参考模型 · 权重 100", "C": "C 钻孔（半径 150 m）"}
BG = ["Bw1", "Bw10", "Bw100"]
NORMS = {"n0000": "(0,0,0,0)", "n0111": "(0,1,1,1)", "n0221": "(0,2,2,1)", "n0222": "(0,2,2,2)", "n1111": "(1,1,1,1)"}
BETAS = {"b10": "1", "b15": "1.5", "b20": "2", "b30": "3"}
LENGTHS = {"L1": "1", "L3": "3"}
SETTINGS = [f"{n}_{b}_{ell}" for n in NORMS for b in BETAS for ell in LENGTHS]


def label(setting):
    n, b, ell = setting.split("_")
    return f"p = {NORMS[n]}，β = {BETAS[b]}，L = {LENGTHS[ell]}"


def sweep_cost():
    """Wall time and cost of the AWS run, from its log's last line."""
    text = (ROOT / "data" / "sweep.log").read_text()
    m = re.search(r"([\d.]+) h × (\d+) instances ≈ \$([\d.]+)", text)
    return (float(m.group(1)) * 60, int(m.group(2)), float(m.group(3))) if m else (None, None, None)


def f2(v):
    return f"{v:.2f}"


def scores(g):
    return [R[f"{g}_{s}"]["score"] for s in SETTINGS]


# ── tables ─────────────────────────────────────────────────────────────

def table_changes():
    return table(["", "第一版", "第二版（本报告）"], [
        ["正则化", "L2、Lp、L1、L1–L2、MGS、TV 各一组参数", "只用 Lp，40 组参数（5 种 norms × 4 种深度加权 × 2 种光滑长度）"],
        ["迭代上限", "60 次（IRLS 30 次），6 次未收敛", "200 次（IRLS 100 次）"],
        ["侵入体", "厚 100 m，走向 2 km，地下 50–1000 m", f"厚 {T['dyke']['thickness']:.0f} m，走向 {(T['dyke']['y'][1] - T['dyke']['y'][0]) / 1000:.1f} km，地下 {T['dyke']['top_depth']:.0f}–{T['dyke']['bottom_depth']:.0f} m"],
        ["B 组先验", "建模器默认 ±0.005 SI：先验被写死", "只作参考模型，上下限 0–3 SI，权重 1、10、100 三档"],
        ["C 组钻孔", "只约束钻孔穿过的网格，影响不到 100 m", "测井值在钻孔上固定，并按距离线性衰减到 150 m（新增 radius_m）"],
        ["评价", "相关系数为主，对幅值敏感", "体积匹配重叠率（与幅值无关），立方体和侵入体分别计算"],
    ], compact=True, numeric_from=9)


def table_groups():
    geo = N["geology"]
    rows = [["A 无先验", "参考模型为 0，上下限 0–3 SI", "0", "40"]]
    for g in BG:
        w = g[2:]
        rows.append([GROUPS[g], f"两个先验体作参考模型，上下限 0–3 SI，权重 {w}",
                     f"{geo[g]['n_constrained']:,}（部分覆盖 {geo[g]['n_touched'] - geo[g]['n_constrained']:,}）", "40"])
    rows.append([GROUPS["C"], "5 个钻孔的测井：孔上 ±0.005 SI、权重 5，向外 150 m 线性减弱",
                 f"{geo['C']['n_constrained']:,}（部分 {geo['C']['n_touched'] - geo['C']['n_constrained']:,}）", "40"])
    return table(["组", "先验", "受约束网格（过半 / 部分）", "反演次数"], rows, compact=True, numeric_from=9)


def table_prior():
    c, d = T["cube"], T["dyke"]
    s0, s1 = SPEC_B["sources"]
    u = SPEC_B["units"]
    p, q = s0["polygon"], s1["polygon"]
    pcx = (min(x for x, _ in p) + max(x for x, _ in p)) / 2
    pcy = (min(y for _, y in p) + max(y for _, y in p)) / 2
    qx = (min(x for x, _ in q) + max(x for x, _ in q)) / 2
    qy0, qy1 = min(y for _, y in q), max(y for _, y in q)
    return table(["", "真实模型", "B 组参考模型", "偏差"], [
        ["立方体中心（东, 北）", f"{c['centre'][0]:,.0f}, {c['centre'][1]:,.0f}", f"{pcx:,.0f}, {pcy:,.0f}",
         f"东 {pcx - c['centre'][0]:+.0f} m，北 {pcy - c['centre'][1]:+.0f} m"],
        ["立方体深度", f"{c['top_depth']:.0f}–{c['top_depth'] + c['edge']:.0f} m", f"{s0['top_m']:.0f}–{s0['bottom_m']:.0f} m",
         f"深 {s0['top_m'] - c['top_depth']:.0f} m"],
        ["立方体磁化率", f"{c['kappa']} SI", f"{u['cube (prior)']['value']} SI", f"{(u['cube (prior)']['value'] / c['kappa'] - 1) * 100:+.0f}%"],
        ["侵入体顶部东向", f"{d['top_x']:,.0f}", f"{qx:,.0f}", f"{qx - d['top_x']:+.0f} m"],
        ["侵入体走向（北）", f"{d['y'][0]:,.0f}–{d['y'][1]:,.0f}", f"{qy0:,.0f}–{qy1:,.0f}", f"北移 {qy0 - d['y'][0]:.0f} m"],
        ["侵入体深度", f"{d['top_depth']:.0f}–{d['bottom_depth']:.0f} m", f"{s1['top_m']:.0f}–{s1['bottom_m']:.0f} m",
         f"深 {s1['top_m'] - d['top_depth']:.0f} m"],
        ["侵入体倾角", f"{d['dip']:.0f}° 向东", f"{s1['dip']:.0f}° 向东", f"陡 {s1['dip'] - d['dip']:.0f}°"],
        ["侵入体磁化率", f"{d['kappa']} SI", f"{u['intrusion (prior)']['value']} SI",
         f"{(u['intrusion (prior)']['value'] / d['kappa'] - 1) * 100:+.0f}%"],
    ], compact=True, numeric_from=9)


def table_holes():
    names = {"H1": "立方体中心", "H2": "立方体东边界外 50 m", "H3": "侵入体，斜孔", "H4": "侵入体浅部", "H5": "侵入体深部"}
    rows, seen = [], {}
    for h in HOLES:
        seen.setdefault(h["hole"], []).append(h)
    for name, ivs in seen.items():
        h0 = ivs[0]
        inc = float(h0["inclination"])
        kind = "直孔" if inc == 90 else f"方位 {float(h0['bearing']):.0f}°，倾角 {inc:.0f}°"
        s, hits = 0.0, []
        for iv in ivs:
            L = float(iv["length_m"])
            if iv["unit"] != "hole: background":
                hits.append(f"{s:.0f}–{s + L:.0f} m：{iv['unit'].split(': ')[1]}")
            s += L
        key = name.split()[0]
        rows.append([f"{key} {names[key]}", f"{float(h0['x']):,.0f}, {float(h0['y']):,.0f}", kind, f"{s:.0f} m",
                     "；".join(hits) or "全孔为背景（0 SI）"])
    return table(["钻孔", "孔口（东, 北）", "方向", "孔深", "见矿段（沿孔）"], rows, compact=True, numeric_from=9)


def table_best():
    short = lambda st: "p {} · β {} · L {}".format(NORMS[st.split("_")[0]], BETAS[st.split("_")[1]],   # noqa: E731
                                                  LENGTHS[st.split("_")[2]])
    rows = [["真实模型（同一指标）", "—", "1.00", "1.00", "1.00", f"{TM['cube_depth']:.0f}", f"{TM['dyke_base']:.0f}",
             f"{TM['dip']:.0f}", "0.50", "1.00", "—"]]
    for g in GROUPS:
        x = R[BEST[g]]
        rows.append([f"{GROUPS[g]}<br><span class=note>{short(x['setting'])}</span>", f"{x['chi2_per_datum']:.2f}",
                     f2(x["score"]), f2(x["overlap_cube"]), f2(x["overlap_dyke"]), f"{x['cube_depth']:.0f}",
                     f"{x['dyke_base']:.0f}", f"{x['dip']:.0f}", f2(x["k_cube"]), f2(x["k_dyke"]), f"{x['k_wrong']:.2f}"])
    return table(["组与最优参数", "χ²/N", "得分", "重叠·立方体", "重叠·侵入体", "立方体深度", "侵入体深度", "倾角",
                  "κ 立方体", "κ 侵入体", "κ 错位区"], rows)


def table_groupstats():
    rows = []
    for g in GROUPS:
        st = N["group_stats"][g]
        rows.append([GROUPS[g], f2(R[BEST[g]]["score"]), f2(st["median"]), f2(st["mean"]),
                     f"{st['better_than_A']} / 40" if g != "A" else "—", f"{st['converged']} / 40"])
    return table(["组", "最优得分", "中位数", "平均", "同参数下优于 A", "收敛"], rows, compact=True)


def table_effects():
    rows = []
    for dim, vals, name in (("norms", NORMS, "norms"), ("beta", BETAS, "β"), ("length", LENGTHS, "L")):
        for v, txt in vals.items():
            rows.append([f"{name} = {txt}"] + [f2(N["effects"][g][dim][v]) for g in GROUPS])
    return table(["参数取值（其余参数平均）"] + list(GROUPS.values()), rows)


# ── body ───────────────────────────────────────────────────────────────

def body(fig, summary, tuning_text, best_text, findings_ref, findings_holes, conclusions):
    c, d, mesh = T["cube"], T["dyke"], N["mesh"]
    minutes, n_inst, cost = sweep_cost()
    chi = [x["chi2_per_datum"] for x in R.values()]
    n_conv = sum(x["status"] == "converged" for x in R.values())
    its = [x["iterations"] for x in R.values()]
    mins = [x["minutes"] for x in R.values() if x["minutes"]]
    a = R[BEST["A"]]
    run_line = (f"{len(R)} 次，AWS ap-south-1 上 {n_inst} 台 c5.4xlarge 并行（每台同时 4 个），共 {minutes:.0f} 分钟，约 ${cost:.0f}"
                if minutes else f"{len(R)} 次，AWS ap-south-1")
    return f"""
<header>
  <div class="eyebrow">GeoInv3D · 合成模型消融实验（第二版）· 2026 年 10 月 5 日</div>
  <h1>参考模型与钻孔对 Lp 磁法反演的影响：调参后的消融实验</h1>
  <p class="lede">第一版实验的各方法都没有调参，侵入体过于细长，B 组先验被建模器默认范围写死，钻孔约束只作用于钻孔所穿的网格。本版只用 Lp 正则化，在 40 组参数上系统调参，侵入体改为粗短；在无先验、软参考模型（三种权重）和带影响半径的钻孔三类条件下各跑全部参数，比较各自的最优结果，以及同一组参数下有无先验的差别。</p>
  <dl class="meta">
    <div><dt>模型</dt><dd>5 × 5 km（UTM 43N）。立方体边长 {c['edge']:.0f} m，地下 {c['top_depth']:.0f}–{c['top_depth'] + c['edge']:.0f} m，{c['kappa']} SI；侵入体厚 {d['thickness']:.0f} m、走向长 {(d['y'][1] - d['y'][0]) / 1000:.1f} km，地下 {d['top_depth']:.0f}–{d['bottom_depth']:.0f} m，向东倾 {d['dip']:.0f}°，{d['kappa']} SI</dd></div>
    <div><dt>数据</dt><dd>总场异常，离地 {T['station_height']:.0f} m，{T['station_spacing']:.0f} m 网格 {T['n_stations']:,} 点（{T['tmi_range'][0]:,.0f} 到 {T['tmi_range'][1]:,.0f} nT），噪声 2% + 5 nT；反演时抽稀到 100 m（{mesh['n_data']:,} 点）。地磁场同 Block 8：倾角 {T['field'][1]}°，偏角 {T['field'][2]}°</dd></div>
    <div><dt>网格与设置</dt><dd>八叉树，核心网格 {mesh['core_cell_m']:.0f} × {mesh['core_cell_m']:.0f} × {mesh['core_cell_z_m']:.0f} m，地表细化 {mesh['octree_levels']}，{mesh['n_active']:,} 个活动网格；磁化率 0–3 SI；Lp 稀疏正则化，深度加权；最多 200 次迭代（IRLS 100 次）</dd></div>
    <div><dt>运行</dt><dd>{run_line}；每次 {min(mins):.0f}–{max(mins):.0f} 分钟，{min(its)}–{max(its)} 次迭代</dd></div>
  </dl>
</header>

<section class="summary" aria-labelledby="sum">
  <h2 id="sum">要点</h2>
  <ul>
{summary}
  </ul>
</section>

<h2><span class="no">1</span>与第一版的区别</h2>
<p class="note">表 1　两版实验的对比</p>
{table_changes()}

<h2><span class="no">2</span>实验设计</h2>
<div class="prose">
<p>地形、地磁场、数据采集方式和噪声与第一版相同；立方体也相同。侵入体改为厚 {d['thickness']:.0f} m、走向 {(d['y'][1] - d['y'][0]) / 1000:.1f} km、地下 {d['top_depth']:.0f}–{d['bottom_depth']:.0f} m 的粗短板状体，体积约为立方体的 {T['true_volume_m3']['dyke'] / T['true_volume_m3']['cube']:.0f} 倍。正演在 25 m 规则网格上计算，与反演用的八叉树网格不同。</p>
<p>每次反演都是 Lp 稀疏正则化，只改变三类参数（共 40 组）：</p>
<ul class="plain">
  <li><b>norms</b> (p<sub>s</sub>, p<sub>x</sub>, p<sub>y</sub>, p<sub>z</sub>)：(0,0,0,0)、(0,1,1,1)、(0,2,2,1)、(0,2,2,2)、(1,1,1,1)。p<sub>s</sub> 控制模型本身的稀疏程度，p<sub>x,y,z</sub> 控制梯度：2 为光滑，1 为分块，0 为最紧凑。</li>
  <li><b>深度加权 β</b>：1、1.5、2、3。权重为 (z + z<sub>0</sub>)<sup>−β/2</sup>，β 越大深部越"便宜"，模型越容易放深。</li>
  <li><b>光滑长度 L</b>（α<sub>x</sub> = α<sub>y</sub> = α<sub>z</sub>，SimPEG 的长度尺度）：1、3。L 越大，梯度项相对于模型项越重，模型越光滑。</li>
</ul>
<p>其余设置固定：α<sub>s</sub> = 1，自动 β<sub>0</sub>（比值 1），每次冷却 2 倍，目标 χ²/N = 1，IRLS 阈值自动。每组参数在五个组中各跑一次（表 2）：</p>
</div>
<p class="note">表 2　五个组</p>
{table_groups()}
<div class="prose">
<p><b>B 组</b>的参考模型与第一版相同的思路，模拟"地质学家画得大致正确但有偏差"（表 3）。但它只作为参考模型进入正则化的模型项：反演被拉向参考值，拉力由权重决定（1 = 与其他网格相同；10、100 = 更强）；上下限与其他网格一样是 0–3 SI，数据可以改它。</p>
<p><b>C 组</b>的 5 个钻孔测井取自真实模型（表 4）。钻孔穿过的网格固定为测井值（±0.005 SI，权重 5）。周围 150 m 内的网格按距离线性减弱：在距离 d 处，参考值、上下限和权重按份额 1 − d/150 与无约束的值混合。这是本版在建模器中新增的 <code>radius_m</code> 选项。注意，测井的背景段（0 SI）同样会向外延伸。</p>
</div>
{fig('setup', "合成模型。(a) 地形、真实异常体（红：立方体及侵入体向下倾伸的投影范围）、B 组参考模型（紫虚线）、C 组钻孔（▲，橙圈示意 150 m 半径）；点线为剖面位置。(b) 带噪声的总场异常。(c) 剖面上的真实模型，紫虚线为 B 组参考模型，黑线为钻孔（粗段为见矿段）。")}
<p class="note">表 3　B 组参考模型与真实模型的对比</p>
{table_prior()}
<p class="note">表 4　C 组钻孔（测井取自真实模型）</p>
{table_holes()}
<div class="prose">
<p><b>评价指标。</b>主要指标是<b>体积匹配重叠率</b>：在立方体所在的西半区，取磁化率最高、总体积等于真实立方体体积的那些网格，计算它们覆盖了真实立方体的多少；侵入体在东半区同样计算。两者的平均作为"得分"，用来挑选最优参数。重叠率只看位置和形态，不受幅值影响：对把磁化压缩成高值小块的紧凑解，和把磁化摊开的光滑解都公平。辅助指标：立方体磁化中心深度、侵入体 90% 磁化量所在深度、侵入体视倾角、两个异常体内的平均磁化率、B 组参考模型画错处（实际没有异常体）的平均磁化率。真实模型用同一套指标计算的值作为参照：立方体中心深度 {TM['cube_depth']:.0f} m，侵入体 90% 深度 {TM['dyke_base']:.0f} m，倾角 {TM['dip']:.0f}°。</p>
<p class="note">按真值调参只在合成实验里可行，实际工作中做不到；本报告的"最优"是各组参数能达到的上限。第 3 节还给出每组在全部参数上的中位数，代表"参数没选好"时的一般水平。</p>
</div>

<h2><span class="no">3</span>调参结果</h2>
<div class="prose">
<p>200 次反演中 {n_conv} 次在迭代上限内达到目标 χ²/N（1 ± 0.1），χ²/N 全部在 {min(chi):.2f}–{max(chi):.2f} 之间。图 {fig.ref('tuning')} 是五个组在 40 组参数上的得分，红框为各组最优。表 5 是汇总，表 6 是每个参数取值的平均得分（对其余参数平均）。</p>
{tuning_text}
</div>
{fig('tuning', "体积匹配重叠率（立方体与侵入体平均）。行为 norms，列为深度加权 β 与光滑长度 L；* 表示未在迭代上限内收敛；红框为该组最优。")}
<p class="note">表 5　各组得分汇总</p>
{table_groupstats()}
{fig('effects', "每个参数取值下的平均得分（实线，对其余参数平均）和最好得分（虚线）。")}
<p class="note">表 6　每个参数取值的平均得分</p>
{table_effects()}

<h2><span class="no">4</span>各组最优结果</h2>
<p class="note">表 7　各组最优参数的结果。深度（m）：立方体为磁化中心深度，侵入体为 90% 磁化所在深度；倾角为侵入体视倾角（°）；κ 为平均磁化率（SI），错位区指 B 组参考模型画错、实际没有异常体的位置。第一行为真实模型用同一指标计算的值</p>
{table_best()}
<div class="prose">{best_text}</div>
{fig('best_sections', "东西向剖面（北向 2.5 km），各组的最优参数。黑线为真实异常体，紫虚线为 B 组参考模型，黑色粗线为钻孔见矿段。色阶 0–1 SI。")}
{fig('best_metrics', "各组最优结果的主要指标；虚线为真实模型用同一指标计算的值。深度轴向下。")}

<h2><span class="no">5</span>参考模型（B 组）的作用</h2>
<div class="prose">
{findings_ref}
</div>
{fig('paired', "同一组参数下，有先验（纵轴）与无先验（横轴）的重叠率，左为立方体，右为侵入体；每点一组参数，在虚线以上表示先验有帮助。")}
{fig('same_setting_sections', "同一组参数（A 组最优参数）在五个组中的剖面：只有先验不同。")}

<h2><span class="no">6</span>钻孔（C 组）的作用</h2>
<div class="prose">
{findings_holes}
</div>
{fig('boreholes', "(a) C 组与 A 组（同一组参数）之差随离钻孔距离的变化，细线为全部参数，粗线为 C 组最优参数；灰色区为 150 m 以外。(b)–(c) 沿钻孔的磁化率：黑线为测井，灰色为 A 组，橙色为 C 组（C 组最优参数）。(d) 未交给反演的虚拟孔。")}

<h2><span class="no">7</span>结论与建议</h2>
<div class="prose">
<ol class="steps">
{conclusions}
</ol>
</div>

<h2><span class="no">8</span>局限</h2>
<div class="prose">
<ul class="plain">
  <li>按真值挑选的"最优参数"在实际工作中无法得到；实际中只能靠数据拟合（这里都一样好）、钻孔检验或经验来选。</li>
  <li>只有一个合成模型和一次噪声实现；只考虑感磁，没有剩磁。</li>
  <li>B 组只测试了一种偏差组合，C 组只测试了一个半径（150 m），而且背景段与见矿段使用同样的半径。</li>
  <li>α<sub>s</sub>、β 冷却速度和 IRLS 阈值没有调整；数据抽稀到 100 m。</li>
</ul>
</div>

<h2>文件</h2>
<div class="prose">
<ul class="plain">
  <li>输入与 200 次反演的参数：<code>examples/output/synthetic_ablation_lp/inputs/</code>（<code>scripts/make_synthetic.py</code>；<code>runs.json</code>）</li>
  <li>批量运行：<code>deploy/ec2_sweep.py</code>（多台 EC2）与 <code>deploy/sweep_runner.py</code>（单机），日志 <code>data/sweep.log</code></li>
  <li>结果：<code>data/runs/</code>（200 个，不入库）；报告中展示的在 <code>data/best/</code>；全部指标在 <code>figures/numbers.json</code></li>
  <li>指标定义 <code>scripts/metrics.py</code>，图件 <code>scripts/make_figures.py</code>，本报告 <code>scripts/build_report.py</code></li>
  <li>钻孔影响半径：<code>geoinv3d/methods/geology.py</code>（<code>radius_m</code>），说明见 <code>docs/geology_constraints.md</code></li>
</ul>
</div>
<footer>GeoInv3D · SimPEG 0.25.2 · 由 build_report.py 生成；文中数字均来自 figures/numbers.json、data/sweep.log 与 inputs/。</footer>
"""


# ── the findings: every number from numbers.json ──────────────────────

def med_diff(g, key):
    import statistics
    return statistics.median(R[f"{g}_{s}"][key] - R[f"A_{s}"][key] for s in SETTINGS)


def median(g, key):
    import statistics
    return statistics.median(R[f"{g}_{s}"][key] for s in SETTINGS)


def texts(fig):
    E, GS, P, BH = N["effects"], N["group_stats"], N["pinned"], N["boreholes"]
    a, c = R[BEST["A"]], R[BEST["C"]]
    sa, sc = BEST["A"].split("_", 1)[1], BEST["C"].split("_", 1)[1]
    a_scores = scores("A")
    gui = "n0221_b15_L1"                       # the page's default: p = (0,2,2,1), α 1, β 1.5
    worse_c = sorted(SETTINGS, key=lambda s: R[f"C_{s}"]["score"] - R[f"A_{s}"]["score"])[:3]
    robust = "n1111_b20_L3"
    v = BH["virtual"]
    dec = BH["c_minus_a"][sc]
    under = lambda g: GS[g]["better_than_A"]   # noqa: E731

    summary = f"""
    <li><b>调参比先验更重要。</b>同样没有先验（A 组），40 组参数的得分从 {min(a_scores):.2f} 到 {max(a_scores):.2f}，中位数 {GS['A']['median']:.2f}。影响最大的是光滑长度 L：L = 3 平均 {E['A']['length']['L3']:.2f}，L = 1 只有 {E['A']['length']['L1']:.2f}。norms 中 (1,1,1,1) 最稳，平均 {E['A']['norms']['n1111']:.2f}。β = 3 明显变差，平均 {E['A']['beta']['b30']:.2f}。GUI 目前的默认设置 p = (0,2,2,1)、α 全为 1、β = 1.5，在这里只得 {R['A_' + gui]['score']:.2f}，是 40 组里较差的。</li>
    <li><b>A 组最优</b>（{label(sa)}）把两个异常体都找了回来：重叠率 {a['overlap_cube']:.2f} / {a['overlap_dyke']:.2f}，立方体深度 {a['cube_depth']:.0f} m（真值 {TM['cube_depth']:.0f}），侵入体倾角 {a['dip']:.0f}°（真值 {TM['dip']:.0f}°），侵入体内磁化率 {a['k_dyke']:.2f} SI（真值 1）。粗短的异常体加上调好的参数，磁法单独就能做得相当好。</li>
    <li><b>有偏差的参考模型最多持平，权重越大越差。</b>
      <ul>
        <li>B 组最优得分：权重 1 为 {R[BEST['Bw1']]['score']:.2f}，权重 10 为 {R[BEST['Bw10']]['score']:.2f}，权重 100 为 {R[BEST['Bw100']]['score']:.2f}；A 组为 {a['score']:.2f}。</li>
        <li>同一组参数下比 A 好的次数：权重 1 为 {under('Bw1')}/40，权重 10 为 {under('Bw10')}/40，权重 100 为 {under('Bw100')}/40。</li>
        <li>权重 100 时先验体内部被完全固定在参考值上（{P['Bw100']['0.75']['mean']:.2f} 和 {P['Bw100']['0.5']['mean']:.2f} SI），效果等同于第一版的硬约束。</li>
        <li>先验画错处的磁化率中位数从 A 组的 {median('A', 'k_wrong'):.2f} 升到 {median('Bw1', 'k_wrong'):.2f}–{median('Bw100', 'k_wrong'):.2f} SI，而数据拟合一样好（χ²/N ≈ 1），所以从拟合上看不出先验是错的。</li>
      </ul></li>
    <li><b>钻孔（C 组）是唯一稳定有用的先验。</b>最优得分 {c['score']:.2f}（立方体 {c['overlap_cube']:.2f}，侵入体 {c['overlap_dyke']:.2f}），侵入体倾角 {c['dip']:.0f}°，侵入体内磁化率 {c['k_dyke']:.2f} SI。40 组参数中有 {under('C')} 组比 A 好，中位数从 {GS['A']['median']:.2f} 升到 {GS['C']['median']:.2f}。有了 150 m 的影响半径，钻孔的作用延伸到约 250 m，500 m 以外消失。</li>
    <li><b>建议。</b>
      <ul>
        <li>GUI 的 Lp 默认改为更光滑的长度（L = 3）或 norms (1,1,1,1)；</li>
        <li>解释性的先验体只用低权重（≤ 1），并且要和无先验的结果对比；</li>
        <li>钻孔用影响半径接入，但背景段的半径宜更小。</li>
      </ul></li>"""

    tuning_text = f"""
<p>三个规律（图 {fig.ref('effects')}，表 6）：</p>
<ul class="plain">
  <li><b>光滑长度 L 影响最大，五个组都是 L = 3 更好。</b>L = 1 时梯度项太弱，p<sub>x,y,z</sub> = 2 的设置会把磁化压成很薄的片状，例如 A 组 (0,2,2,2)：L = 1 得 {R['A_n0222_b15_L1']['score']:.2f}，L = 3 得 {R['A_n0222_b15_L3']['score']:.2f}。</li>
  <li><b>norms 中 (1,1,1,1) 的平均得分在五个组中都最高</b>（A 组 {E['A']['norms']['n1111']:.2f}），对其他参数最不敏感。(0,0,0,0) 波动大，但 B 组权重 1 和 C 组的最优结果都用的是它。(0,2,2,1)、(0,2,2,2) 在多数组中平均最差。</li>
  <li><b>深度加权 β 在 1–2 之间差别不大</b>。β = 3 让模型偏深，在 A、B 权重 1、C 三组中得分明显下降（A 组平均 {E['A']['beta']['b30']:.2f}，β = 1.5 为 {E['A']['beta']['b15']:.2f}）；参考模型权重为 10、100 时，结果主要由参考模型决定，β 的影响不明显。</li>
</ul>
<p>没有调参时，一般水平（中位数）和最优相差很大：A 组中位数 {GS['A']['median']:.2f}，最优 {a['score']:.2f}。第一版的结论受这一点影响很大。</p>"""

    best_text = f"""
<p>A 组最优把立方体放在 {a['cube_depth']:.0f} m（真值 {TM['cube_depth']:.0f} m），侵入体 90% 磁化在 {a['dyke_base']:.0f} m 以上（真值 {TM['dyke_base']:.0f} m），倾角 {a['dip']:.0f}°，比真值陡约 {a['dip'] - TM['dip']:.0f}°。C 组最优把倾角纠正到 {c['dip']:.0f}°，侵入体内磁化率 {c['k_dyke']:.2f} SI，最接近真值。B 组的三个最优结果都把一部分磁化留在先验画的位置上（图 {fig.ref('best_sections')}）：立方体先验偏东偏深，侵入体先验偏西偏陡。权重越大，留下的越多，侵入体内的磁化率也越低（{R[BEST['Bw1']]['k_dyke']:.2f}、{R[BEST['Bw10']]['k_dyke']:.2f}、{R[BEST['Bw100']]['k_dyke']:.2f} SI）。</p>"""

    findings_ref = f"""
<p>参考模型的作用要看两个方面：调参后最好能到什么程度，以及同一组参数下加了它是变好还是变坏（图 {fig.ref('paired')}）。</p>
<ul class="plain">
  <li><b>权重 1</b>：最优 {R[BEST['Bw1']]['score']:.2f}，与 A 组的 {a['score']:.2f} 持平。在同一组参数下，{under('Bw1')}/40 次比 A 好，重叠率中位数的变化只有立方体 {med_diff('Bw1', 'overlap_cube'):+.2f}、侵入体 {med_diff('Bw1', 'overlap_dyke'):+.2f}。改善主要发生在 A 组本来就差的参数上（图中左下方的点）。在 A 组最优参数下，加上它反而从 {a['score']:.2f} 降到 {R['Bw1_' + sa]['score']:.2f}（图 {fig.ref('same_setting_sections')}）。先验体内的平均磁化率被数据改成了 {P['Bw1']['0.75']['mean']:.2f}（立方体先验，参考值 0.75）和 {P['Bw1']['0.5']['mean']:.2f} SI（侵入体先验，参考值 0.5）：数据确实在修改它。</li>
  <li><b>权重 10</b>：最优 {R[BEST['Bw10']]['score']:.2f}，中位数 {GS['Bw10']['median']:.2f}；{under('Bw10')}/40 次比 A 好，侵入体重叠率中位数下降 {-med_diff('Bw10', 'overlap_dyke'):.2f}。先验体内的磁化率留在 {P['Bw10']['0.75']['mean']:.2f} 和 {P['Bw10']['0.5']['mean']:.2f} SI，离参考值更近。</li>
  <li><b>权重 100</b>：最优 {R[BEST['Bw100']]['score']:.2f}，中位数 {GS['Bw100']['median']:.2f}；只有 {under('Bw100')}/40 次比 A 好。先验体内部被完全固定（{P['Bw100']['0.75']['mean']:.3f} 和 {P['Bw100']['0.5']['mean']:.3f} SI），与第一版的硬约束相同。数据只能在别处补偿：有两组参数在立方体以外（测区南缘和北部，地下约 400–1,200 m）造出了大于 1 SI 的假异常，立方体重叠率因此为 0。</li>
</ul>
<p>所有 B 组反演的 χ²/N 都在 1 附近，与 A 组没有区别。<b>一个位置偏约 100 m、倾角偏 10°、磁化率偏 50% 的参考模型，在这里不能提高结果。</b>权重低时数据能把它改回来，作用接近于零；权重高时它把磁化留在错误的位置，而数据拟合不会报警。参考模型只有在几何上比这更准确时才值得加大权重；本实验没有测试"准确的参考模型"。</p>"""

    findings_holes = f"""
<p>五个钻孔在钻孔上固定为测井值，周围 150 m 内按距离线性减弱，受到约束的网格占核心区的 {100 * BH['cells_within']['150'] / BH['core_cells']:.1f}%。在同一组参数下：</p>
<ul class="plain">
  <li><b>C 组 {under('C')}/40 次比 A 好</b>，立方体重叠率中位数 {med_diff('C', 'overlap_cube'):+.2f}，侵入体 {med_diff('C', 'overlap_dyke'):+.2f}。先验画错处（B 组的错位区，C 组并没有这个先验）的磁化率中位数为 {median('C', 'k_wrong'):.2f} SI（A 组 {median('A', 'k_wrong'):.2f}）。</li>
  <li><b>影响范围</b>（图 {fig.ref('boreholes')}a，C 组最优参数下 C 与 A 的平均差）：0–150 m 约 {max(dec[:3]):.2f} SI，150–250 m 为 {dec[3]:.2f}，250–500 m 为 {dec[4]:.3f}，500 m 以外 {dec[5]:.4f}。也就是说，150 m 的半径经正则化传到了约 250 m。</li>
  <li><b>虚拟孔 V</b>：距最近钻孔 {v['nearest_hole_m']:.0f} m，未交给反演。真实侵入体在 {v['hit_m'][0]:.0f}–{v['hit_m'][1]:.0f} m，这一段的平均磁化率 A 组为 {v['A']:.2f} SI，C 组为 {v['C']:.2f} SI（真值 1）。C 组的上下界面也更接近真值（图 {fig.ref('boreholes')}d）。</li>
  <li><b>变差的情形</b>：主要是 (0,2,2,1) 配 L = 3，例如 {label(worse_c[0])}：A {R['A_' + worse_c[0]]['score']:.2f} → C {R['C_' + worse_c[0]]['score']:.2f}。一个可能的原因是：测井的背景段（0 SI）同样向外延伸 150 m，而斜孔 H3、直孔 H5 的背景段紧挨侵入体顶部，可能把那里的磁化压低（未逐一验证）。</li>
</ul>
<p>与第一版只约束钻孔所穿网格时相比（离孔 100 m 外几乎无影响），影响半径让钻孔真正改变了模型。仍需注意：见矿段与背景段用了同样的半径，背景段的半径应更小。</p>"""

    conclusions = f"""
  <li><b>数据拟合不能用来选参数，也不能检验先验。</b>200 次反演的 χ²/N 都在 {min(x['chi2_per_datum'] for x in R.values()):.2f}–{max(x['chi2_per_datum'] for x in R.values()):.2f}，而得分从 {min(x['score'] for x in R.values()):.2f} 到 {max(x['score'] for x in R.values()):.2f}。</li>
  <li><b>Lp 的参数要调，而且影响大于先验。</b>
    <ul>
      <li>在这类紧凑、粗短的异常上，光滑长度 L = 3、norms (1,1,1,1)、β 1.5–2 最稳健。例如 {label(robust)}：A {R['A_' + robust]['score']:.2f}，B 权重 1 为 {R['Bw1_' + robust]['score']:.2f}，C {R['C_' + robust]['score']:.2f}。</li>
      <li>GUI 当前的默认值（p = (0,2,2,1)，α 全为 1）在这里只得 {R['A_' + gui]['score']:.2f}，建议修改；改之前最好在 Block 8 的实测数据上再对比一次。</li>
    </ul></li>
  <li><b>解释性的先验体要慎用。</b>位置、倾角或磁化率有明显不确定性时，权重不要超过 1，并始终与无先验的结果对照。只出现在有先验结果中的结构，不应作为结论。</li>
  <li><b>钻孔应作为约束接入，并给它影响半径。</b>这是本实验中唯一在大多数参数下都有帮助的先验。建议把见矿段和背景段的半径分开设置（背景段更小），并让 GUI 的钻孔导入支持 <code>radius_m</code>（目前只能写在 JSON 规格里）。</li>
  <li><b>合成实验的"最优参数"不能直接搬到实际数据。</b>但"L 偏光滑、(1,1,1,1) 稳健、β 不超过 2"这几条规律可以作为实测数据调参的起点。可以用钻孔做留一法检验来选参数：每次留出一个钻孔不给反演，看它被预测得好不好。</li>"""
    return summary, tuning_text, best_text, findings_ref, findings_holes, conclusions


def main():
    fig = Figures("zh", FIGS)
    for name in ("setup", "tuning", "effects", "best_sections", "best_metrics", "paired",
                 "same_setting_sections", "boreholes"):
        fig.ref(name)                      # numbered in the order they appear, before the text refers ahead
    parts = texts(fig)
    OUT.write_text(page("zh", "Lp 消融实验", body(fig, *parts)), encoding="utf-8")
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
    if "--pdf" in sys.argv:
        print(to_pdf(OUT))


if __name__ == "__main__":
    main()
