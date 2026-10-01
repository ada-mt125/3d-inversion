"""Build the Karnataka gravity comparison report: Chinese and English HTML, optionally the English PDF.

    py examples/output/karnataka_gravity/scripts/make_figures.py
    py examples/output/karnataka_gravity/scripts/build_reports.py [--pdf]

Every number in the text comes from figures/numbers.json.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from common import FIGS, FULL, LOWRES, MAIN_HIGH, NW_HIGH, ROOT
from report_style import Figures, esc, page, table
# the published geology (karnataka_inputs/shared, put on the path by report_style)
from literature import LEGEND, POSITIONS_NOTE, RIDGE_MIN_M, RIDGE_RELIEF_M, cite, references_html

N = json.loads((FIGS / "numbers.json").read_text(encoding="utf-8"))
F, LOW, S8, SP, D, G = N["full"], N["low"], N["simpeg_8km"], N["spectrum"], N["data"], N["geology"]
KEYS = [k for k, *_ in FULL]
NEW = KEYS[1:]                      # everything except the original settings
DW = ["as1_beta0.5", "as1_beta1", "as1_beta1.5", "as0.1_beta1"]
OUT = {"zh": ROOT / "karnataka_gravity_report_zh.html", "en": ROOT / "karnataka_gravity_report_en.html"}
REFS = ["MM93", "GSI", "IJERT", "MEAI", "MGR23", "IBMK", "IBMN", "GEM", "SB14", "SIN20", "NMET"]
LOC_REFS = ["IBMK", "IBMN", "GEM", "SB14", "SIN20", "NMET"]      # sources of the mine and occurrence positions


def label(key, lang):
    """Run label as HTML (the figure labels write α_s for α with subscript s)."""
    text = next(zh if lang == "zh" else en for k, zh, en, *_ in FULL if k == key)
    return esc(text).replace("α_s", "α<sub>s</sub>")


def mgal(x):
    return f"{x:.1f}".replace("-", "−")


def pct(x):
    return f"{100 * x:.0f}%"


def rng(b):
    return f"{b['top_km']:.1f}–{b['bottom_km']:.1f}"


def cen(key, which="main"):
    return F[key][which]["centroid_km"]


def lc(key):
    return LOW[key]["main"]["centroid_km"]


def span(values, fmt="{:.2f}"):
    return f"{fmt.format(min(values))}–{fmt.format(max(values))}"


def warn(text, bad):
    return (text, "n warn" if bad else "n")


def half(x):
    """A distance rounded to the nearest half kilometre."""
    return f"{round(2 * x) / 2:g}"


def xy(p, sep=", "):
    return f"{p[0] / 1e3:.1f}{sep}{p[1] / 1e3:.1f} km"


def cite_zh(*keys, **kw):
    """literature.cite with Chinese parentheses and separators."""
    return "（" + cite(*keys, **kw)[1:-1].replace("</a>; <a", "</a>；<a") + "）"


def refs_zh():
    note = ("矿山和矿点的位置已换算到 UTM 43N（EPSG:32643）。取自 GEM.wiki 和论文中经纬度范围的位置是近似的，误差约 1 km，"
            "Mincheri 区块的轮廓也是近似的。Kumaraswamy 和 NEB Range 的坐标采用 IBM 检查报告中印出的数值，按 UTM 43N 读取。")
    return references_html(REFS).replace("<h2>References</h2>", "<h2>参考文献</h2>").replace(POSITIONS_NOTE, note)


# numbers quoted in the text
C = N["integrated_corr"]
_ix = {k: i for i, k in enumerate(C["keys"])}
_dw_lat = [LOW[k]["lateral"] for k in LOWRES if "_dw" in k and k.startswith("as1_")]
V = dict(
    chi=span([F[k]["chi2"] for k in KEYS]), rms=span([F[k]["rms"] for k in KEYS]),
    corr_min=f"{C['min_offdiag']:.2f}",
    corr_max=f"{max(C['matrix'][i][j] for i in range(len(KEYS)) for j in range(len(KEYS)) if i != j):.2f}",
    corr_new=f"{min(C['matrix'][_ix[a]][_ix[b]] for a in NEW for b in NEW if a != b):.2f}",
    core_min=pct(min(F[k]["core"] for k in NEW if k != "as1_beta1.5")),
    nw_new=span([cen(k, "nw") for k in NEW], "{:.1f}"),
    nw_gap=span([cen(k) - cen(k, "nw") for k in NEW], "{:.1f}"),
    lowgap=max(abs(lc(lk) - cen(fk)) for lk, fk in
               [("as1_p0222_dw0.5", "as1_beta0.5"), ("as1_p0222_dw1", "as1_beta1"), ("as1_p0222_dw1.5", "as1_beta1.5")]),
    as01gap=abs(lc("as0.1_p0222_dw1") - cen("as0.1_beta1")),
    dw_lat=f"{pct(min(_dw_lat))}–{pct(max(_dw_lat))}",
    time8=S8["time_s"] / 60,
    # against the schematic belt of literature.py (Section 6)
    dense_ref=pct(G["dense"][G["run"]]["inside"]),
    dense_all=f"{pct(min(v['inside'] for v in G['dense'].values()))}–{pct(max(v['inside'] for v in G['dense'].values()))}",
    outline_share=pct(G["outline_area_share"]),
)
BELT, RH, R5 = G["belt"], G["residual_high"], G["residual_5mgal"]


# ---------------------------------------------------------------- tables

def runs_table(lang):
    zh = lang == "zh"
    norms_h = "p = [p<sub>s</sub>, p<sub>x</sub>, p<sub>y</sub>, p<sub>z</sub>]"
    head = (["运行", "正则化", "α<sub>s</sub>", "范数 " + norms_h, "深度加权", "提交方式"] if zh else
            ["Run", "Regularization", "α<sub>s</sub>", "Norms " + norms_h, "Depth weighting", "Submitted from"])
    script, sweep = ("脚本 ec2_multi_run.py", "上传页面（参数扫描）") if zh else ("script ec2_multi_run.py", "upload page (sweep)")
    rows = []
    for k in KEYS:
        s = F[k]["settings"]
        if s["regularization_type"] == "l1l2":
            reg = "弹性网（Utsugi 2019），L1 占 0.8，IRLS 求解" if zh else "elastic net (Utsugi 2019), L1 share 0.8, solved by IRLS"
            a_s, norms = "—", "—"
        else:
            reg, a_s = "sparse (IRLS)", f"{s['alpha_s']:g}"
            norms = "[" + ", ".join(f"{p:g}" for p in s["norms"]) + "]"
        if s.get("depth_weighting") == "depth":
            w = f"Li &amp; Oldenburg, β = {s['depth_weighting_exponent']:g}"
        else:
            w = "灵敏度" if zh else "sensitivity"
        via = sweep if k in ("as1_beta0.5", "as1_beta1.5") else script
        rows.append([label(k, lang), (reg, "wrap"), a_s, norms, w, via])
    return table(head, rows, numeric_from=99)


def fit_table(lang):
    head = (["运行", "χ²/N", "RMS (mGal)", "最大 |残差| (mGal)", "观测与预测的相关系数"] if lang == "zh" else
            ["Run", "χ²/N", "RMS (mGal)", "Max |residual| (mGal)", "Correlation, observed vs predicted"])
    rows = [[label(k, lang), f"{F[k]['chi2']:.2f}", f"{F[k]['rms']:.2f}", f"{F[k]['max_abs']:.1f}",
             f"{F[k]['corr']:.4f}"] for k in KEYS]
    return table(head, rows, compact=True)


def corr_table(lang):
    short = (["原设置", "L1–L2", "β = 0.5", "β = 1", "β = 1.5", "α<sub>s</sub> = 0.1"] if lang == "zh" else
             ["Original", "L1–L2", "β = 0.5", "β = 1", "β = 1.5", "α<sub>s</sub> = 0.1"])
    M = C["matrix"]
    rows = [[short[i]] + [("—", "n") if i == j else (f"{M[i][j]:.2f}", "n good" if M[i][j] >= 0.9 else "n")
                          for j in range(len(short))] for i in range(len(short))]
    return table([""] + short, rows, compact=True)


def depth_table(lang):
    zh = lang == "zh"
    head = (["运行", "核心区内", "侧向 padding", "10 km 以下", "D50 km", "D90 km",
             "主高值 半峰值 km", "主高值 质心 km", "西北高值 半峰值 km", "西北高值 质心 km"] if zh else
            ["Run", "Core", "Lateral padding", "Below 10 km", "D50 km", "D90 km",
             "Main high, half-max km", "Main high, centroid km", "NW high, half-max km", "NW high, centroid km"])
    rows = []
    for k in KEYS:
        f = F[k]
        rows.append([label(k, lang), pct(f["core"]), warn(pct(f["lateral"]), f["lateral"] > 0.2),
                     warn(pct(f["below"]), f["below"] > 0.3), warn(f"{f['D50']:.1f}", f["D50"] > 7), f"{f['D90']:.1f}",
                     warn(rng(f["main"]), f["main"]["bottom_km"] > 12), f"{cen(k):.1f}", rng(f["nw"]),
                     f"{cen(k, 'nw'):.1f}"])
    rows.append(["SimPEG 8 km 深网格（9 月 24 日）" if zh else "SimPEG, 8 km deep mesh (24 Sep)", "—", "—", "—", "—", "—",
                 rng(S8["main"]), f"{S8['main']['centroid_km']:.1f}", rng(S8["nw"]), f"{S8['nw']['centroid_km']:.1f}"])
    return table(head, rows)


def lowres_table(lang):
    zh = lang == "zh"
    head = (["运行", "α<sub>s</sub>", "范数 p", "深度加权", "χ²/N", "核心区内", "侧向 padding", "10 km 以下", "D50 km",
             "主高值 半峰值 km", "主高值 质心 km"] if zh else
            ["Run", "α<sub>s</sub>", "Norms p", "Weighting", "χ²/N", "Core", "Lateral padding", "Below 10 km", "D50 km",
             "Main high, half-max km", "Main high, centroid km"])
    rows = []
    for k in LOWRES:
        f, s = LOW[k], LOW[k]["settings"]
        w = (f"β = {s['depth_weighting_exponent']:g}" if s.get("depth_weighting") == "depth"
             else ("灵敏度" if zh else "sensitivity"))
        norms = "[" + ",".join(f"{p:g}" for p in s["norms"]) + "]"
        rows.append([f"<code>{k}</code>", (f"{s['alpha_s']:g}", ""), (norms, ""), (w, ""), f"{f['chi2']:.2f}",
                     pct(f["core"]), warn(pct(f["lateral"]), f["lateral"] > 0.2), warn(pct(f["below"]), f["below"] > 0.3),
                     warn(f"{f['D50']:.1f}", f["D50"] > 7), warn(rng(f["main"]), f["main"]["bottom_km"] > 12),
                     f"{lc(k):.1f}"])
    return table(head, rows, numeric_from=4)


# ---------------------------------------------------------------- Chinese

def body_zh(fig):
    o, t = F["original_sparse"], F
    return f"""
<header>
  <div class="eyebrow">GeoInv3D · 野外数据测试 · 2026 年 9 月 28 日 · 文献对照 2026 年 10 月 1 日</div>
  <h1>Karnataka 重力反演：不同正则化的对比</h1>
  <p class="lede">同一套布格重力数据、同一个网格，六组正则化设置在 AWS EC2 上做了全分辨率反演。本报告比较它们的数据拟合、横向结构和深度，与 9 月 24 日 SimPEG 8 km 深网格和 Tomofast-x 的结果交叉检验，并检查 2 km 粗网格实验能否代替全分辨率计算。</p>
  <dl class="meta">
    <div><dt>区域</dt><dd>70 × 70 km，东向 641–711 km，北向 1634–1704 km（UTM 43N，EPSG:32643）</dd></div>
    <div><dt>数据</dt><dd>NGPM 布格异常，500 m 网格抽稀到 1 km：{D['n']:,} 个点；去除二阶趋势面</dd></div>
    <div><dt>网格</dt><dd>核心区 1 km × 1 km × 500 m，深 10 km；侧向和底部约 20 km padding（扩展系数 1.3）：84 × 84 × 27 = 190,512 个单元</dd></div>
    <div><dt>运行</dt><dd>6 组全分辨率反演在 AWS EC2（ap-south-1）完成，其中 2 组从上传页面提交；16 组 2 km 网格实验在本地完成</dd></div>
  </dl>
</header>

<section class="summary" aria-labelledby="sum">
  <h2 id="sum">摘要</h2>
  <ul>
    <li><b>六组结果的数据拟合几乎一样。</b>χ²/N 在 {V['chi']} 之间，残差 RMS {V['rms']} mGal。只看拟合无法在这些模型之间做选择。</li>
    <li><b>横向结构是可靠的。</b>各模型垂向积分密度图的两两相关系数为 {V['corr_min']}–{V['corr_max']}，去掉原设置后不低于 {V['corr_new']}。NW–SE 向的高密度带和东侧几条低密度带在每个模型中位置、走向都一致。</li>
    <li><b>深度由正则化决定。</b>主布格高值正下方的质心深度：β = 0.5 为 {cen('as1_beta0.5'):.1f} km，L1–L2 {cen('l1l2_irls'):.1f} km，β = 1 {cen('as1_beta1'):.1f} km，α<sub>s</sub> = 0.1 {cen('as0.1_beta1'):.1f} km，β = 1.5 {cen('as1_beta1.5'):.1f} km。原设置得到的是从地表延伸到网格底部（{o['main']['bottom_km']:.1f} km）的柱子。β 越大模型越深，而 χ²/N 没有相应的变化。</li>
    <li><b>不同的设置给出相同的深度。</b>9 月 24 日 SimPEG 在 8 km 深网格上的结果（α<sub>s</sub> = 0.05，灵敏度加权），主体半峰值范围 {rng(S8['main'])} km、质心 {S8['main']['centroid_km']:.1f} km，与本报告 β = 1 的 {rng(t['as1_beta1']['main'])} km、{cen('as1_beta1'):.1f} km 一致。Tomofast-x 拟合更紧（χ = 0.65），但模型没有底，并且用了约 2.7 小时，SimPEG 只用了 {V['time8']:.0f} 分钟。</li>
    <li><b>2 km 粗网格实验可以用来筛选参数。</b>三个 β 值下，粗网格预测的主体质心与 1 km 结果相差不超过 {V['lowgap']:.1f} km；α<sub>s</sub> = 0.1 相差 {V['as01gap']:.1f} km。粗网格每次约 25 秒，全分辨率每次约 5 分钟。</li>
    <li><b>建议。</b>α<sub>s</sub> = 1（已是流程默认值），Li &amp; Oldenburg 深度加权，β 取 0.5–1，给出一个深度范围而不是单一模型：β = 0.5 与 L1–L2 一致，β = 1 与 8 km 深网格的结果一致。</li>
    <li><b>与已发表的地质资料对照（第 6 节）。</b>高密度带与地质图上的 Sandur 片岩带重合，带内的四个铁矿山都在这条带上。本报告的深度范围包含了唯一一个已发表的估计，即深约 6 km 的盆地{cite_zh('MGR23')}。</li>
    <li><b>注意。</b>仅凭重力数据不能确定深度，本报告中的深度是正则化选择的结果。实际深度需要岩石密度、地质剖面或地震资料来约束。</li>
  </ul>
</section>

<h2><span class="no">1</span>数据与参与对比的运行</h2>
<div class="prose">
<p>区内布格异常为 <span class="num">{mgal(D['raw'][0])}</span> 到 <span class="num">{mgal(D['raw'][1])}</span> mGal，主要特征是从 (650, 1685) 到 (675, 1660) km 的 NW–SE 向高值带。去除最小二乘二阶趋势面（区域场，<span class="num">{mgal(D['regional'][0])}</span> 到 <span class="num">{mgal(D['regional'][1])}</span> mGal）后，剩余异常为 <span class="num">{mgal(D['residual'][0])}</span> 到 <span class="num">+{D['residual'][1]:.1f}</span> mGal，这就是反演的输入（图 {fig.ref('data')}）。每个点的误差取 0.5 mGal。布格异常向下为正，SimPEG 的 g<sub>z</sub> 向上为正，流程在反演前把数据乘以 −1，结果中的观测和预测数据按原来的符号返回。</p>
</div>
{fig('data', "左：区内布格异常（抽稀到 1 km）。中：最小二乘二阶趋势面，作为区域场去除。右：剩余异常，即反演的数据（红色为正，高密度）。")}
<div class="prose">
<p>六组运行使用相同的网格、数据和误差，只改变正则化：</p>
</div>
{runs_table('zh')}
<p class="note">共同设置：密度差上下界 −0.2 到 +0.5 g/cc（岩样密度 2.52–3.40 g/cc，中值 2.66）；β 按 χ² = N 自动选取；最多 30 次迭代、30 次 IRLS 迭代。α<sub>x</sub> = α<sub>y</sub> = α<sub>z</sub> = 1，按长度尺度传给 SimPEG。Li &amp; Oldenburg 深度加权在模型范数中的权重为 (z + z₀)<sup>−β/2</sup>，z 为到最近测点的深度，z₀ 为最小单元的一半。</p>

<h2><span class="no">2</span>全分辨率结果的对比</h2>

<h3><span class="no">2.1</span>数据拟合</h3>
<div class="prose">
<p>六组都达到了目标拟合，统计量几乎一样。残差的空间分布（图 {fig.ref('residuals')}）则有区别：</p>
<ul class="plain">
  <li><b>原设置：</b>残差集中在几处梯度很陡的地方，形成成片的正负残差，最大 {o['max_abs']:.1f} mGal。柱状模型很难同时拟合相邻的窄异常。</li>
  <li><b>L1–L2：</b>主高值上方有一片大范围的正残差，周围为负，说明模型整体上低估了主异常的幅度（主体已达到 +0.5 g/cc 的上界）。它的 χ²/N 也最高（{t['l1l2_irls']['chi2']:.2f}）。</li>
  <li><b>深度加权的四组：</b>残差最散，主要是沿走向的短波长条带和几个孤立点。(660, 1680) 和 (702, 1659) km 附近的孤立点在每个模型中都出现，很可能是原始网格化带来的离群值。</li>
</ul>
</div>
{fit_table('zh')}
{fig('residuals', "六个全分辨率模型的残差（观测 − 预测），色标 ±3 mGal。")}

<h3><span class="no">2.2</span>横向结构</h3>
<div class="prose">
<p>图 {fig.ref('slices')} 是五个模型在 1.8、4.8 和 7.8 km 深度的切片。图 {fig.ref('robust')} 上排是垂向积分密度（每一列密度差乘厚度之和），下排是每一列的质心深度。</p>
<ul class="plain">
  <li><b>积分密度图几乎一样。</b>两两相关系数 {V['corr_min']}–{V['corr_max']}，去掉原设置后不低于 {V['corr_new']}（下表）。NW–SE 向高密度带、它西北端的次级高值以及东侧几条平行的低密度带，在每个模型中位置和走向都相同。</li>
  <li><b>不同的是强弱和边界。</b>β 越大，积分密度越强、边界越清楚；原设置的模型最平滑。</li>
  <li><b>质心深度图完全不同。</b>同一条高密度带，β = 0.5 和 L1–L2 的质心在 2–4 km，β = 1 在 4–6 km，β = 1.5 在 6–8 km（局部超过 12 km），原设置在 10 km 以下。</li>
  <li><b>切片也说明了这一点。</b>β = 0.5 在 7.8 km 已经没有异常，β = 1.5 在 1.8 km 只剩零碎的小块，原设置在三个深度几乎一样。</li>
</ul>
</div>
{corr_table('zh')}
<p class="note">核心区（70 × 70 km）内垂向积分密度图之间的相关系数。绿色：≥ 0.90。</p>
{fig('slices', "五个模型在三个深度的切片（只显示核心区），色标 ±0.3 g/cc。")}
{fig('robust', "上排：垂向积分密度（g/cc·km），五个模型的图形基本一致。下排：每一列 |密度差| 的质心深度（只显示积分值超过最大值 20% 的列），随正则化变化很大。")}

<h3><span class="no">2.3</span>深度</h3>
<div class="prose">
<p>图 {fig.ref('sections')} 是过主布格高值（北向 1664.5 km）的东西向剖面，图 {fig.ref('mass_profiles')} 左是核心区下方 |质量| 随深度的分布。下表列出质量的分布，以及主布格高值（东向 671.5 km，北向 1664.5 km）和西北高值（650.5，1682.5 km）正下方密度剖面的指标。</p>
<p>深度用<b>质心</b>（正密度差按厚度加权的平均深度）表示，而不是峰值深度：α<sub>s</sub> = 1 时主体达到 +0.5 g/cc 的上界，峰值是一段平台，没有唯一的深度。半峰值范围是密度超过峰值一半的深度区间。</p>
</div>
{depth_table('zh')}
<p class="note">核心区内、侧向 padding 和 10 km 以下三项之和为 100%。D50、D90：核心区下方一半和 90% 的 |质量| 位于该深度以上。红色：侧向 padding &gt; 20%、10 km 以下 &gt; 30%、D50 &gt; 7 km 或主体底部深于 12 km。8 km 深网格是另一个网格，质量分布指标不可比，只列剖面指标。</p>
{fig('sections', "六个模型过主布格高值的东西向剖面（北向 1664.5 km）。竖直点线为主高值位置，水平虚线为核心网格底界（10 km），以下为 padding。色标 ±0.3 g/cc。")}
<div class="prose">
<ul class="plain">
  <li><b>原设置是柱子。</b>主高值下方密度差只有 {o['main']['peak']:.2f} g/cc，从 {o['main']['top_km']:.1f} km 一直延伸到 {o['main']['bottom_km']:.1f} km，质心 {cen('original_sparse'):.1f} km；{pct(o['below'])} 的质量在 10 km 以下，{pct(o['lateral'])} 在侧向 padding。</li>
  <li><b>其他五组都有底。</b>主体在 12 km 以上结束，{V['core_min']} 以上的质量在核心区内（β = 1.5 为 {pct(t['as1_beta1.5']['core'])}）。</li>
  <li><b>β 单调地控制深度。</b>β = 0.5、1、1.5 的主体质心为 {cen('as1_beta0.5'):.1f}、{cen('as1_beta1'):.1f}、{cen('as1_beta1.5'):.1f} km，半峰值范围 {rng(t['as1_beta0.5']['main'])}、{rng(t['as1_beta1']['main'])}、{rng(t['as1_beta1.5']['main'])} km，而 χ²/N 分别为 {t['as1_beta0.5']['chi2']:.2f}、{t['as1_beta1']['chi2']:.2f}、{t['as1_beta1.5']['chi2']:.2f}，与深度没有关系。这就是重力反演的非唯一性：数据不能决定 β。</li>
  <li><b>L1–L2 与 β = 0.5 几乎相同。</b>两种完全不同的正则化给出同一个半峰值范围（{rng(t['l1l2_irls']['main'])} km），质心相差 {abs(cen('l1l2_irls') - cen('as1_beta0.5')):.1f} km。</li>
  <li><b>α<sub>s</sub> = 0.1 介于 β = 1 和 β = 1.5 之间。</b>主体质心 {cen('as0.1_beta1'):.1f} km，峰值 {t['as0.1_beta1']['main']['peak']:.2f} g/cc，接近但没有达到上界；模型比 α<sub>s</sub> = 1 更平滑，西北高值从 {t['as0.1_beta1']['nw']['top_km']:.1f} km 延伸到 {t['as0.1_beta1']['nw']['bottom_km']:.1f} km。</li>
  <li><b>β = 1.5 开始向深部拖尾。</b>东侧的低密度体延伸到 20 km（图 {fig.ref('sections')}），{pct(t['as1_beta1.5']['below'])} 的质量在 10 km 以下。</li>
  <li><b>西北高值总是更浅。</b>除原设置外，西北高值的质心在 {V['nw_new']} km，比主高值浅 {V['nw_gap']} km。这个相对关系在每个模型中都成立。</li>
</ul>
</div>
{fig('mass_profiles', "核心区下方 |质量| 随深度的分布（每 km 的百分比）。左：六个全分辨率模型。右：2 km 实验中比较 α<sub>s</sub> 和深度加权的五组（第 4 节）。虚线为核心网格底界。")}
<div class="prose">
<p>剩余异常的径向平均功率谱（图 {fig.ref('spectrum')}）有三段直线，对应的平均源顶深度为 {SP['deep']:.1f}、{SP['mid']:.1f} 和 {SP['shallow']:.1f} km（Spector &amp; Grant）。它说明场源分布在不同深度，但不能在模型之间做出选择：β = 0.5 和 L1–L2 的主体顶部（{t['as1_beta0.5']['main']['top_km']:.1f} km）接近中部估计，β = 1 的质心（{cen('as1_beta1'):.1f} km）接近深部估计。功率谱给出的是一组场源的平均顶深，不确定性很大。</p>
</div>
{fig('spectrum', "剩余异常的径向平均功率谱（500 m 网格，已去除二阶趋势）。三段直线的斜率给出场源的平均顶深 h = −斜率 / 4π。", narrow=True)}

<h3><span class="no">2.4</span>哪些结论可靠</h3>
<div class="prose">
<p><b>在所有设置下都一致的：</b></p>
<ul class="plain">
  <li>异常体的横向位置、走向和范围：NW–SE 向高密度带和东侧的低密度带。</li>
  <li>符号和相对强弱：主布格高值下方是最强的正密度体。</li>
  <li>主体有底（除原设置外的五个模型，以及独立的 8 km 深网格结果都是如此）；西北高值比主高值浅。</li>
</ul>
<p><b>随正则化改变的：</b></p>
<ul class="plain">
  <li>绝对深度：主体质心 {cen('as1_beta0.5'):.1f}–{cen('as1_beta1.5'):.1f} km，取决于 β。</li>
  <li>厚度和密度差：α<sub>s</sub> = 1 时主体达到 +0.5 / −0.2 g/cc 的上下界，这是正则化造成的块状、饱和的模型，不是对密度的测量。</li>
  <li>深部细节，例如 β = 1.5 中东侧低密度体延伸到 20 km。</li>
</ul>
<p>χ²/N（{V['chi']}）不能用来在这些模型之间做选择。</p>
</div>

<h2><span class="no">3</span>交叉验证：SimPEG 8 km 深网格与 Tomofast-x</h2>
<div class="prose">
<p>9 月 24 日用两个独立的程序反演了同一套 Karnataka 重力数据：本项目使用的 SimPEG，以及开源并行程序 Tomofast-x 2.0。两者使用同一个 8 km 深的网格（水平 500 m、垂向 250 m 的单元）、同一台云主机（16 vCPU，123 GB 内存）和同一个噪声模型。本节只摘录重力部分，磁法部分见原报告（Desktop/Tomofastx_vs_SimPEG_package）。</p>
</div>
<div class="tablewrap"><table>
<thead><tr><th></th><th>SimPEG</th><th>Tomofast-x 2.0</th></tr></thead>
<tbody>
<tr><td>数据拟合 χ</td><td class="n">{S8['chi']:.2f}</td><td class="n">0.65</td></tr>
<tr><td>使用的数据</td><td>{S8['n_data']:,} 个点（隔一个取一个）</td><td>19,600 个点（全部）</td></tr>
<tr><td>主布格高值下方</td><td class="wrap">半峰值 {rng(S8['main'])} km，质心 {S8['main']['centroid_km']:.1f} km，峰值 {S8['main']['peak']:.2f} g/cc；9.2 km 处只剩 {S8['at_9km']:.2f} g/cc，而网格深到 {S8['deepest_km']:.1f} km</td><td class="wrap">没有底：每层密度的 1%、50%、99% 分位数从地表到 8 km 变化不到 10%</td></tr>
<tr><td>正则化</td><td class="wrap">一次设定：sparse，范数 (0, 2, 2, 1)，α<sub>s</sub> = 0.05，基于灵敏度的深度加权</td><td class="wrap">24 次试算搜索两个光滑权重，仍然没有形状</td></tr>
<tr><td>得到最终模型的时间</td><td>{V['time8']:.0f} 分钟（{S8['time_s']:.0f} s）</td><td>约 2.7 小时（参数搜索 1.7 h + 完整运行 1.0 h）</td></tr>
</tbody></table></div>
{fig('tomofastx', "过异常中心的东西向和南北向剖面（仅重力，取自原报告）。左：SimPEG，高密度体上下都封闭；右：Tomofast-x，从地表到网格底部的一组竖条。注意两边色标不同。")}
{fig('centre_profiles', "主布格高值正下方的密度随深度变化。红色：SimPEG 8 km 深网格（9 月 24 日，没有设上下界）；其余为本报告的六个全分辨率模型。虚线为 8 km 网格最深的单元。", narrow=True)}
<div class="prose">
<h3>与本报告的关系</h3>
<ul class="plain">
  <li><b>9 月 24 日的 SimPEG 模型为什么有底。</b>它用的是 α<sub>s</sub> = 0.05。在长度尺度的 α 下，最小模型项约为梯度项的五分之一，紧凑约束能起作用；本报告的原设置用 α<sub>s</sub> = 10⁻⁴，弱约 2500 倍，结果是柱子（第 5 节）。</li>
  <li><b>两个不同的设置给出同一深度。</b>8 km 网格（500 m 单元，灵敏度加权，{S8['n_data']:,} 个数据）得到 {rng(S8['main'])} km、质心 {S8['main']['centroid_km']:.1f} km；1 km 张量网格、α<sub>s</sub> = 1、β = 1（{D['n']:,} 个数据）得到 {rng(t['as1_beta1']['main'])} km、质心 {cen('as1_beta1'):.1f} km（图 {fig.ref('centre_profiles')}）。西北高值也一致（质心 {S8['nw']['centroid_km']:.1f} km 与 {cen('as1_beta1', 'nw'):.1f} km）。这说明结果对网格和数据抽稀不敏感；但两者都是 SimPEG 中同一类有深度加权的紧凑模型，并不能消除非唯一性：β = 0.5 和 L1–L2 更浅，β = 1.5 更深，数据分不出来。</li>
  <li><b>χ 好不等于模型好。</b>Tomofast-x 的 χ = 0.65，比目标还低，模型却没有任何垂向形状；本报告的 β 扫描也说明，χ 几乎相同的模型深度可以相差一倍。只看 χ 的自动参数搜索会选出没有物理意义的设置。已发表的 Tomofast-x 应用（Bird et al., 2025，南极水深反演）得出同样结论，改用 72 次完整长度的试算来选参数。</li>
</ul>
<h3>计算成本（仅重力）</h3>
</div>
<div class="tablewrap"><table class="compact">
<thead><tr><th>阶段</th><th class="n">大约用时</th></tr></thead>
<tbody>
<tr><td>SimPEG 重力最终运行（8 km 网格，16 vCPU）</td><td class="n">{V['time8']:.0f} 分钟（{S8['time_s']:.0f} s）</td></tr>
<tr><td>Tomofast-x 重力：24 次试算的参数搜索</td><td class="n">1.7 小时</td></tr>
<tr><td>Tomofast-x 重力：完整长度运行</td><td class="n">1.0 小时</td></tr>
<tr><td>SimPEG（本报告，EC2 c5.2xlarge，1 km 网格）</td><td class="n">每组 5–6 分钟，约 $0.03</td></tr>
</tbody></table></div>
<div class="prose">
<p>Tomofast-x 的成本主要来自一点：它不会自动调节正则化，两个光滑权重必须在运行前通过反复试算确定，每次试算在完整问题上要 12–17 分钟。SimPEG 在运行中自动调节 β，不需要搜索。</p>
<h3>这个对比不能说明的</h3>
<ul class="plain">
  <li><b>SimPEG 用的数据更少：</b>{S8['n_data']:,} 个点，Tomofast-x 用了 19,600 个。Tomofast-x 用小波压缩灵敏度矩阵，能处理全部数据，这是它真正的优势，本对比没有体现。</li>
  <li><b>SimPEG 也有自己的问题：</b>当时流程中的两个错误（一个不起作用的网格深度设置、一个缺失的求解器加速）花了几个小时才找到。它们已经修复，但这些时间没有计入上表。</li>
  <li><b>Tomofast-x 在这里能收敛：</b>在 3 km 深的网格上，通过参数搜索达到了目标拟合（重力 χ = 1.87）。它的问题是在深部得不到有界的密度体，而不是拟合不了数据。</li>
  <li><b>可能没有用好 Tomofast-x：</b>它的已发表应用还用了上下界约束等控制，我们没有试。它的光滑项不能像 SimPEG 那样做成"块状"，这是我们对"向下复制"现象的最好解释，但只是假设，没有进一步检验。</li>
  <li><b>一套数据、一位操作者：</b>只有这一个测区，由一位刚接触 Tomofast-x 的人完成。</li>
</ul>
</div>

<h2><span class="no">4</span>2 km 参数实验与全分辨率结果</h2>
<div class="prose">
<p>在全分辨率计算之前，先在本地用 2 km × 2 km × 1 km 的网格（每 4 个点取 1 个，1,295 个数据）跑了 16 组 sparse 反演，每组约 25 秒。下表列出全部 16 组，图 {fig.ref('mass_profiles')} 右是其中五组的质量-深度分布。</p>
</div>
{lowres_table('zh')}
<p class="note">"灵敏度"为 SimPEG 的灵敏度加权，"β = …"为 Li &amp; Oldenburg 深度加权。红色标记规则同第 2.3 节。</p>
<div class="prose">
<h3>粗网格上看到的规律</h3>
<ul class="plain">
  <li><b>α<sub>s</sub> 决定模型是否延伸到网格底。</b>灵敏度加权下，α<sub>s</sub> = 10⁻⁴、10⁻²、0.1、1 时 10 km 以下的质量为 {pct(LOW['base']['below'])}、{pct(LOW['as1e-2']['below'])}、{pct(LOW['as0.1']['below'])}、{pct(LOW['as1']['below'])}。</li>
  <li><b>深度加权决定侧向泄漏。</b>α<sub>s</sub> = 1 时，灵敏度加权有 {pct(LOW['as1_p0222']['lateral'])} 的质量在侧向 padding，深度加权只有 {V['dw_lat']}。</li>
  <li><b>β 决定平均深度。</b>β = 0.5、1、1.5、2 的 D50 为 {LOW['as1_p0222_dw0.5']['D50']:.1f}、{LOW['as1_p0222_dw1']['D50']:.1f}、{LOW['as1_p0222_dw1.5']['D50']:.1f}、{LOW['as1_p0222_dw2']['D50']:.1f} km，主体质心 {lc('as1_p0222_dw0.5'):.1f}、{lc('as1_p0222_dw1'):.1f}、{lc('as1_p0222_dw1.5'):.1f}、{lc('as1_p0222_dw2'):.1f} km。重力常用的 β = 2 在这里最深，已有 {pct(LOW['as1_p0222_dw2']['below'])} 的质量在 10 km 以下。</li>
  <li><b>范数影响较小。</b>α<sub>s</sub> = 1、β = 1 时，p = [0,2,2,2]、[0,2,2,1]、[0,1,1,1] 的主体质心为 {lc('as1_p0222_dw1'):.1f}、{lc('as1_p0221_dw1'):.1f}、{lc('as1_p0111_dw1'):.1f} km。</li>
</ul>
<h3>和 1 km 结果比较</h3>
<p>在同一位置比较（图 {fig.ref('lowres_vs_full')}）：三个 β 值下粗网格的主体质心与 1 km 结果相差不超过 {V['lowgap']:.1f} km，底界相差不超过 0.5 km；顶部在粗网格上偏浅（β = 0.5 为 {LOW['as1_p0222_dw0.5']['main']['top_km']:.1f} km 对 {t['as1_beta0.5']['main']['top_km']:.1f} km），这是 1 km 厚的单元造成的。α<sub>s</sub> = 0.1 是例外：粗网格上主体没有饱和（峰值 {LOW['as0.1_p0222_dw1']['main']['peak']:.2f} g/cc），范围 {rng(LOW['as0.1_p0222_dw1']['main'])} km，质心比 1 km 结果浅 {V['as01gap']:.1f} km。所以粗网格适合筛选 β 和比较趋势，最终结果仍要在全分辨率上确认。</p>
</div>
{fig('lowres_vs_full', "主布格高值下方的半峰值深度范围（色条）和质心（圆点）：2 km 实验（灰色）与 1 km 全分辨率（蓝色）。", narrow=True)}

<h2><span class="no">5</span>原设置为什么得到柱状体</h2>
<div class="prose">
<p>原设置为 sparse（IRLS），p = [0, 2, 2, 1]，α<sub>s</sub> = 10⁻⁴，α<sub>x</sub> = α<sub>y</sub> = α<sub>z</sub> = 1（长度尺度），灵敏度加权。主要有三个原因：</p>
<ol class="steps">
  <li><b>最小模型项几乎不起作用。</b>流程把 α<sub>x,y,z</sub> 当作长度尺度传给 SimPEG，梯度项的权重为 (1 × 500 m)²，每个单元约 0.25 Δm²，比 10⁻⁴ m² 的最小模型项强约 2500 倍，紧凑约束 p<sub>s</sub> = 0 因此几乎无效。</li>
  <li><b>垂向梯度用 L1。</b>p<sub>z</sub> = 1 偏好随深度分段不变的模型，一根不随深度变化的柱子几乎没有代价。</li>
  <li><b>灵敏度加权让 padding 很便宜。</b>测区外和深部的单元灵敏度低、权重也低，质量放在那里代价很小（{pct(o['lateral'])} 在侧向 padding）。</li>
</ol>
<p>2 km 实验把这三个因素分开了：只改深度加权（<code>base_dw1</code>，α<sub>s</sub> 仍为 10⁻⁴）时 10 km 以下仍有 {pct(LOW['base_dw1']['below'])}；只把 p<sub>z</sub> 改为 2（<code>as1e-4_p0222</code>）也是 {pct(LOW['as1e-4_p0222']['below'])}；只把 α<sub>s</sub> 改为 1 降到 {pct(LOW['as1']['below'])}；α<sub>s</sub> = 1 再加深度加权降到 {pct(LOW['as1_p0222_dw1']['below'])}。第一个原因是主要的。流程的默认 α<sub>s</sub> 已改为 1。</p>
</div>

<h2><span class="no">6</span>与已发表地质资料的对照</h2>
<div class="added">2026 年 10 月 1 日补充 · 文献对照</div>
<div class="prose">
<p>本报告的反演没有参考地质图。本节把其中可靠的结果，即高密度带的位置，与该区已发表的资料对照，并把第 2.3 节的深度与唯一一个已发表的估计相比较。</p>
<ul class="plain">
  <li><b>高密度带就是 Sandur 片岩带。</b>地质图上，Sandur 片岩带是一条 NW–SE 向的绿岩带，长约 60 km，中部宽可达 18 km，四周为花岗岩{cite_zh('MM93', 'GSI')}。图 {fig.ref('data')} 中剩余异常高值带的长度、宽度和走向与之相同，它的西北端（西北高值，{xy(NW_HIGH, '，')}）指向 Hosapete（Hospet），地质图上片岩带正是向那里延伸{cite_zh('IJERT')}。</li>
  <li><b>铁矿山都在这条带上。</b>Kumaraswamy、Donimalai、Ramandurg 和 NEB Range 开采的是片岩带中的铁建造{cite_zh('GSI', 'MEAI')}，它们都位于剩余异常高值带上或其边缘（图 {fig.ref('localities')}）。主布格高值（{xy(MAIN_HIGH, '，')}）在 Donimalai 以西 {G['main_high_to']['donimalai_west_km']:.0f} km、Kumaraswamy 矿区以北约 {half(G['main_high_to']['kumaraswamy_north_km'])} km，位于片岩带东南部，也就是已发表资源量集中的地方：整条片岩带约有 1,876 Mt 铁矿石，品位约 63% Fe{cite_zh('GSI')}。</li>
  <li><b>密度相符。</b>Maurya 等在片岩带的镁铁质和超镁铁质岩石上测得 2.86–3.56 g/cc 的密度，并发现强重力异常位于片岩带之上，弱异常位于花岗岩之上{cite_zh('MGR23')}。这与联合反演报告中用作上下界的岩样一致（变玄武岩 2.94–2.98 g/cc，铁建造 3.39 g/cc，花岗岩约 2.63 g/cc）。</li>
  <li><b>低密度带是花岗岩。</b>高密度带以东的几条低密度带，位于较年轻的花岗岩（2.5–2.6 Ga）叠覆在片岩带东缘之上的地方{cite_zh('MM93')}。</li>
  <li><b>深度相容，但没有得到证实。</b>Maurya 等根据重磁联合解释，认为片岩带是一个深约 6 km 的盆地{cite_zh('MGR23')}。这个深度落在本报告的范围之内：主体（半峰值）的底界在 β = 0.5 时为 {t['as1_beta0.5']['main']['bottom_km']:.1f} km，β = 1 时为 {t['as1_beta1']['main']['bottom_km']:.1f} km，β = 1.5 时为 {t['as1_beta1.5']['main']['bottom_km']:.1f} km。他们的深度同样依赖假定的密度差，摘要中也没有给出剖面，所以这种一致只能说明第 7 节建议的 β 范围是合理的，不能说明哪个 β 是对的。有约束的联合反演把范围缩小了（底界约 5.5 km，见联合反演报告第 5 节）。</li>
  <li><b>重力分辨不了带内的形态。</b>片岩带传统上被解释为一个复向斜；一项构造研究则认为，它是中央变火山岩带两侧的两条变沉积岩带，两端都不相连，层理陡立，褶皱倾伏约 45°{cite_zh('MM93')}。本报告每个模型中都只有一个高密度体，与两种解释都相容。不过地层层序给出了一点倾向。铁建造位于层序的顶部：西部层序自下而上从变玄武岩到铁建造{cite_zh('MM93')}，属于 Bababudan 型组合，即铁建造覆于镁铁质火山岩和石英岩之上{cite_zh('GSI')}。经过陡倾的近等斜褶皱（D1），铁建造在片岩带两侧边缘表现为近于直立的薄层，厚度不足以主导重力异常，因此模型中的高密度体更可能是中央的变火山岩。这与两翼的解释更吻合；如果是简单的向形，核部应当是层位最新的铁建造。</li>
</ul>
</div>
{fig('localities', f"已发表的地点叠加在布格异常（左）和反演用的剩余异常（右）上。白色方块：铁矿山（Kumaraswamy 旁的两个小矩形是它的 B、C 开发区块）；黄色星号：金矿点；虚线框：Mincheri 铜矿普查区块（在测区边缘被截断）；黑点：城镇。来源：{cite_zh(*LOC_REFS)}。")}
<div class="prose">
<p>图 {fig.ref('geology_comparison')} 把模型与由 DEM 绘制的片岩带示意图对照。用同样的方法沿主轴量测，示意图中主片岩带的轮廓长 {BELT['length_km']:.0f} km、宽 {BELT['width_km']:.0f} km，走向 {BELT['strike_deg']:.0f}°；剩余异常超过其最大值一半（{RH['threshold_mgal']:.1f} mGal）的部分长 {RH['length_km']:.0f} km、宽 {RH['width_km']:.0f} km，走向 {RH['strike_deg']:.0f}°（超过 5 mGal 的部分长 {R5['length_km']:.0f} km，与地质图上的长度相当）。去掉测区边缘 {G['edge_km']:.0f} km 宽的条带后，参考模型（{label(G['run'], 'zh')}）垂向积分密度超过其最大值一半的单元列中，有 {V['dense_ref']} 位于轮廓之内（六个模型为 {V['dense_all']}），而轮廓只占这部分面积的 {V['outline_share']}。在轮廓内，参考模型的积分密度在山脊以外平均为 {G['integ_off_ridges']:.1f} g/cc·km，在山脊上为 {G['integ_on_ridges']:.1f} g/cc·km；但单元为 1 km，山脊又位于轮廓边缘，任何位于带中央的密度体都会给出这样的结果，所以这只说明与上面的解释相容，并不是对它的检验。</p>
</div>
{fig('geology_comparison', f"(a) 本报告由输入数据中的 450 m DEM 绘制的片岩带示意图，叠加图 {fig.ref('localities')} 中的地点。山脊（深绿）是地面比 12 km 窗口内的中值高出 {RIDGE_RELIEF_M:.0f} m 以上（且海拔高于 {RIDGE_MIN_M:.0f} m）的地方；铁建造支撑着片岩带的山丘{cite_zh('MEAI')}，所以把它们看作铁建造山脊。片岩带轮廓（浅绿）是山脊的包络，西、东两条山脉之间的中央变火山岩带按 Mukhopadhyay 和 Matin{cite_zh('MM93', text='1993')}标注。图中英文标注：western (Sandur) range、eastern range 为西、东两条山脉，central metavolcanic terrane 为中央变火山岩带，north-eastern belt 为东北的片岩带，granite and gneiss 为花岗岩和片麻岩。这不是地质图：地质图上的片岩带（约 60 km，向西北延伸到 Hosapete 方向）比这个包络（{BELT['length_km']:.0f} × {BELT['width_km']:.0f} km）长。(b) 反演用的剩余异常（同图 {fig.ref('data')}），叠加 (a) 中的轮廓（实线）和山脊（点线）。(c) 参考模型 {label(G['run'], 'zh')} 的垂向积分密度（同图 {fig.ref('robust')} 上排），叠加同一轮廓。符号同图 {fig.ref('localities')}。")}
<div class="callout"><b>对本报告而言。</b>重力结果中可靠的部分，即高密度带的位置和走向，与地质图上的绿岩带及其铁矿区一致。第 2.3 节的深度范围包含了唯一一个已发表的估计（约 6 km），但并没有被它证实。</div>

<h2><span class="no">7</span>建议</h2>
<div class="prose">
<ol class="steps">
  <li><b>默认设置：</b>sparse，α<sub>s</sub> = 1，p = [0,2,2,2] 或 [0,2,2,1]，深度加权 <code>depth</code>，β 在 0.5–1 之间。把 β = 0.5 和 β = 1 两个模型作为深度范围一起给出（主体质心约 {cen('as1_beta0.5'):.1f}–{cen('as1_beta1'):.1f} km），L1–L2 作为独立的检查。</li>
  <li><b>先粗后细：</b>先在 2 km 网格上筛选参数，再在 1 km 网格上确认。上传页面的参数扫描可以一次提交多个 β，并在同一个 workflow 里比较。</li>
  <li><b>解释时分清可靠和不可靠的部分：</b>横向位置、走向和相对强弱可以直接解释；深度、厚度和密度差需要独立约束。</li>
  <li><b>用独立资料约束深度：</b>岩石密度、地质剖面、钻孔或地震资料；也可以做一个与本区相似的重力合成模型，检查哪个 β 能恢复已知的深度。</li>
  <li><b>数据：</b>去除孤立的离群点或使用稳健的拟合函数，并检查 0.5 mGal 的噪声下限是否合适。</li>
  <li><b>Tomofast-x：</b>只在 SimPEG 抽稀后仍超出内存，或需要重磁联合反演时考虑。如果使用，参数试算要用完整长度的运行（不要用缩短的代替），并检查恢复模型的深度剖面，不能只看 χ。</li>
</ol>
<h3>文件</h3>
<ul class="plain note">
  <li>本报告和所有图：<code>examples/output/karnataka_gravity/scripts/</code>（<code>make_figures.py</code>、<code>build_reports.py</code>）</li>
  <li>六组全分辨率结果：<code>data/ec2_runs/</code>；2 km 实验：<code>data/lowres_runs/</code></li>
  <li>所有运行的交互式 workflow（DAG 查看器）：<code>depth_study/karnataka_depth_study.geoinv3d_viewer.html</code></li>
</ul>
</div>

{refs_zh()}
<footer>GeoInv3D · SimPEG 0.25.2 · 由 build_reports.py 生成，所有数字来自上述运行，第 6 节中引自文献的数值除外，其来源见该节的引用。</footer>
"""


# ---------------------------------------------------------------- English

def body_en(fig):
    o, t = F["original_sparse"], F
    return f"""
<header>
  <div class="eyebrow">GeoInv3D · field-data test · 28 September 2026 · literature cross-check 1 October 2026</div>
  <h1>Karnataka Gravity Inversion: Comparing Regularizations</h1>
  <p class="lede">The same Bouguer gravity data on the same mesh, inverted at full resolution on AWS EC2 with six regularization settings. This report compares their data fit, lateral structure and depth, cross-checks them against the 24 September SimPEG run on an 8 km deep mesh and against Tomofast-x, and tests whether a coarse 2 km study can stand in for full-resolution runs.</p>
  <dl class="meta">
    <div><dt>Area</dt><dd>70 × 70 km, easting 641–711 km, northing 1634–1704 km (UTM 43N, EPSG:32643)</dd></div>
    <div><dt>Data</dt><dd>NGPM Bouguer anomaly, 500 m grid decimated to 1 km: {D['n']:,} points; second-order trend surface removed</dd></div>
    <div><dt>Mesh</dt><dd>1 km × 1 km × 500 m core, 10 km deep; about 20 km padding at the sides and bottom (factor 1.3): 84 × 84 × 27 = 190,512 cells</dd></div>
    <div><dt>Runs</dt><dd>6 full-resolution inversions on AWS EC2 (ap-south-1), 2 of them submitted from the upload page; a 16-run 2 km study run locally</dd></div>
  </dl>
</header>

<section class="summary" aria-labelledby="sum">
  <h2 id="sum">Summary</h2>
  <ul>
    <li><b>All six runs fit the data about equally well.</b> χ²/N {V['chi']}, residual RMS {V['rms']} mGal. The fit alone cannot choose between these models.</li>
    <li><b>The lateral structure is robust.</b> The vertically integrated density maps correlate at {V['corr_min']}–{V['corr_max']} pairwise, and at no less than {V['corr_new']} without the original settings. The NW–SE dense belt and the low-density belts to the east have the same position and strike in every model.</li>
    <li><b>The regularization sets the depth.</b> Centroid depth under the main Bouguer high: {cen('as1_beta0.5'):.1f} km for β = 0.5, {cen('l1l2_irls'):.1f} km for L1–L2, {cen('as1_beta1'):.1f} km for β = 1, {cen('as0.1_beta1'):.1f} km for α<sub>s</sub> = 0.1 and {cen('as1_beta1.5'):.1f} km for β = 1.5. The original settings give a column from the surface to the bottom of the mesh ({o['main']['bottom_km']:.1f} km). A larger β gives a deeper model with no matching change in χ²/N.</li>
    <li><b>A different set-up gives the same depth.</b> The 24 September SimPEG run on an 8 km deep mesh (α<sub>s</sub> = 0.05, sensitivity weighting) put the main body at {rng(S8['main'])} km (half maximum), centroid {S8['main']['centroid_km']:.1f} km; this report's β = 1 run gives {rng(t['as1_beta1']['main'])} km and {cen('as1_beta1'):.1f} km. Tomofast-x fitted the data more closely (chi 0.65) but its model has no base, and it took about 2.7 hours against SimPEG's {V['time8']:.0f} minutes.</li>
    <li><b>The coarse 2 km study is good for screening.</b> For three values of β it predicted the full-resolution centroid of the main body to within {V['lowgap']:.1f} km; for α<sub>s</sub> = 0.1 it was {V['as01gap']:.1f} km off. A coarse run takes about 25 s, a full-resolution run about 5 minutes.</li>
    <li><b>Recommendation.</b> α<sub>s</sub> = 1 (now the pipeline default) with Li &amp; Oldenburg depth weighting, β between 0.5 and 1, reporting a range of depths rather than a single model: β = 0.5 agrees with L1–L2, β = 1 with the 8 km deep mesh.</li>
    <li><b>Against the published geology</b> (Section 6). The dense belt coincides with the mapped Sandur schist belt, and the four iron-ore mines of the belt lie on it. The depth range brackets the one published estimate, a basin about 6 km deep {cite('MGR23')}.</li>
    <li><b>Caveat.</b> Gravity data alone do not determine depth; the depths in this report are a consequence of the regularization. The actual depths need constraints from rock densities, geological sections or seismic data.</li>
  </ul>
</section>

<h2><span class="no">1</span>Data and the runs compared</h2>
<div class="prose">
<p>Within the area the Bouguer anomaly ranges from <span class="num">{mgal(D['raw'][0])}</span> to <span class="num">{mgal(D['raw'][1])}</span> mGal. Its main feature is a NW–SE trending high from (650, 1685) to (675, 1660) km. After removing a least-squares second-order trend surface (the regional field, <span class="num">{mgal(D['regional'][0])}</span> to <span class="num">{mgal(D['regional'][1])}</span> mGal), the residual anomaly ranges from <span class="num">{mgal(D['residual'][0])}</span> to <span class="num">+{D['residual'][1]:.1f}</span> mGal; this is what was inverted (Figure {fig.ref('data')}). Every point was given an error of 0.5 mGal. Bouguer data are positive downward and SimPEG's g<sub>z</sub> is positive upward, so the pipeline multiplies the data by −1 before the inversion and returns the observed and predicted data in the original convention.</p>
</div>
{fig('data', "Left: Bouguer anomaly in the area (decimated to 1 km). Centre: least-squares second-order trend surface, removed as the regional field. Right: the residual anomaly, i.e. the data inverted (red is positive, dense).")}
<div class="prose">
<p>The six runs share the mesh, the data and the errors; only the regularization differs:</p>
</div>
{runs_table('en')}
<p class="note">Common settings: density contrast bounded to −0.2 to +0.5 g/cc (rock samples 2.52–3.40 g/cc, median 2.66); β chosen automatically for χ² = N; at most 30 iterations and 30 IRLS iterations. α<sub>x</sub> = α<sub>y</sub> = α<sub>z</sub> = 1, passed to SimPEG as length scales. Li &amp; Oldenburg depth weighting weights the model norm by (z + z₀)<sup>−β/2</sup>, where z is the depth below the nearest station and z₀ half the smallest cell.</p>

<h2><span class="no">2</span>Comparing the full-resolution results</h2>

<h3><span class="no">2.1</span>Data fit</h3>
<div class="prose">
<p>All six runs reached the target fit, with nearly the same statistics. The residual maps (Figure {fig.ref('residuals')}) differ, though:</p>
<ul class="plain">
  <li><b>Original settings:</b> the residuals cluster at a few places with steep gradients, as patches of positive and negative misfit up to {o['max_abs']:.1f} mGal. A model made of columns struggles to fit adjacent narrow anomalies.</li>
  <li><b>L1–L2:</b> a broad positive residual over the main high, negative around it: the model underestimates the amplitude of the main anomaly (its main body is at the +0.5 g/cc bound). Its χ²/N is also the highest ({t['l1l2_irls']['chi2']:.2f}).</li>
  <li><b>The four depth-weighted runs:</b> the least structured residuals, mostly short-wavelength stripes along strike and a few isolated points. The isolated points near (660, 1680) and (702, 1659) km appear in every model and are probably outliers from the original gridding.</li>
</ul>
</div>
{fit_table('en')}
{fig('residuals', "Residuals (observed − predicted) of the six full-resolution models; colour scale ±3 mGal.")}

<h3><span class="no">2.2</span>Lateral structure</h3>
<div class="prose">
<p>Figure {fig.ref('slices')} shows five models at 1.8, 4.8 and 7.8 km depth. The top row of Figure {fig.ref('robust')} shows the vertically integrated density (the sum of density contrast × thickness in each column), the bottom row the centroid depth of each column.</p>
<ul class="plain">
  <li><b>The integrated density maps are nearly the same.</b> Pairwise correlations are {V['corr_min']}–{V['corr_max']}, and no less than {V['corr_new']} without the original settings (table below). The NW–SE dense belt, the secondary high at its north-western end and the parallel low-density belts to the east have the same position and strike in every model.</li>
  <li><b>What differs is the amplitude and sharpness.</b> A larger β gives stronger integrated density and sharper edges; the original settings give the smoothest model.</li>
  <li><b>The centroid depth maps are completely different.</b> For the same dense belt the centroid lies at 2–4 km for β = 0.5 and L1–L2, 4–6 km for β = 1, 6–8 km for β = 1.5 (locally below 12 km), and deeper than 10 km for the original settings.</li>
  <li><b>The slices show the same thing.</b> For β = 0.5 there is no anomaly left at 7.8 km; for β = 1.5 only scattered fragments remain at 1.8 km; the original model is almost the same at all three depths.</li>
</ul>
</div>
{corr_table('en')}
<p class="note">Correlation between the vertically integrated density maps in the 70 × 70 km core. Green: ≥ 0.90.</p>
{fig('slices', "Depth slices of five models (core area only); colour scale ±0.3 g/cc.")}
{fig('robust', "Top: vertically integrated density (g/cc·km); the five maps are essentially the same. Bottom: centroid depth of |density contrast| in each column (only columns whose integral exceeds 20% of the maximum); it changes strongly with the regularization.")}

<h3><span class="no">2.3</span>Depth</h3>
<div class="prose">
<p>Figure {fig.ref('sections')} shows an E–W section through the main Bouguer high (northing 1664.5 km), and the left panel of Figure {fig.ref('mass_profiles')} the distribution of |mass| with depth below the core area. The table gives where the mass lies, and measures of the density profiles under the main Bouguer high (easting 671.5 km, northing 1664.5 km) and the north-western high (650.5, 1682.5 km).</p>
<p>Depth is given as the <b>centroid</b> (the mean depth of the positive density contrast, weighted by thickness) rather than the depth of the peak: with α<sub>s</sub> = 1 the main body reaches the +0.5 g/cc bound, so the peak is a plateau with no single depth. The half-maximum range is the depth interval where the density exceeds half the peak.</p>
</div>
{depth_table('en')}
<p class="note">Core, lateral padding and below 10 km add up to 100%. D50, D90: half and 90% of the |mass| below the core area lie above this depth. Red: lateral padding &gt; 20%, below 10 km &gt; 30%, D50 &gt; 7 km, or the main body reaching below 12 km. The 8 km deep mesh is a different mesh, so only the profile measures are comparable.</p>
{fig('sections', "E–W sections through the main Bouguer high (northing 1664.5 km) in the six models. The vertical dotted line marks the high, the horizontal dashed line the base of the core mesh (10 km); below it is padding. Colour scale ±0.3 g/cc.")}
<div class="prose">
<ul class="plain">
  <li><b>The original settings give columns.</b> Under the main high the density contrast is only {o['main']['peak']:.2f} g/cc and extends from {o['main']['top_km']:.1f} to {o['main']['bottom_km']:.1f} km, centroid {cen('original_sparse'):.1f} km; {pct(o['below'])} of the mass lies below 10 km and {pct(o['lateral'])} in the lateral padding.</li>
  <li><b>The other five have a base.</b> The main body ends above 12 km, and at least {V['core_min']} of the mass lies in the core ({pct(t['as1_beta1.5']['core'])} for β = 1.5).</li>
  <li><b>β controls the depth monotonically.</b> For β = 0.5, 1 and 1.5 the centroid of the main body is at {cen('as1_beta0.5'):.1f}, {cen('as1_beta1'):.1f} and {cen('as1_beta1.5'):.1f} km, the half-maximum range {rng(t['as1_beta0.5']['main'])}, {rng(t['as1_beta1']['main'])} and {rng(t['as1_beta1.5']['main'])} km, while χ²/N is {t['as1_beta0.5']['chi2']:.2f}, {t['as1_beta1']['chi2']:.2f} and {t['as1_beta1.5']['chi2']:.2f}, unrelated to depth. This is the non-uniqueness of gravity inversion: the data cannot choose β.</li>
  <li><b>L1–L2 is nearly the same as β = 0.5.</b> Two quite different regularizations give the same half-maximum range ({rng(t['l1l2_irls']['main'])} km), with centroids {abs(cen('l1l2_irls') - cen('as1_beta0.5')):.1f} km apart.</li>
  <li><b>α<sub>s</sub> = 0.1 lies between β = 1 and β = 1.5.</b> Centroid {cen('as0.1_beta1'):.1f} km, peak {t['as0.1_beta1']['main']['peak']:.2f} g/cc, close to but not at the bound. The model is smoother than with α<sub>s</sub> = 1; the north-western high extends from {t['as0.1_beta1']['nw']['top_km']:.1f} to {t['as0.1_beta1']['nw']['bottom_km']:.1f} km.</li>
  <li><b>β = 1.5 starts to trail to depth.</b> The low-density body in the east reaches 20 km (Figure {fig.ref('sections')}), and {pct(t['as1_beta1.5']['below'])} of the mass lies below 10 km.</li>
  <li><b>The north-western high is always shallower.</b> Apart from the original settings, its centroid lies at {V['nw_new']} km, {V['nw_gap']} km above that of the main high. This relation holds in every model.</li>
</ul>
</div>
{fig('mass_profiles', "Distribution of |mass| with depth below the core area (percent per km). Left: the six full-resolution models. Right: five runs of the 2 km study comparing α<sub>s</sub> and depth weighting (Section 4). The dashed line is the base of the core mesh.")}
<div class="prose">
<p>The radially averaged power spectrum of the residual anomaly (Figure {fig.ref('spectrum')}) has three straight segments, giving mean depths to the top of the sources of {SP['deep']:.1f}, {SP['mid']:.1f} and {SP['shallow']:.1f} km (Spector &amp; Grant). It shows sources at several depths but cannot choose between the models: the top of the main body for β = 0.5 and L1–L2 ({t['as1_beta0.5']['main']['top_km']:.1f} km) is near the intermediate estimate, the centroid for β = 1 ({cen('as1_beta1'):.1f} km) near the deep one. Spectral depths are averages over an ensemble of sources and are very uncertain.</p>
</div>
{fig('spectrum', "Radially averaged power spectrum of the residual anomaly (500 m grid, second-order trend removed). The slopes of the three segments give the mean depth to the top of the sources, h = −slope / 4π.", narrow=True)}

<h3><span class="no">2.4</span>What is robust</h3>
<div class="prose">
<p><b>The same under every setting:</b></p>
<ul class="plain">
  <li>The lateral position, strike and extent of the anomalous bodies: the NW–SE dense belt and the low-density belts to the east.</li>
  <li>Sign and relative strength: the strongest positive body lies under the main Bouguer high.</li>
  <li>The main body has a base (in the five models other than the original settings, and in the independent 8 km deep mesh run); the north-western high is shallower than the main high.</li>
</ul>
<p><b>Changes with the regularization:</b></p>
<ul class="plain">
  <li>Absolute depth: the centroid of the main body lies at {cen('as1_beta0.5'):.1f}–{cen('as1_beta1.5'):.1f} km depending on β.</li>
  <li>Thickness and density contrast: with α<sub>s</sub> = 1 the main bodies reach the +0.5 / −0.2 g/cc bounds. This blocky, saturated model is a product of the regularization, not a measurement of density.</li>
  <li>Deep details, such as the low-density body in the east reaching 20 km for β = 1.5.</li>
</ul>
<p>χ²/N ({V['chi']}) cannot be used to choose between these models.</p>
</div>

<h2><span class="no">3</span>Cross-check: SimPEG on an 8 km deep mesh, and Tomofast-x</h2>
<div class="prose">
<p>On 24 September the same Karnataka gravity data were inverted with two independent codes: SimPEG, used throughout this project, and Tomofast-x 2.0, an open-source parallel code. Both used the same 8 km deep mesh (500 m horizontal and 250 m vertical cells), the same cloud machine (16 vCPU, 123 GB memory) and the same noise model. Only the gravity part is summarised here; for magnetics see the original report (Desktop/Tomofastx_vs_SimPEG_package).</p>
</div>
<div class="tablewrap"><table>
<thead><tr><th></th><th>SimPEG</th><th>Tomofast-x 2.0</th></tr></thead>
<tbody>
<tr><td>Data fit, chi</td><td class="n">{S8['chi']:.2f}</td><td class="n">0.65</td></tr>
<tr><td>Data used</td><td>{S8['n_data']:,} points (every second node)</td><td>19,600 points (all)</td></tr>
<tr><td>Under the main Bouguer high</td><td class="wrap">{rng(S8['main'])} km at half maximum, centroid {S8['main']['centroid_km']:.1f} km, peak {S8['main']['peak']:.2f} g/cc; {S8['at_9km']:.2f} g/cc at 9.2 km, while the mesh reaches {S8['deepest_km']:.1f} km</td><td class="wrap">No base: the 1st, 50th and 99th percentiles of each layer change by less than 10% from the surface to 8 km</td></tr>
<tr><td>Regularization</td><td class="wrap">Chosen once: sparse, norms (0, 2, 2, 1), α<sub>s</sub> = 0.05, sensitivity-based depth weighting</td><td class="wrap">24-trial search of two smoothing weights; still no shape</td></tr>
<tr><td>Time to the final model</td><td>{V['time8']:.0f} minutes ({S8['time_s']:.0f} s)</td><td>About 2.7 hours (1.7 h search + 1.0 h full run)</td></tr>
</tbody></table></div>
{fig('tomofastx', "E–W and N–S sections through the anomaly centre (gravity only, from the original report). Left: SimPEG, a body closed at both ends. Right: Tomofast-x, vertical stripes from the surface to the mesh floor. Note the different colour scales.")}
{fig('centre_profiles', "Density with depth under the main Bouguer high. Red: SimPEG on the 8 km deep mesh (24 September, no bounds); the others are this report's six full-resolution models. The dashed line is the deepest cell of the 8 km mesh.", narrow=True)}
<div class="prose">
<h3>How it relates to this report</h3>
<ul class="plain">
  <li><b>Why the 24 September SimPEG model has a base.</b> It used α<sub>s</sub> = 0.05. With length-scale alphas the smallness term is then about a fifth of the gradient terms, so the compactness constraint acts; the original settings in this report used α<sub>s</sub> = 10⁻⁴, about 2500 times weaker, and gave columns (Section 5).</li>
  <li><b>Two different set-ups give the same depth.</b> The 8 km mesh (500 m cells, sensitivity weighting, {S8['n_data']:,} data) gives {rng(S8['main'])} km with the centroid at {S8['main']['centroid_km']:.1f} km; the 1 km tensor mesh with α<sub>s</sub> = 1 and β = 1 ({D['n']:,} data) gives {rng(t['as1_beta1']['main'])} km and {cen('as1_beta1'):.1f} km (Figure {fig.ref('centre_profiles')}). The north-western high agrees too (centroid {S8['nw']['centroid_km']:.1f} against {cen('as1_beta1', 'nw'):.1f} km). The result is therefore not sensitive to the mesh or the thinning, but both are the same kind of depth-weighted compact SimPEG model and do not remove the non-uniqueness: β = 0.5 and L1–L2 are shallower, β = 1.5 deeper, and the data cannot tell them apart.</li>
  <li><b>A good chi is not evidence of a good model.</b> Tomofast-x reached chi 0.65, below the target, with no vertical shape at all; this report's β sweep also shows models with nearly the same chi whose depths differ by a factor of two. A settings search that looks only at chi selects physically meaningless settings. A published Tomofast-x application (Bird et al., 2025, Antarctic bathymetry) reached the same conclusion and ran 72 full-length trial inversions instead.</li>
</ul>
<h3>Cost (gravity only)</h3>
</div>
<div class="tablewrap"><table class="compact">
<thead><tr><th>Stage</th><th class="n">Approx. time</th></tr></thead>
<tbody>
<tr><td>SimPEG gravity, final run (8 km mesh, 16 vCPU)</td><td class="n">{V['time8']:.0f} minutes ({S8['time_s']:.0f} s)</td></tr>
<tr><td>Tomofast-x gravity: 24-trial settings search</td><td class="n">1.7 hours</td></tr>
<tr><td>Tomofast-x gravity: full-length run</td><td class="n">1.0 hour</td></tr>
<tr><td>SimPEG (this report, EC2 c5.2xlarge, 1 km mesh)</td><td class="n">5–6 minutes per run, about $0.03</td></tr>
</tbody></table></div>
<div class="prose">
<p>Most of Tomofast-x's cost comes from one feature: it does not tune its own regularization, so its two smoothing weights must be found before a run by repeated trials, each 12–17 minutes on the full problem. SimPEG adjusts β during the run and needs no search.</p>
<h3>What this does not show</h3>
<ul class="plain">
  <li><b>SimPEG used less data:</b> {S8['n_data']:,} points against Tomofast-x's 19,600. Tomofast-x handles the full data volume with a wavelet-compressed sensitivity matrix, a real strength this comparison does not credit.</li>
  <li><b>SimPEG had its own problems:</b> two faults in the workflow at the time (a mesh-depth setting that had no effect, a missing solver speed-up) took hours to find. They are fixed, but the time is not in the table.</li>
  <li><b>Tomofast-x can converge here:</b> on a shallower 3 km mesh it reached an on-target fit (gravity chi 1.87) with a search. The failure is in producing a bounded body at depth, not in fitting the data.</li>
  <li><b>We may not have used Tomofast-x well:</b> its published applications also use bound constraints and other controls that we did not try. Its smoothness term cannot be made "blocky" as SimPEG's can, our best explanation for the copied-downward pattern, but that is a hypothesis and was not tested further.</li>
  <li><b>One dataset, one operator:</b> a single survey handled by one person new to Tomofast-x.</li>
</ul>
</div>

<h2><span class="no">4</span>The 2 km study against the full-resolution runs</h2>
<div class="prose">
<p>Before the full-resolution runs, 16 sparse inversions were run locally on a 2 km × 2 km × 1 km mesh (every fourth point, 1,295 data), about 25 s each. The table lists all 16; the right panel of Figure {fig.ref('mass_profiles')} shows the depth distribution of five of them.</p>
</div>
{lowres_table('en')}
<p class="note">"sensitivity" is SimPEG's sensitivity weighting, "β = …" Li &amp; Oldenburg depth weighting. Red cells as in Section 2.3.</p>
<div class="prose">
<h3>Trends on the coarse mesh</h3>
<ul class="plain">
  <li><b>α<sub>s</sub> decides whether the model reaches the bottom of the mesh.</b> With sensitivity weighting, α<sub>s</sub> = 10⁻⁴, 10⁻², 0.1 and 1 leave {pct(LOW['base']['below'])}, {pct(LOW['as1e-2']['below'])}, {pct(LOW['as0.1']['below'])} and {pct(LOW['as1']['below'])} of the mass below 10 km.</li>
  <li><b>Depth weighting decides the lateral leakage.</b> With α<sub>s</sub> = 1, sensitivity weighting puts {pct(LOW['as1_p0222']['lateral'])} of the mass in the lateral padding, depth weighting only {V['dw_lat']}.</li>
  <li><b>β sets the mean depth.</b> β = 0.5, 1, 1.5 and 2 give D50 = {LOW['as1_p0222_dw0.5']['D50']:.1f}, {LOW['as1_p0222_dw1']['D50']:.1f}, {LOW['as1_p0222_dw1.5']['D50']:.1f} and {LOW['as1_p0222_dw2']['D50']:.1f} km and main-body centroids of {lc('as1_p0222_dw0.5'):.1f}, {lc('as1_p0222_dw1'):.1f}, {lc('as1_p0222_dw1.5'):.1f} and {lc('as1_p0222_dw2'):.1f} km. The β = 2 often used for gravity is the deepest here, with {pct(LOW['as1_p0222_dw2']['below'])} of the mass below 10 km.</li>
  <li><b>The norms matter less.</b> With α<sub>s</sub> = 1 and β = 1, p = [0,2,2,2], [0,2,2,1] and [0,1,1,1] give main-body centroids of {lc('as1_p0222_dw1'):.1f}, {lc('as1_p0221_dw1'):.1f} and {lc('as1_p0111_dw1'):.1f} km.</li>
</ul>
<h3>Against the 1 km runs</h3>
<p>At the same place (Figure {fig.ref('lowres_vs_full')}), for the three values of β the coarse centroid is within {V['lowgap']:.1f} km of the 1 km result and the base within 0.5 km; the top is shallower on the coarse mesh ({LOW['as1_p0222_dw0.5']['main']['top_km']:.1f} against {t['as1_beta0.5']['main']['top_km']:.1f} km for β = 0.5) because its cells are 1 km thick. α<sub>s</sub> = 0.1 is the exception: on the coarse mesh the body does not saturate (peak {LOW['as0.1_p0222_dw1']['main']['peak']:.2f} g/cc), spans {rng(LOW['as0.1_p0222_dw1']['main'])} km, and its centroid is {V['as01gap']:.1f} km shallower than at 1 km. The coarse mesh is good for screening β and comparing trends; the final result should still be confirmed at full resolution.</p>
</div>
{fig('lowres_vs_full', "Half-maximum depth range (bars) and centroid (dots) under the main Bouguer high: the 2 km study (grey) and the 1 km full-resolution runs (blue).", narrow=True)}

<h2><span class="no">5</span>Why the original settings give columns</h2>
<div class="prose">
<p>The original settings were sparse (IRLS), p = [0, 2, 2, 1], α<sub>s</sub> = 10⁻⁴, α<sub>x</sub> = α<sub>y</sub> = α<sub>z</sub> = 1 (length scales), with sensitivity weighting. There are three reasons:</p>
<ol class="steps">
  <li><b>The smallness term does almost nothing.</b> The pipeline passes α<sub>x,y,z</sub> to SimPEG as length scales, so the gradient terms are weighted by (1 × 500 m)², about 0.25 Δm² per cell, some 2500 times stronger than the 10⁻⁴ m² of the smallness term; the compactness constraint p<sub>s</sub> = 0 therefore has almost no effect.</li>
  <li><b>L1 on the vertical gradient.</b> p<sub>z</sub> = 1 favours models that are piecewise constant with depth, so a column that does not change with depth costs almost nothing.</li>
  <li><b>Sensitivity weighting makes the padding cheap.</b> Cells outside the area and at depth have low sensitivity and therefore low weights, so mass placed there costs little ({pct(o['lateral'])} went into the lateral padding).</li>
</ol>
<p>The 2 km study separates the three: changing only the depth weighting (<code>base_dw1</code>, α<sub>s</sub> still 10⁻⁴) leaves {pct(LOW['base_dw1']['below'])} of the mass below 10 km; changing only p<sub>z</sub> to 2 (<code>as1e-4_p0222</code>) leaves {pct(LOW['as1e-4_p0222']['below'])}; changing only α<sub>s</sub> to 1 brings it down to {pct(LOW['as1']['below'])}, and α<sub>s</sub> = 1 with depth weighting to {pct(LOW['as1_p0222_dw1']['below'])}. The first reason dominates. The pipeline default is now α<sub>s</sub> = 1.</p>
</div>

<h2><span class="no">6</span>Comparison with the published geology</h2>
<div class="added">Added 1 October 2026 · literature cross-check</div>
<div class="prose">
<p>The inversions of this report were run without reference to the geological map. This section compares their robust result, the position of the dense belt, with what is published about the area, and sets the depths of Section 2.3 against the one published estimate.</p>
<ul class="plain">
  <li><b>The dense belt is the Sandur schist belt.</b> The belt is mapped as a NW–SE greenstone belt about 60 km long and up to 18 km wide in its centre, enclosed by granite {cite('MM93', 'GSI')}. The residual high of Figure {fig.ref('data')} has the same length, width and strike, and its north-western end (the north-western high, {xy(NW_HIGH)}) points at Hosapete (Hospet), where the belt is mapped to continue {cite('IJERT')}.</li>
  <li><b>The iron-ore mines sit on it.</b> Kumaraswamy, Donimalai, Ramandurg and NEB Range, which work the iron formation of the belt {cite('GSI', 'MEAI')}, all lie on or at the edge of the residual high (Figure {fig.ref('localities')}). The main Bouguer high ({xy(MAIN_HIGH)}) lies {G['main_high_to']['donimalai_west_km']:.0f} km west of Donimalai and about {half(G['main_high_to']['kumaraswamy_north_km'])} km north of the Kumaraswamy leases, in the south-eastern part of the belt where the published resource is concentrated: about 1,876 Mt of iron ore at about 63% Fe in the belt as a whole {cite('GSI')}.</li>
  <li><b>The densities agree.</b> Maurya et al. measured 2.86 to 3.56 g/cc on the mafic and ultramafic rocks of the belt and found the strong gravity anomalies over the schist belts, the weak ones over the granites {cite('MGR23')}. This matches the samples used as bounds in the joint report (metabasalt 2.94 to 2.98 g/cc, iron formation 3.39 g/cc, granite about 2.63 g/cc).</li>
  <li><b>The light belts are granite.</b> The low-density belts east of the dense belt lie where younger granites (2.5 to 2.6 Ga) override the eastern margin of the belt {cite('MM93')}.</li>
  <li><b>The depth is consistent but not confirmed.</b> From a joint gravity–magnetic interpretation Maurya et al. describe the belt as a basin about 6 km deep {cite('MGR23')}. That lies inside the range of this report: the main body ends at {t['as1_beta0.5']['main']['bottom_km']:.1f} km for β = 0.5, {t['as1_beta1']['main']['bottom_km']:.1f} km for β = 1 and {t['as1_beta1.5']['main']['bottom_km']:.1f} km for β = 1.5 (half maximum). Their depth also rests on an assumed density contrast, and the abstract gives no profiles, so the agreement shows that the β range of Section 7 is reasonable, not which β is right. The constrained joint inversion narrows it (base at about 5.5 km, joint report Section 5).</li>
  <li><b>The shape inside the belt is not resolved by gravity.</b> The belt is traditionally read as a synclinorium; a structural study finds instead two metasedimentary belts on either side of a central metavolcanic terrane that do not join at either end, with steep bedding and folds plunging about 45° {cite('MM93')}. A single dense body, as in every model here, is compatible with either reading. The stratigraphy gives a weak preference, however. The iron formation is a layer at the top of the succession: the western sequence runs from metabasalt at the base to iron formation at the top {cite('MM93')}, a Bababudan-type assemblage of iron formation over mafic volcanics and quartzite {cite('GSI')}. Folded steeply (near-isoclinal D1 folds), it stands as thin, near-vertical sheets on the margins of the belt, too thin to dominate the gravity, so the dense body of the models is more likely the metavolcanic core in the centre. That fits the two-flank reading better than a simple synform, whose youngest unit, the iron formation, would lie in its core.</li>
</ul>
</div>
{fig('localities', f"Published localities on the Bouguer anomaly (left) and the residual anomaly that was inverted (right). {LEGEND} Sources: {cite(*LOC_REFS)}.")}
<div class="prose">
<p>Figure {fig.ref('geology_comparison')} sets the models against a schematic of the belt drawn from the DEM. Measured the same way, along the principal axes, the outline of its main belt is {BELT['length_km']:.0f} km long and {BELT['width_km']:.0f} km wide with a strike of {BELT['strike_deg']:.0f}°, and the residual anomaly above half its maximum ({RH['threshold_mgal']:.1f} mGal) {RH['length_km']:.0f} km, {RH['width_km']:.0f} km and {RH['strike_deg']:.0f}° (above 5 mGal it is {R5['length_km']:.0f} km long, about the mapped length). Leaving out a {G['edge_km']:.0f} km band along the edges of the area, {V['dense_ref']} of the columns where the integrated density of the reference model ({label(G['run'], 'en')}) exceeds half its maximum lie inside the outline ({V['dense_all']} in the six models), which covers {V['outline_share']} of that area. Inside the outline the reference model averages {G['integ_off_ridges']:.1f} g/cc·km off the ridges and {G['integ_on_ridges']:.1f} g/cc·km on them; with 1 km cells and the ridges along the margins, any body centred in the belt would show this, so it is consistent with the reading above, not a test of it.</p>
</div>
{fig('geology_comparison', f"(a) A schematic of the belt drawn for this report from the 450 m DEM of the inputs, with the localities of Figure {fig.ref('localities')}. Ridges (dark green) are where the ground stands more than {RIDGE_RELIEF_M:.0f} m above the median of a 12 km window (and above {RIDGE_MIN_M:.0f} m); they are read as the iron-formation ridges because the iron formation holds up the hills of the belt {cite('MEAI')}. The belt outline (light green) is the envelope of the ridges, and the central metavolcanic terrane between the western and eastern ranges is placed after Mukhopadhyay &amp; Matin {cite('MM93', text='1993')}. It is not a geological map: the mapped belt is longer (about 60 km, continuing NW towards Hosapete) than the envelope ({BELT['length_km']:.0f} × {BELT['width_km']:.0f} km). (b) The residual anomaly that was inverted (as in Figure {fig.ref('data')}), with the outline (solid) and the ridges (dotted) of panel a. (c) The vertically integrated density contrast of the reference model, {label(G['run'], 'en')} (as in the top row of Figure {fig.ref('robust')}), with the same outline. Symbols as in Figure {fig.ref('localities')}.")}
<div class="callout"><b>For this report.</b> The robust part of the gravity result, the position and strike of the dense belt, matches the mapped greenstone belt and its iron-ore district. The depth range of Section 2.3 brackets the one published estimate (about 6 km) but is not confirmed by it.</div>

<h2><span class="no">7</span>Recommendations</h2>
<div class="prose">
<ol class="steps">
  <li><b>Default settings:</b> sparse, α<sub>s</sub> = 1, p = [0,2,2,2] or [0,2,2,1], depth weighting <code>depth</code> with β between 0.5 and 1. Report the β = 0.5 and β = 1 models together as a depth range (main-body centroid about {cen('as1_beta0.5'):.1f}–{cen('as1_beta1'):.1f} km), with L1–L2 as an independent check.</li>
  <li><b>Coarse first, then fine:</b> screen settings on the 2 km mesh, confirm on the 1 km mesh. The upload page's parameter sweep submits several β at once and compares them in one workflow.</li>
  <li><b>Separate the robust from the rest when interpreting:</b> lateral position, strike and relative strength can be interpreted directly; depth, thickness and density contrast need independent constraints.</li>
  <li><b>Constrain depth with independent data:</b> rock densities, geological sections, boreholes or seismic data; or a gravity synthetic model resembling this area, to check which β recovers a known depth.</li>
  <li><b>Data:</b> remove isolated outliers or use a robust misfit, and check whether the 0.5 mGal noise floor is appropriate.</li>
  <li><b>Tomofast-x:</b> worth considering only where SimPEG exceeds memory even after thinning, or for joint gravity–magnetic inversion. If it is used, run full-length trials (not short proxies) and check the recovered depth profile, not chi alone.</li>
</ol>
<h3>Files</h3>
<ul class="plain note">
  <li>This report and all figures: <code>examples/output/karnataka_gravity/scripts/</code> (<code>make_figures.py</code>, <code>build_reports.py</code>)</li>
  <li>The six full-resolution results: <code>data/ec2_runs/</code>; the 2 km study: <code>data/lowres_runs/</code></li>
  <li>All runs in one interactive workflow (DAG viewer): <code>depth_study/karnataka_depth_study.geoinv3d_viewer.html</code></li>
</ul>
</div>

{references_html(REFS)}
<footer>GeoInv3D · SimPEG 0.25.2 · Generated by build_reports.py; every number comes from the runs above, except the published values of Section 6, which come from the sources cited there.</footer>
"""


def find_chrome():
    for p in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chrome")):
        if p and Path(p).exists():
            return p
    return None


def complete(pdf):
    """True once Chrome has finished writing the PDF (its last bytes hold the %%EOF marker)."""
    if not pdf.exists() or pdf.stat().st_size < 1024:
        return False
    with pdf.open("rb") as f:
        f.seek(-1024, 2)
        return b"%%EOF" in f.read()


def main():
    titles = {"zh": "Karnataka 重力反演对比", "en": "Karnataka Gravity Comparison"}
    bodies = {"zh": body_zh, "en": body_en}
    for lang in ("zh", "en"):
        OUT[lang].write_text(page(lang, titles[lang], bodies[lang](Figures(lang))), encoding="utf-8")
        print(OUT[lang], f"{OUT[lang].stat().st_size / 1e6:.2f} MB")
    if "--pdf" in sys.argv:
        chrome = find_chrome()
        if not chrome:
            sys.exit("Chrome not found; the PDF needs headless Chrome")
        pdf = OUT["en"].with_suffix(".pdf")
        pdf.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
            proc = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                                     "--virtual-time-budget=15000", f"--user-data-dir={profile}", f"--print-to-pdf={pdf}",
                                     OUT["en"].resolve().as_uri()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # headless Chrome on macOS can stay alive after printing: wait for a complete PDF, then stop it
            start = time.time()
            while proc.poll() is None and not complete(pdf) and time.time() - start < 180:
                time.sleep(0.5)
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
        if not complete(pdf):
            sys.exit("Chrome did not write the PDF")
        print(pdf, f"{pdf.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
