"""DAG visualization: render the workflow graph.

Two output modes:
  - Matplotlib figure (via networkx) for interactive use
  - Mermaid text for embedding in Markdown / documentation
"""

from __future__ import annotations

from typing import Optional

from ..core.node import Graph, Node


# ── Node styling by type name ──────────────────────────────────────────

_NODE_COLORS = {
    "MeshCreateNode": "#4CAF50",
    "ModelCreateNode": "#2196F3",
    "ModelFromArrayNode": "#2196F3",
    "SurveyCreateNode": "#FF9800",
    "ForwardNode": "#9C27B0",
    "SingleInversionNode": "#F44336",
    "SparseInversionNode": "#E91E63",
    "JointInversionNode": "#D32F2F",
    "RegularizationNode": "#795548",
    "CrossGradientNode": "#607D8B",
    "LogTransformNode": "#00BCD4",
    "ScaleNode": "#00BCD4",
    "OffsetNode": "#00BCD4",
    "ConductivityToResistivityNode": "#00BCD4",
    "ModelExportNode": "#8BC34A",
    "ResultExportNode": "#8BC34A",
}

_DEFAULT_COLOR = "#9E9E9E"


def _color_for(node: Node) -> str:
    return _NODE_COLORS.get(type(node).__name__, _DEFAULT_COLOR)


def _label_for(node: Node) -> str:
    """Short label: name + key params.

    Inversion nodes include model configuration details (norms,
    regularization type, preconditioner, etc.) so different model
    configurations are distinguishable at a glance.
    """
    parts = [node.name]
    p = node.params()
    node_type = type(node).__name__

    if node_type == "SparseInversionNode":
        parts.append(f"method={p.get('method_type', '?')}")
        norms = p.get("norms", [])
        if norms:
            parts.append(f"norms={_fmt_norms(norms)}")
        parts.append(f"IRLS cool={p.get('irls_cooling_factor', '?')}")
        parts.append(f"max_irls={p.get('max_irls_iterations', '?')}")
        if p.get("use_preconditioner"):
            parts.append("precond=ON")
        parts.append(f"iter={p.get('max_iter', '?')}")
    elif node_type == "SingleInversionNode":
        parts.append(f"method={p.get('method_type', '?')}")
        parts.append(f"reg=smooth (L2)")
        parts.append(f"cool={p.get('cooling_factor', '?')}")
        parts.append(f"iter={p.get('max_iter', '?')}")
    elif node_type == "JointInversionNode":
        methods = p.get("method_types", [])
        weights = p.get("weights", [])
        parts.append(f"methods={'+'.join(methods)}")
        if weights:
            parts.append(f"weights={weights}")
        cg = p.get("cross_gradient_weight", 0)
        if cg:
            parts.append(f"cross_grad={cg}")
        parts.append(f"iter={p.get('max_iter', '?')}")
    else:
        for key in ("method_type", "value", "factor", "offset", "weight",
                    "alpha_s", "max_iter"):
            if key in p:
                parts.append(f"{key}={p[key]}")

    return "\n".join(parts)


def _fmt_norms(norms) -> str:
    """Format norms tuple as a compact string like '(0,2,2,1)'."""
    def _n(v):
        return str(int(v)) if float(v) == int(v) else str(v)
    return "(" + ",".join(_n(v) for v in norms) + ")"


# ── Matplotlib rendering ──────────────────────────────────────────────

def plot_dag(
    graph: Graph,
    figsize: tuple[float, float] = (14, 8),
    title: str = "Inversion Workflow DAG",
    save_path: Optional[str] = None,
):
    """Plot the DAG using matplotlib and networkx.

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for DAG visualization: "
                          "pip install networkx")

    G = nx.DiGraph()

    for node in graph.nodes:
        G.add_node(node.id, label=_label_for(node), color=_color_for(node))
        for inp in node.inputs:
            G.add_edge(inp.id, node.id)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Use a layered layout (topological)
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except (ImportError, Exception):
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

    colors = [G.nodes[n]["color"] for n in G.nodes]
    labels = {n: G.nodes[n]["label"] for n in G.nodes}

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=colors, node_size=2000,
        alpha=0.9, edgecolors="white", linewidths=2,
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=7, font_color="white", font_weight="bold",
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#666666", arrows=True,
        arrowsize=15, arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Mermaid text output ───────────────────────────────────────────────

def dag_to_mermaid(graph: Graph) -> str:
    """Generate a Mermaid flowchart string from the DAG.

    Can be pasted into Markdown or rendered with mermaid-cli.
    Inversion nodes include model configuration details.
    """
    lines = ["graph TD"]

    for node in graph.nodes:
        node_type = type(node).__name__
        label = _mermaid_label(node).replace('"', "'")
        if "Create" in node_type or "Input" in node_type or "Survey" in node_type:
            lines.append(f'    N{node.id}(["{label}"])')
        elif "Inversion" in node_type:
            lines.append(f'    N{node.id}{{{{"{label}"}}}}')
        elif "Export" in node_type:
            lines.append(f'    N{node.id}[["{label}"]]')
        else:
            lines.append(f'    N{node.id}["{label}"]')

    for node in graph.nodes:
        for inp in node.inputs:
            lines.append(f"    N{inp.id} --> N{node.id}")

    lines.append("")
    lines.append("    classDef input fill:#4CAF50,color:white")
    lines.append("    classDef model fill:#2196F3,color:white")
    lines.append("    classDef survey fill:#FF9800,color:white")
    lines.append("    classDef transform fill:#00BCD4,color:white")
    lines.append("    classDef inversion fill:#F44336,color:white")
    lines.append("    classDef sparse fill:#E91E63,color:white")
    lines.append("    classDef export fill:#8BC34A,color:white")

    return "\n".join(lines)


def _mermaid_label(node: Node) -> str:
    """Build a Mermaid node label with model config for inversion nodes."""
    p = node.params()
    node_type = type(node).__name__

    if node_type == "SparseInversionNode":
        norms = _fmt_norms(p.get("norms", []))
        precond = "ON" if p.get("use_preconditioner") else "OFF"
        return (f"{node.name}<br/>"
                f"{p.get('method_type','?')} | norms={norms}<br/>"
                f"precond={precond} | iter={p.get('max_iter','?')}")
    elif node_type == "SingleInversionNode":
        return (f"{node.name}<br/>"
                f"{p.get('method_type','?')} | smooth L2<br/>"
                f"cool={p.get('cooling_factor','?')} | iter={p.get('max_iter','?')}")
    elif node_type == "JointInversionNode":
        methods = "+".join(p.get("method_types", []))
        return (f"{node.name}<br/>"
                f"{methods} | iter={p.get('max_iter','?')}")
    else:
        return node.name
