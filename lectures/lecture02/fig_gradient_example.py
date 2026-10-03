"""Example graph for the gradient slide: f at vertices, (grad f)(e) on oriented edges,
the signed incidence matrix B (= nabla) that represents the same operator, and the
divergence of the flow X = grad f (counting measure, m = 1)."""
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY, GREEN = "#2563eb", "#ea580c", "#8a857b", "#15803d"

POS = {"A": (-2.0, 0.0), "B": (-1.0, 1.0), "C": (-1.0, -1.0), "D": (0.0, 0.0), "E": (1.3, 0.0)}
FVAL = {"A": 1.0, "B": 3.5, "C": -0.5, "D": 2.0, "E": -1.5}
EDGES = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
VERTS = list(POS)


def _divergence():
    """div(X)_i = sum_{out of i} X_e - sum_{into i} X_e, for X = grad f, m = 1."""
    div = {v: 0.0 for v in VERTS}
    for i, j in EDGES:
        g = FVAL[j] - FVAL[i]
        div[i] += g
        div[j] -= g
    return div


DIV = _divergence()


def _draw_graph(ax, vertex_values=FVAL, vertex_color=BLUE, vertex_fmt="{:.1f}"):
    for i, j in EDGES:
        xi, yi = POS[i]; xj, yj = POS[j]
        ax.annotate("", xy=(xj, yj), xytext=(xi, yi),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8,
                                     shrinkA=24, shrinkB=24, mutation_scale=18))
        mx, my = (xi + xj) / 2, (yi + yj) / 2
        dx, dy = xj - xi, yj - yi
        nrm = (dx**2 + dy**2)**0.5
        ox, oy = 0.2 * (-dy) / nrm, 0.2 * dx / nrm
        grad = FVAL[j] - FVAL[i]
        ax.text(mx + ox, my + oy, f"${grad:+.1f}$", color=ORANGE, fontsize=13,
                ha="center", va="center", bbox=dict(boxstyle="round,pad=0.15", fc="#fbfaf7", ec="none"))

    for v, (x, y) in POS.items():
        ax.scatter([x], [y], s=1300, color=vertex_color, zorder=3, edgecolors="white", linewidths=2)
        ax.text(x, y, f"${vertex_fmt.format(vertex_values[v])}$", color="white", fontsize=12, fontweight="bold",
                ha="center", va="center", zorder=4)
        ax.text(x, y - 0.4, f"${v}$", color=GREY, fontsize=11, ha="center", va="center")

    ax.set_xlim(-2.9, 2.0); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal"); ax.set_axis_off()


def _draw_incidence(ax):
    n, m = len(EDGES), len(VERTS)
    col = {v: c for c, v in enumerate(VERTS)}

    for c, v in enumerate(VERTS):
        ax.text(c, -0.7, f"${v}$", color=BLUE, fontsize=12, fontweight="bold",
                ha="center", va="center")
    for r, (i, j) in enumerate(EDGES):
        ax.text(-0.9, r, f"${i}\\to {j}$", color=GREY, fontsize=11, ha="right", va="center")

    for r, (i, j) in enumerate(EDGES):
        for c in range(m):
            val = -1 if c == col[i] else (1 if c == col[j] else 0)
            txt = f"${val:+d}$" if val else "$0$"
            color = ORANGE if val else "#c8c3b8"
            weight = "bold" if val else "normal"
            ax.text(c, r, txt, color=color, fontsize=13, fontweight=weight,
                    ha="center", va="center")

    for c in range(m + 1):
        ax.axvline(c - 0.5, color="#e4e0d8", lw=1, ymin=0.06, ymax=0.98)
    for r in range(n + 1):
        ax.axhline(r - 0.5, color="#e4e0d8", lw=1, xmin=0.03, xmax=1.0)

    ax.set_xlim(-1.6, m - 0.4); ax.set_ylim(n - 0.3, -1.1)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(r"$B \in \mathbb{R}^{E\times V}$", fontsize=13, pad=10)


def draw(figsize=(6, 3.6)):
    fig, ax = plt.subplots(figsize=figsize)
    _draw_graph(ax)
    fig.tight_layout()
    return fig


def draw_divergence(figsize=(6, 3.6)):
    fig, ax = plt.subplots(figsize=figsize)
    _draw_graph(ax, vertex_values=DIV, vertex_color=GREEN, vertex_fmt="{:+.1f}")
    fig.tight_layout()
    return fig


def draw_with_incidence(figsize=(11, 3.8)):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.3, 1]})
    _draw_graph(ax0)
    _draw_incidence(ax1)
    fig.tight_layout()
    return fig
