"""Lecture 10: figures for 'Follow the leader fails' and 'Regularization stabilizes the maximizer'.
Run from lectures/lecture10:  python3 scripts/stability_figures.py   (numpy + matplotlib)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, GREY, GREEN, BG, INK = "#2563eb", "#ea580c", "#8a857b", "#15803d", "#fbfaf7", "#1c1b19"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "svg.fonttype": "path"})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"); os.makedirs(OUT, exist_ok=True)

# ---- Figure A: linear maximizer on a triangle jumps under a small rotation ----
V = np.array([[-1, 0], [1, 0], [0, -np.sqrt(3)]])
def triangle_panel(ax, theta, color, label):
    ax.fill(V[:, 0], V[:, 1], color="#efece5", zorder=0); ax.plot(*np.vstack([V, V[:1]]).T, color=GREY, lw=1.2)
    c = np.array([np.sin(theta), np.cos(theta)])
    vals = V @ c
    for lev in np.linspace(vals.min(), vals.max(), 7)[1:-1]:
        # intersect line c.x = lev with triangle edges
        pts = []
        for i in range(3):
            a, b = V[i], V[(i + 1) % 3]; fa, fb = a @ c - lev, b @ c - lev
            if fa * fb < 0: pts.append(a + (b - a) * fa / (fa - fb))
        if len(pts) == 2: ax.plot(*np.array(pts).T, ls="--", lw=0.9, color=color, alpha=0.55)
    k = np.argmax(vals); ax.scatter(*V[k], s=160, color=ORANGE, zorder=5, edgecolor="white", lw=1.5)
    g = V.mean(0); ax.annotate("", xy=g + 0.75 * c, xytext=g, arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
    ax.text(*(g + 0.85 * c + np.array([0.12, 0])), label, color=color, fontsize=15)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.95, 0.4); ax.set_aspect("equal"); ax.set_axis_off()
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
triangle_panel(axes[0], -0.06, BLUE, r"$S$"); triangle_panel(axes[1], 0.06, BLUE, r"$S+M$")
axes[0].set_title("maximizer: left vertex", fontsize=12, color=INK); axes[1].set_title("maximizer: right vertex", fontsize=12, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "linear-max-jumps.svg"), bbox_inches="tight"); plt.close(fig)

# ---- Figure B: linear vs entropy-regularized objective on the 1-simplex ----
p = np.linspace(1e-4, 1 - 1e-4, 400); H = -(p * np.log(p) + (1 - p) * np.log(1 - p)); eta = 2.0
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for delta, col, name in [(-0.1, BLUE, r"$S$"), (0.1, ORANGE, r"$S+M$")]:
    lin = delta * (p - 0.5); axes[0].plot(p, lin, color=col, lw=2.2, label=name)
    j = np.argmax(lin); axes[0].scatter(p[j], lin[j], s=90, color=col, zorder=5, edgecolor="white")
    reg = delta * (p - 0.5) + H / eta; axes[1].plot(p, reg, color=col, lw=2.2, label=name)
    pstar = 1 / (1 + np.exp(-eta * delta)); r = delta * (pstar - 0.5) - (pstar * np.log(pstar) + (1 - pstar) * np.log(1 - pstar)) / eta
    axes[1].scatter(pstar, r, s=90, color=col, zorder=5, edgecolor="white")
axes[0].annotate("", xy=(0.97, 0.062), xytext=(0.03, 0.062), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
axes[0].text(0.5, 0.068, "maximizer jumps by the whole interval", ha="center", fontsize=12)
axes[0].set_title(r"linear: $\langle S,x\rangle$", fontsize=13); axes[0].set_ylim(-0.065, 0.085)
axes[1].set_title(r"regularized: $\langle S,x\rangle-\frac{1}{\eta}R(x)$, entropy $R$", fontsize=13)
p1, p2 = 1 / (1 + np.exp(0.2)), 1 / (1 + np.exp(-0.2))
axes[1].annotate("", xy=(p2, 0.395), xytext=(p1, 0.395), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
axes[1].text(0.5, 0.405, r"moves by $\leq\eta\,\|M\|$ (in $\|\cdot\|_1$)", ha="center", fontsize=12); axes[1].set_ylim(0.0, 0.5)
for ax in axes:
    ax.set_xlabel(r"$x$ = weight on expert 1"); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([]); ax.legend(frameon=False, loc="lower center")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "regularized-max-stable.svg"), bbox_inches="tight"); plt.close(fig)
print("wrote figures/linear-max-jumps.svg, figures/regularized-max-stable.svg")
