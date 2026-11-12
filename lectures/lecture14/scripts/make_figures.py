"""Lecture 14 figures. Run from lectures/lecture14:  python3 scripts/make_figures.py  (numpy, scipy, matplotlib).
Prints the numbers quoted on the slides.

Discretization used throughout = a weighted path graph: N cells of width h on [0,1], cell centers x_i;
edge (i,i+1) has conductance p(x_{i+1/2})/h, vertex i has mass r(x_i) h; this is the generalized problem
K f = lambda M f, the Neumann Sturm-Liouville problem -(p f')' = lambda r f, f'(0)=f'(1)=0, in the limit."""
import os, numpy as np, matplotlib
from scipy.linalg import eigh
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, GREY, GREEN, BG, INK = "#2563eb", "#ea580c", "#8a857b", "#15803d", "#fbfaf7", "#1c1b19"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "svg.fonttype": "path"})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"); os.makedirs(OUT, exist_ok=True)

def path_operator(N, p=lambda x: np.ones_like(x), r=lambda x: np.ones_like(x)):
    h = 1.0 / N; x = (np.arange(N) + 0.5) * h; xm = (np.arange(1, N)) * h
    c = p(xm) / h; K = np.zeros((N, N))
    for i, ci in enumerate(c):
        K[i, i] += ci; K[i + 1, i + 1] += ci; K[i, i + 1] -= ci; K[i + 1, i] -= ci
    M = np.diag(r(x) * h)
    return x, K, M

def fig_path():
    N = 20; x, K, M = path_operator(N); lam, V = eigh(K, M)
    j = np.arange(N); exact = (np.pi * j) ** 2
    print("path N=20: discrete/continuum for j=1,5,10,19:", [round(lam[k] / exact[k], 4) for k in (1, 5, 10, 19)])
    print("check formula h^-2(2-2cos(pi j h)):", np.allclose(lam, N ** 2 * (2 - 2 * np.cos(np.pi * j / N))))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    ax = axes[0]; xx = np.linspace(0, 1, 400)
    for k, c in zip((1, 2, 3), (BLUE, GREEN, ORANGE)):
        v = V[:, k] / np.sqrt(np.sum(V[:, k] ** 2) / N); s = np.sign(v[0])
        ax.plot(xx, np.sqrt(2) * np.cos(np.pi * k * xx), color=c, lw=1.6, alpha=0.6)
        ax.plot(x, s * v, "o", color=c, ms=5, label=f"$j={k}$")
    ax.set_xlabel("$x$"); ax.set_title(r"eigenvectors of the path ($N=20$) vs. $\sqrt{2}\cos(\pi jx)$", fontsize=12); ax.legend(frameon=False, fontsize=10)
    ax = axes[1]
    ax.plot(j, exact, "-", color=GREY, lw=1.8, label=r"$-f''$: $(\pi j)^2$, unbounded")
    ax.plot(j, lam, "o", color=BLUE, ms=5, label=r"path: $h^{-2}(2-2\cos\pi jh)\leq 4h^{-2}$")
    ax.axhline(4 * N ** 2, color=BLUE, ls=":", lw=1)
    ax.set_xlabel("mode $j$"); ax.set_title("eigenvalues: low modes converge, high modes saturate", fontsize=12); ax.legend(frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "path-vs-continuum.svg"), bbox_inches="tight")
    if os.environ.get("PREVIEW"): fig.savefig(os.environ["PREVIEW"] + "-path.png", dpi=70, bbox_inches="tight")

def fig_bottleneck():
    eps, w = 0.02, 0.04
    p = lambda x: np.where(np.abs(x - 0.5) < w / 2, eps, 1.0)
    N = 2000; x, K, M = path_operator(N, p); lam, V = eigh(K, M)
    xs = np.linspace(0, 1, 200001); R = np.trapezoid(1 / p(xs), xs)
    print(f"bottleneck: lambda_2 = {lam[1]:.3f}, resistance R = int 1/p = {R:.3f}, 4/R = {4/R:.3f}, lambda_3 = {lam[2]:.2f} (uniform p: pi^2 = {np.pi**2:.2f})")
    zeros = [int(np.sum(np.diff(np.sign(V[:, k])) != 0)) for k in range(1, 6)]
    print("bottleneck: zero counts of phi_2..phi_6:", zeros)
    # Weyl check for a smooth variable-coefficient problem
    p2 = lambda x: 1 + 0.8 * np.sin(2 * np.pi * x); r2 = lambda x: 1 + 0.5 * x
    _, K2, M2 = path_operator(N, p2, r2); lam2 = eigh(K2, M2, eigvals_only=True)
    xs2 = np.linspace(0, 1, 200001); I = np.trapezoid(np.sqrt(r2(xs2) / p2(xs2)), xs2)
    ks = np.array([5, 10, 20, 40]); print("Weyl: lambda_k / (k pi / int sqrt(r/p))^2 for k=5,10,20,40:", np.round(lam2[ks] / (ks * np.pi / I) ** 2, 4))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    ax = axes[0]
    ax.fill_between(x, 0, p(x), color=GREY, alpha=0.25, step="mid"); ax.plot(x, p(x), color=GREY, lw=1.2, label="$p(x)$")
    ax.set_ylim(0, 1.15); ax.set_xlabel("$x$"); ax.set_title(r"conductance $p$: a bottleneck at $x=1/2$", fontsize=12); ax.legend(frameon=False, loc="lower left")
    ax = axes[1]
    for k, c in zip((1, 2, 3), (BLUE, GREEN, ORANGE)):
        v = V[:, k] / np.sqrt(np.sum(V[:, k] ** 2) / N); v *= np.sign(v[0])
        ax.plot(x, v, color=c, lw=2, label=fr"$\varphi_{k+1}$, $\lambda={lam[k]:.2f}$")
    ax.axhline(0, color=GREY, lw=0.8); ax.set_xlabel("$x$"); ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.set_title(r"Neumann eigenfunctions of $-(pf')'=\lambda f$", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "bottleneck.svg"), bbox_inches="tight")
    if os.environ.get("PREVIEW"): fig.savefig(os.environ["PREVIEW"] + "-bottle.png", dpi=70, bbox_inches="tight")

def green_check():
    N = 400; h = 1 / N; x = (np.arange(1, N)) * h        # interior nodes, Dirichlet
    T = (2 * np.eye(N - 1) - np.eye(N - 1, k=1) - np.eye(N - 1, k=-1)) / h ** 2
    G = np.minimum.outer(x, x) - np.outer(x, x)
    print("Green's function check: max |T^{-1} - G h^{-1}... | =", np.abs(np.linalg.inv(T) - G * h).max())

if __name__ == "__main__":
    fig_path(); fig_bottleneck(); green_check(); print("wrote figures/path-vs-continuum.svg, figures/bottleneck.svg")
