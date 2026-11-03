"""Lecture 11: run the warm-started MMWU loop for balanced separator on the Lecture 10 example.
Run from lectures/lecture11:  python3 scripts/loop_figure.py   (numpy, networkx, matplotlib).
Prints the numbers quoted on the slides and writes figures/loop-whiskers.svg.

Conventions (match the slides): normalized Laplacian NL = D^{-1/2} L D^{-1/2}; plays Y_t are density
matrices on sqrt(pi)^perp, Y_t ∝ exp(-tau0*NL - eta*S_{t-1}) (FTRL centred at the heat kernel C = e^{-tau0 NL}/Z);
||v_i||^2 = Y_ii / pi_i; losses 2*gamma*P Delta_H P only; heavy set H_t = {||v_i||^2 >= 16/b};
tau0 = 8 ln n / gamma, eta = 1/(4 gamma), T = ceil(32 ln n)."""
import os, sys, numpy as np, networkx as nx, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lecture10", "scripts"))
from make_figures import build, normalized, WL  # same graph as the Lecture 10 figure
BLUE, ORANGE, GREY, GREEN, BG, INK = "#2563eb", "#ea580c", "#8a857b", "#15803d", "#fbfaf7", "#1c1b19"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "svg.fonttype": "path"})
OUT = os.path.join(HERE, "..", "figures"); os.makedirs(OUT, exist_ok=True)

def loop(A, b, gamma, seed=0, projections=20):
    n = len(A); NL, d = normalized(A); vol = d.sum(); pi = d / vol
    sp = np.sqrt(pi); P = np.eye(n) - np.outer(sp, sp)
    tau0, eta, T = 8 * np.log(n) / gamma, 1 / (4 * gamma), int(np.ceil(32 * np.log(n)))
    S = np.zeros((n, n)); hist = []; rng = np.random.default_rng(seed)
    for t in range(1, T + 1):
        lam, U = np.linalg.eigh(P @ (tau0 * NL + eta * S) @ P + 1e4 * np.outer(sp, sp))  # play lives on sqrt(pi)^perp
        Y = (U * np.exp(-(lam - lam.min()))) @ U.T; Y /= np.trace(Y)
        energy = float(np.sum(NL * Y)); norms = np.diag(Y) / pi
        heavy = norms >= 16 / b; heavy_var = float(np.diag(Y)[heavy].sum())
        case = "a" if energy >= gamma else ("c" if heavy_var >= 0.5 else "b")
        hist.append((t, case, energy, heavy_var, float(norms.max())))
        if case == "a": return dict(T=T, tau0=tau0, hist=hist, Y=Y, cut=None)   # certificate
        if case == "c": S += 2 * gamma * (P @ np.diag(heavy.astype(float)) @ P); continue
        lY, UY = np.linalg.eigh(Y); V = (UY * np.sqrt(np.clip(lY, 0, None))) @ UY.T / sp[:, None]
        best = (np.inf, 0, None)
        for _ in range(projections):
            y = V @ rng.standard_normal(n); ins = np.zeros(n, bool); cut = 0.0
            for v in np.argsort(y)[:-1]:
                cut += d[v] - 2 * A[v, ins].sum(); ins[v] = True
                p = pi[ins].sum(); bal = min(p, 1 - p)
                if bal >= b / 4 and cut / (vol * bal) < best[0]: best = (cut / (vol * bal), bal, ins.copy())
        return dict(T=T, tau0=tau0, hist=hist, Y=Y, cut=best)
    return dict(T=T, tau0=tau0, hist=hist, Y=Y, cut=None)

def layout(G, region, anchor):
    pos = {}; centers = {"L": np.array([-1.45, 0.0]), "R": np.array([1.45, 0.0])}
    sub = {s: nx.spring_layout(G.subgraph([v for v in G if region[v] == s]), seed=4) for s in "LR"}
    for s in "LR":
        for v, p in sub[s].items(): pos[v] = centers[s] + 0.75 * p
    for j, l in enumerate(WL):
        a = anchor[j]; s = region[a]; c = centers[s]
        ang = {"L": [2.2, 3.4, 4.3][j // 2], "R": [0.9, -0.2, -1.1][j // 2]}[s]
        dirv = np.array([np.cos(ang), np.sin(ang)]); pos[a] = c + 0.8 * dirv
        for i, v in enumerate([v for v in G if region[v] == f"w{j}"]): pos[v] = pos[a] + 0.075 * (i + 1) * dirv
    return pos

def main(b=0.25, gammas=(0.02, 0.05)):
    G, region, anchor = build(); n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(n)); runs = {}
    for g in gammas + (0.1,):
        r = loop(A, b, g); runs[g] = r; h = r["hist"]
        phi, bal, S = r["cut"]
        print(f"gamma={g}: tau0={r['tau0']:.0f}, T={r['T']}, stopped at round {len(h)} (case {h[-1][1]}), "
              f"energy={h[-1][2]:.4f}, heavy var={h[-1][3]:.2f}, max||v||^2={h[-1][4]:.1f}; "
              f"cut Phi={phi:.4f}, balance={bal:.3f}, regions={dict(Counter(region[v] for v in np.where(S)[0]))}")
    pos = layout(G, region, anchor)
    cmap = LinearSegmentedColormap.from_list("w", ["#e7e4dc", "#f4b183", ORANGE, "#9a3412"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax in axes:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cfcac0", width=0.6); ax.set_axis_off(); ax.set_aspect("equal")
    g0 = gammas[0]; share = np.diag(runs[g0]["Y"])
    nx.draw_networkx_nodes(G, pos, ax=axes[0], node_size=16, node_color=share / share.max(), cmap=cmap, vmin=0, vmax=1, linewidths=0)
    axes[0].set_title(f"first play $Y_1=C$: heat kernel at $\\tau_0={runs[g0]['tau0']:.0f}$", fontsize=12)
    axes[0].text(0.5, -0.04, f"energy {runs[g0]['hist'][0][2]:.4f} < $\\gamma={g0}$; heavy set empty: spread", transform=axes[0].transAxes, ha="center", va="top", fontsize=12, color=INK)
    for ax, g in zip(axes[1:], gammas):
        phi, bal, S = runs[g]["cut"]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=16, node_color=[BLUE if S[v] else GREEN for v in G], linewidths=0)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges if S[u] != S[v]], ax=ax, edge_color=ORANGE, width=1.6)
        ax.set_title(f"rounded cut, $\\gamma={g}$", fontsize=12)
        ax.text(0.5, -0.04, f"$\\Phi={phi:.4f}$, balance ${bal:.2f}$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "loop-whiskers.svg"), bbox_inches="tight")
    fig.savefig(os.environ.get("PREVIEW", os.devnull + ".png"), dpi=70, bbox_inches="tight") if os.environ.get("PREVIEW") else None
    print("wrote figures/loop-whiskers.svg")

if __name__ == "__main__":
    main()
