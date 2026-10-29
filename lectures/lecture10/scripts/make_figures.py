"""Lecture 10 figures. Run from lectures/lecture10:  python3 scripts/make_figures.py
Needs numpy, networkx, matplotlib. Prints the numbers quoted on the slides."""
import os, numpy as np, networkx as nx, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
BLUE, ORANGE, GREY, GREEN, BG = "#2563eb", "#ea580c", "#8a857b", "#15803d", "#fbfaf7"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "svg.fonttype": "path"})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)

N1, DEG, BRIDGES, WL = 80, 4, 10, (22, 20, 18, 16, 14, 12)

def build():
    G = nx.disjoint_union(nx.random_regular_graph(DEG, N1, seed=1), nx.random_regular_graph(DEG, N1, seed=2))
    for k in range(BRIDGES):
        G.add_edge(k * 7 % N1, N1 + (k * 11) % N1)
    region = {v: ("L" if v < N1 else "R") for v in G}
    anchor, nxt = {}, G.number_of_nodes()
    for j, l in enumerate(WL):
        prev = (j * 19 + 5) % N1 if j % 2 == 0 else N1 + (j * 23 + 3) % N1
        anchor[j] = prev
        for _ in range(l):
            G.add_edge(prev, nxt); region[nxt] = f"w{j}"; prev = nxt; nxt += 1
    return G, region, anchor

def normalized(A):
    d = A.sum(1); s = 1 / np.sqrt(d)
    return np.eye(len(d)) - s[:, None] * A * s[None, :], d

def sweep(A, f):
    d = A.sum(1); vol = d.sum(); order = np.argsort(f)
    ins = np.zeros(len(f), bool); cut = 0.0; best = (np.inf, None)
    for v in order[:-1]:
        cut += d[v] - 2 * A[v, ins].sum(); ins[v] = True
        vs = d[ins].sum(); phi = cut / min(vs, vol - vs)
        if phi < best[0]: best = (phi, ins.copy())
    S = best[1]
    return best[0], (~S if d[S].sum() > vol / 2 else S)

def region_mass(share, region):
    c = Counter()
    for v, s in enumerate(share): c[region[v]] += s
    return c

def main():
    G, region, anchor = build(); n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(n)); Nl, d = normalized(A)
    lam, U = np.linalg.eigh(Nl)
    print(f"n={n}, m={G.number_of_edges()}, lambda_2..4 = {np.round(lam[1:4], 4)}")
    phi, S = sweep(A, U[:, 1] / np.sqrt(d))
    print(f"Fiedler sweep: Phi={phi:.4f}, balance vol(S)/vol(V)={d[S].sum()/d.sum():.3f}, S={dict(Counter(region[v] for v in np.where(S)[0]))}")
    side = np.array([region[v] == "L" or (region[v][0] == "w" and int(region[v][1:]) % 2 == 0) for v in range(n)])
    print(f"Planted balanced cut: Phi={A[side][:, ~side].sum()/min(d[side].sum(), d[~side].sum()):.4f}, balance={d[side].sum()/d.sum():.3f}")
    alive = np.ones(n, bool); removed = 0.0; rounds = []
    while removed < d.sum() / 4:
        idx = np.where(alive)[0]; Ai = A[np.ix_(idx, idx)]
        Ni, di = normalized(Ai); _, Ui = np.linalg.eigh(Ni)
        ph, Si = sweep(Ai, Ui[:, 1] / np.sqrt(di)); rem = idx[Si]
        removed += d[rem].sum(); alive[rem] = False
        rounds.append((ph, dict(Counter(region[v] for v in rem)), removed / d.sum()))
        print(f"  recursive round {len(rounds)}: Phi={ph:.4f}, removed {rounds[-1][1]}, total removed vol fraction {removed/d.sum():.3f}")
    P = np.eye(n) - np.outer(U[:, 0], U[:, 0])
    fied = U[:, 1] ** 2
    shares = {"fiedler": fied}
    for t in (20,):
        K = P @ (U * np.exp(-t * lam)) @ U.T @ P; shares[t] = np.diag(K) / np.trace(K)
    for k, sh in shares.items():
        c = region_mass(sh, region)
        print(f"variance share ({k}):", {r: round(float(x), 3) for r, x in sorted(c.items())})

    # layout: two clusters, whiskers as straight rays
    rng = np.random.default_rng(0); pos = {}
    centers = {"L": np.array([-1.45, 0.0]), "R": np.array([1.45, 0.0])}
    sub = {s: nx.spring_layout(G.subgraph([v for v in G if region[v] == s]), seed=4) for s in "LR"}
    for s in "LR":
        for v, p in sub[s].items(): pos[v] = centers[s] + 0.75 * p
    for j, l in enumerate(WL):
        a = anchor[j]; s = region[a]; c = centers[s]
        ang = {"L": [2.2, 3.4, 4.3][j // 2], "R": [0.9, -0.2, -1.1][j // 2]}[s]
        dirv = np.array([np.cos(ang), np.sin(ang)])
        pos[a] = c + 0.8 * dirv
        ws = [v for v in G if region[v] == f"w{j}"]
        for i, v in enumerate(ws): pos[v] = pos[a] + 0.075 * (i + 1) * dirv
    cmap = LinearSegmentedColormap.from_list("w", ["#e7e4dc", "#f4b183", ORANGE, "#9a3412"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    titles = {"fiedler": r"Fiedler vector  ($\eta=\infty$: follow the leader)", 20: r"Heat kernel $e^{-20\mathcal{L}}$  (finite $\eta$, regularized)"}
    for ax, (k, sh) in zip(axes, shares.items()):
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cfcac0", width=0.6)
        bridges = [(u, v) for u, v in G.edges if {region[u], region[v]} == {"L", "R"}]
        nx.draw_networkx_edges(G, pos, edgelist=bridges, ax=ax, edge_color=BLUE, width=0.9, alpha=0.6)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=16, node_color=sh / sh.max(), cmap=cmap, vmin=0, vmax=1, linewidths=0)
        c = region_mass(sh, region); wsum = sum(x for r, x in c.items() if r[0] == "w")
        ax.set_title(titles[k], fontsize=13)
        ax.text(0.5, -0.04, f"share of variance on whiskers: {wsum:.0%}  ·  on the longest one: {c['w0']:.0%}",
                transform=ax.transAxes, ha="center", va="top", fontsize=12, color="#1c1b19")
        ax.set_axis_off(); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fiedler-vs-heat.svg"), bbox_inches="tight")
    print("wrote figures/fiedler-vs-heat.svg")

if __name__ == "__main__":
    main()
