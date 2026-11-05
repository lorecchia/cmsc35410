"""Lecture 11: data for demos/heat-candidate-demo.html (the heat kernel at tau0 as a candidate for SDP(b) <= gamma).
Run from lectures/lecture11:  python3 scripts/heat_candidate_demo.py   (numpy, networkx).
Rewrites the `const DATA = ...;` line of the demo and prints the numbers quoted on the slide.

For each graph we store the spectrum of NL = D^{-1/2} L D^{-1/2} on sqrt(pi)^perp and, for the modes that can carry
weight for gamma <= 1, q_k(i) = u_k(i)^2 / pi_i. The demo then computes, for tau0 = 8 ln n / gamma,
  Y = e^{-tau0 NL} / tr  on sqrt(pi)^perp,   ||v_i||^2 = Y_ii / pi_i = sum_k w_k q_k(i) / sum_k w_k,   w_k = e^{-tau0 lam_k}."""
import os, re, sys, json, numpy as np, networkx as nx
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "lecture10", "scripts"))
from make_figures import build, normalized          # the Lecture 10 example
from loop_figure import layout as whisker_layout    # its drawing in figures/loop-whiskers.svg
DEMO = os.path.join(HERE, "..", "demos", "heat-candidate-demo.html")
GAMMA_MAX, CUTOFF = 1.0, 30.0   # keep modes with tau0(GAMMA_MAX) * (lam_k - lam_2) <= CUTOFF

def side_by_side(G, parts, gap=2.6, seed=4):
    pos = {}
    for j, nodes in enumerate(parts):
        sub = nx.spring_layout(G.subgraph(nodes), seed=seed + j)
        for v, p in sub.items(): pos[v] = np.array([j * gap, 0.0]) + p
    return pos

def dumbbell(m=80, deg=4, bridges=3):
    G = nx.disjoint_union(nx.random_regular_graph(deg, m, seed=1), nx.random_regular_graph(deg, m, seed=2))
    for k in range(bridges): G.add_edge(k * 7 % m, m + (k * 11) % m)
    return G, side_by_side(G, [range(m), range(m, 2 * m)])

def expander_whisker(m=150, deg=8, L=10):
    G = nx.random_regular_graph(deg, m, seed=3); pos = nx.spring_layout(G, seed=4)
    a = max(range(m), key=lambda v: pos[v][0]); dirv = np.array([1.0, 0.0]); prev = a
    for i in range(L): G.add_edge(prev, m + i); pos[m + i] = pos[a] + 0.12 * (i + 1) * dirv; prev = m + i
    return G, pos

def expander(m=150, deg=4):
    G = nx.random_regular_graph(deg, m, seed=5); return G, nx.spring_layout(G, seed=4)

def grid(k=14):
    G = nx.grid_2d_graph(k, k); pos = {v: np.array(v, float) for v in G}
    idx = {v: i for i, v in enumerate(G)}; return nx.relabel_nodes(G, idx), {idx[v]: p for v, p in pos.items()}

def whiskers():
    G, region, anchor = build(); return G, whisker_layout(G, region, anchor)

GRAPHS = [("dumbbell", "two expanders + 3 bridges", dumbbell),
          ("whiskers", "two clusters + six whiskers (Lecture 10)", whiskers),
          ("whisker1", "8-regular expander + one whisker", expander_whisker),
          ("expander", "expander (4-regular)", expander),
          ("grid", "14 × 14 grid", grid)]

def pack(key, label, make):
    G, pos = make(); n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(n)); NL, d = normalized(A); pi = d / d.sum()
    lam, U = np.linalg.eigh(NL); lam, U = lam[1:], U[:, 1:]          # drop sqrt(pi): plays live on W
    tau_min = 8 * np.log(n) / GAMMA_MAX; K = int(np.sum(tau_min * (lam - lam[0]) <= CUTOFF))
    q = (U[:, :K] ** 2 / pi[:, None]).T
    P = np.array([pos[v] for v in range(n)]); P = (P - P.min(0)) / (P.max(0) - P.min(0)).max()
    g = lambda x: float(f"{x:.3g}")
    return dict(key=key, label=label, n=n, pi=[g(x) for x in pi], lam=[g(x) for x in lam],
                q=[[g(x) if x >= 1e-4 else 0 for x in row] for row in q], xy=[[round(x, 3), round(y, 3)] for x, y in P],
                edges=[[int(u), int(v)] for u, v in G.edges]), (lam, U, pi)

def check(lam, U, pi, gamma, b):
    n = len(pi); tau = 8 * np.log(n) / gamma; w = np.exp(-tau * (lam - lam[0])); w /= w.sum()
    norms = (U ** 2 * w).sum(1) / pi; viol, heavy = norms > 1 / b, norms >= 16 / b
    return (tau, float(lam @ w), norms.max(), pi[viol].sum(), (pi * norms)[viol].sum(),
            pi[heavy].sum(), (pi * norms)[heavy].sum())

def main(gamma=0.02, b=0.25):
    data = []
    for key, label, make in GRAPHS:
        D, (lam, U, pi) = pack(key, label, make); data.append(D)
        tau, E, mx, vol, var, volH, varH = check(lam, U, pi, gamma, b)
        print(f"{label:42s} n={D['n']:3d} modes={len(D['q']):3d} lam2={lam[0]:.4f} tau0={tau:.0f} "
              f"energy={E:.4f} max||v||^2={mx:.1f} violators: vol {vol:.3f}, variance {var:.2f}; "
              f"heavy: vol {volH:.4f}, variance {varH:.2f}")
    html = open(DEMO).read()
    html = re.sub(r"^const DATA = .*;$", lambda _: "const DATA = " + json.dumps(data, separators=(",", ":")) + ";",
                  html, count=1, flags=re.M)
    open(DEMO, "w").write(html); print(f"wrote {os.path.relpath(DEMO)} ({len(html) // 1024} KB)")

if __name__ == "__main__":
    main()
