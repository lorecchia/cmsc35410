"""Lecture 12 figure: sparsification by effective-resistance (leverage) sampling vs. uniform sampling.
Run from lectures/lecture12:  python3 scripts/make_figures.py   (numpy, networkx, matplotlib).
Prints the numbers quoted on the slides and writes figures/sampling.svg."""
import os, numpy as np, networkx as nx, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, GREY, GREEN, BG, INK = "#2563eb", "#ea580c", "#8a857b", "#15803d", "#fbfaf7", "#1c1b19"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "svg.fonttype": "path"})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"); os.makedirs(OUT, exist_ok=True)

def build(n1=150, d=3, bridges=3, seed=1):
    """A dense half (complete graph K_n1) joined by a few edges to a sparse half (random d-regular graph)."""
    G = nx.disjoint_union(nx.complete_graph(n1), nx.random_regular_graph(d, n1, seed=seed))
    rng = np.random.default_rng(seed)
    for _ in range(bridges): G.add_edge(int(rng.integers(n1)), n1 + int(rng.integers(n1)))
    return G

def setup(G):
    n = G.number_of_nodes(); E = np.array(G.edges()); m = len(E)
    B = np.zeros((m, n)); B[np.arange(m), E[:, 0]] = 1; B[np.arange(m), E[:, 1]] = -1
    L = B.T @ B; lam, U = np.linalg.eigh(L); keep = lam > 1e-9
    Lph = (U[:, keep] / np.sqrt(lam[keep])) @ U[:, keep].T        # L^{+/2}
    V = B @ Lph                                                     # rows v_e = L^{+/2} b_e (unit weights)
    lev = np.sum(V ** 2, axis=1)
    return n, m, V, lev, U[:, keep]

def relative_spectrum(V, Uk, counts, probs, q):
    w = counts / (q * probs)                                        # reweighting 1/(q p_e) per sample
    M = (V * w[:, None]).T @ V                                      # sum_e w_e v_e v_e^T, on im L
    return np.linalg.eigvalsh(Uk.T @ M @ Uk)

def main():
    G = build(); n, m, V, lev, Uk = setup(G); rng = np.random.default_rng(0)
    print(f"n={n}, m={m}, sum of leverage scores={lev.sum():.3f} (n-1={n-1}), n ln n={n*np.log(n):.0f}")
    big = np.sort(lev)[::-1][:6]; print("largest leverage scores:", np.round(big, 3), " dense-half median:", round(float(np.median(lev[lev<0.05])), 4), " sparse-half median:", round(float(np.median(lev[lev>0.05])), 3), " leverage mass on sparse half:", round(float(lev[lev>0.05].sum()), 1), " edges with lev>0.05:", int((lev>0.05).sum()))
    qs = np.unique(np.round(np.logspace(np.log10(500), np.log10(60000), 12)).astype(int))
    res = {"leverage": [], "uniform": []}
    for q in qs:
        for name, pr in (("leverage", lev / lev.sum()), ("uniform", np.full(m, 1 / m))):
            errs = []
            for _ in range(10):
                counts = rng.multinomial(q, pr); ev = relative_spectrum(V, Uk, counts, pr, q)
                errs.append(max(ev.max() - 1, 1 - ev.min()))
            res[name].append(np.median(errs))
    for i, q in enumerate(qs):
        print(f"q={q:6d}: median eps leverage={res['leverage'][i]:.3f}, uniform={res['uniform'][i]:.3f}")
    q0 = 4000
    spectra = {}
    for name, pr in (("leverage", lev / lev.sum()), ("uniform", np.full(m, 1 / m))):
        counts = rng.multinomial(q0, pr); spectra[name] = relative_spectrum(V, Uk, counts, pr, q0)
        print(f"q={q0}, {name}: distinct edges {int((counts>0).sum())}, spectrum in [{spectra[name].min():.3f}, {spectra[name].max():.3f}]")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.loglog(qs, res["leverage"], "o-", color=BLUE, lw=2, label="leverage sampling")
    ax.loglog(qs, res["uniform"], "s-", color=ORANGE, lw=2, label="uniform sampling")
    ax.loglog(qs, np.sqrt(n * np.log(n) / qs), "--", color=GREY, lw=1.5, label=r"$\sqrt{n\ln n/q}$")
    ax.axvline(m, color=GREY, lw=0.8, ls=":"); ax.text(m * 1.05, 0.25, "$q=m$", color=GREY)
    ax.set_xlabel("samples $q$"); ax.set_ylabel(r"$\varepsilon=\max|\lambda-1|$  (median of 10)")
    ax.set_title("distortion vs. number of samples", fontsize=13); ax.legend(frameon=False, fontsize=11)
    ax = axes[1]; k = np.arange(1, n)
    ax.plot(k, spectra["uniform"], color=ORANGE, lw=2, label="uniform")
    ax.plot(k, spectra["leverage"], color=BLUE, lw=2, label="leverage")
    ax.axhline(1, color=GREY, lw=0.8)
    ax.set_xlabel("eigenvalue index"); ax.set_ylabel(r"spectrum of $L_G^{+/2}L_HL_G^{+/2}$")
    ax.set_title(f"one sample of $q={q0}$ edges", fontsize=13); ax.legend(frameon=False, fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sampling.svg"), bbox_inches="tight")
    if os.environ.get("PREVIEW"): fig.savefig(os.environ["PREVIEW"], dpi=70, bbox_inches="tight")
    print("wrote figures/sampling.svg")

if __name__ == "__main__":
    main()
