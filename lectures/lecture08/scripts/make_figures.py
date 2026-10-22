"""Figures for Lecture 8 (run: python make_figures.py). Writes ../figures/*.svg"""
import os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', 'figures') + os.sep
src = open(os.path.join(HERE, 'verify_bounds.py')).read()
exec(src.split('# ---------- 1.')[0])                                   # topk, sin_theta, procrustes
exec(src[src.index('def expander'):src.index('for mode in')])             # expander, planted, analyze_planted
exec(src[src.index('def sbm'):src.index('n, k = 400')])                   # sbm, kmeans_err
plt.rcParams.update({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.transparent': True, 'svg.fonttype': 'path', 'font.family': 'DejaVu Sans'})
BLUE, ORANGE, GREY, GREEN = '#2a6fdb', '#e8792b', '#888', '#2a9d5c'

# --- Fig 1: SBM in three regimes: spectrum of A and the embedding
n, k, p = 600, 2, 0.3
fig, ax = plt.subplots(2, 3, figsize=(13, 6.2), gridspec_kw={'height_ratios': [1, 1.25]})
for c, q in enumerate([0.05, 0.18, 0.27]):
    A, P, lab = sbm(n, k, p, q, np.random.default_rng(4))
    w, U = np.linalg.eigh(A); w = w[::-1]; U = U[:, ::-1]
    sigma = np.sqrt((n/k - 1)*p*(1-p) + (n - n/k)*q*(1-q)); delta0 = n*(p-q)/k
    V = np.zeros((n, k)); V[lab == 0, 0] = V[lab == 1, 1] = np.sqrt(k/n)
    s, _ = sin_theta(V, U[:, :k]); nE = np.linalg.norm(A - P, 2)
    print(f"q={q}: delta0={delta0:.1f} sigma={sigma:.2f} |E|={nE:.2f} 2sigma={2*sigma:.2f} sin={s:.3f} sigma/delta0={sigma/delta0:.3f} DK={2*nE/delta0:.3f} mis={kmeans_err(U[:, :k], lab, k, np.random.default_rng(0))}")
    a = ax[0, c]; a.hist(w, bins=90, color='#c9d8f3')
    out = w[w > 2*sigma*1.02]; a.plot(out, np.zeros_like(out)+1, 'o', color=BLUE, ms=6)
    for x in [-2*sigma, 2*sigma]: a.axvline(x, color=GREY, ls='--', lw=1)
    a.set_yticks([]); a.set_title(f"$q={q}$:  $\\delta_0={delta0:.0f}$,  $\\sigma={sigma:.1f}$", fontsize=13)
    a.set_xlabel('eigenvalues of $A$')
    b = ax[1, c]; b.scatter(U[:, 0], U[:, 1], c=[BLUE if l == 0 else ORANGE for l in lab], s=7, alpha=.7, rasterized=True)
    b.set_xticks([]); b.set_yticks([])
    b.set_title(f"$\\|\\sin\\Theta\\|={s:.2f}$,  $\\sigma/\\delta_0={min(1,sigma/delta0):.2f}$,  DK $={2*nE/delta0:.2f}$", fontsize=12)
    b.set_xlabel('$\\hat v_1$'); b.set_ylabel('$\\hat v_2$')
plt.tight_layout(); plt.savefig(OUT + 'sbm-regimes.svg', dpi=150); plt.close()

# --- Fig 2: planted clusters, spread vs hubs
def planted2(k, m, deg, ncross, mode, rng):
    nn = k*m; W = np.zeros((nn, nn)); lab = np.repeat(np.arange(k), m)
    for j in range(k): W[j*m:(j+1)*m, j*m:(j+1)*m] = expander(m, rng, deg)
    for _ in range(ncross):
        j, l = rng.choice(k, 2, replace=False)
        u, v = (j*m + rng.integers(m), l*m + rng.integers(m)) if mode == 'spread' else (j*m + rng.integers(3), l*m + rng.integers(3))
        W[u, v] += 1; W[v, u] += 1
    return W, lab
K = 3; cs = np.arange(0, 401, 25)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
for a, mode in zip(ax, ['spread', 'hubs']):
    R = [analyze_planted(*planted2(K, 80, 20, c, mode, np.random.default_rng(1)), K) for c in cs]
    cap = np.sqrt(K)
    for c, r in zip(cs, R):
        if c in (0, 50, 200, 400): print(mode, c, {x: round(float(y), 3) for x, y in r.items()})
    a.plot(cs, [min(cap, 2*np.sqrt(K)*r['normE']/r['delta0']) for r in R], '-o', color=ORANGE, ms=4, label='Davis–Kahan  $2\\sqrt{k}\\,\\|E\\|_D/\\delta_0$')
    a.plot(cs, [min(cap, r['energy']) for r in R], '-o', color=GREEN, ms=4, label='energy  $\\sqrt{\\sum_j\\Phi(S_j)/\\mu_{k+1}}$')
    a.plot(cs, [r['sinF'] for r in R], '-o', color=BLUE, ms=4, label='actual $\\|\\sin\\Theta\\|_F$')
    a.axhline(cap, color=GREY, ls=':', lw=1)
    a.set_title(('cross edges spread out' if mode == 'spread' else 'cross edges on 3 hubs per cluster') + '  ($k=3$, 80 vertices, degree 20)', fontsize=12)
    a.set_xlabel('number of cross edges')
ax[0].set_ylabel('$\\|\\sin\\Theta\\|_F$  (capped at $\\sqrt{k}$)'); h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=12, loc='lower center', ncol=3, frameon=False)
plt.tight_layout(rect=(0, 0.1, 1, 1)); plt.savefig(OUT + 'planted-spread-vs-hubs.svg'); plt.close()
print('ok')
