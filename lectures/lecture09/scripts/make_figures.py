"""Figures for Lecture 9 (run: python make_figures.py). Writes ../figures/*.svg and prints the numbers quoted on the slides."""
import os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from core import *
import scipy.sparse.linalg
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', 'figures') + os.sep
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.transparent': True, 'svg.fonttype': 'path', 'font.family': 'DejaVu Sans'})
BLUE, ORANGE, GREY, GREEN, PURPLE, RED = '#2a6fdb', '#e8792b', '#888', '#2a9d5c', '#8e44ad', '#c0392b'
FIGS = os.environ.get('FIGS', '12345').strip()

def xy(k, u): return np.array([u % k, -(u // k)], float)

# ---------------------------------------------------------------- Fig 1: a fundamental cycle
if '1' in FIGS:
    k = 6; n, U, V, W = grid(k); T = bfs_tree(n, U, V, 0); tr = RootedTree(n, U, V, W, T)
    inT = np.zeros(len(U), bool); inT[T] = True
    e = [i for i in range(len(U)) if not inT[i] and U[i] == 3 * k + 3 and V[i] == 3 * k + 4][0]
    cyc = tr.path_edges(V[e], U[e])
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for i in range(len(U)):
        a, b = xy(k, U[i]), xy(k, V[i])
        if inT[i]: ax.plot(*zip(a, b), color=BLUE, lw=2.2, zorder=2)
        else: ax.plot(*zip(a, b), color='#bbb', lw=1, ls=(0, (3, 3)), zorder=1)
    for i in cyc:
        a, b = xy(k, U[i]), xy(k, V[i]); ax.plot(*zip(a, b), color=ORANGE, lw=6, alpha=.55, zorder=3)
    a, b = xy(k, U[e]), xy(k, V[e]); ax.plot(*zip(a, b), color=RED, lw=4, zorder=4)
    ax.annotate('$e$', (a + b) / 2 + np.array([0, -.32]), ha='center', color=RED, fontsize=14)
    ax.annotate('$a$', a + np.array([-.25, -.3]), fontsize=13); ax.annotate('$b$', b + np.array([.1, -.3]), fontsize=13)
    P = np.array([xy(k, u) for u in range(n)]); ax.scatter(P[:, 0], P[:, 1], s=18, color='#333', zorder=5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'tree $T$ (blue), off-tree $e$ (red), cycle $C_e$; stretch$_T(e)={len(cyc)}$', fontsize=11.5)
    plt.tight_layout(); plt.savefig(OUT + 'cycle-toggle.svg'); plt.close()
    print('fig1: stretch of e =', len(cyc))

# ---------------------------------------------------------------- Fig 2: BFS vs recursive tree on the grid
if '2' in FIGS:
    k = 32; n, U, V, W = grid(k)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.0))
    norm = LogNorm(1, 64)
    for ax, (name, T) in zip(axes, [('BFS tree', bfs_tree(n, U, V, 0)), ('recursive "cross" tree', grid_lowstretch_tree(k))]):
        tr = RootedTree(n, U, V, W, T); inT = np.zeros(len(U), bool); inT[T] = True
        st = np.array([tr.dist(U[e], V[e]) for e in range(len(U))])
        off = np.where(~inT)[0]
        segs = [np.array([xy(k, U[e]), xy(k, V[e])]) for e in off]
        lc = LineCollection(segs, cmap='YlOrRd', norm=norm, linewidths=2.2); lc.set_array(st[off]); ax.add_collection(lc)
        ax.add_collection(LineCollection([np.array([xy(k, U[e]), xy(k, V[e])]) for e in T], colors=BLUE, linewidths=1.3))
        ax.set_xlim(-1, k); ax.set_ylim(-k, 1); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f'{name}: average stretch {st.mean():.1f}', fontsize=13)
        print(f'fig2: {name} avg stretch {st.mean():.2f}, max {st.max():.0f}')
    cb = fig.colorbar(lc, ax=axes, fraction=0.025, pad=0.02); cb.set_label('stretch of off-tree edge')
    plt.savefig(OUT + 'grid-trees.svg', bbox_inches='tight'); plt.close()

# ---------------------------------------------------------------- Fig 3: convergence on grids
def ref_solution(L, chi):
    """Exact solve of the grounded system (vertex 0 fixed), then centred."""
    n = L.shape[0]; x = np.zeros(n)
    x[1:] = scipy.sparse.linalg.spsolve(L[1:, 1:].tocsc(), chi[1:])
    return x - x.mean()

def kosz_run(n, U, V, W, T, chi, xs, L, nrm, max_tau, max_work, seed=3):
    K = KOSZ(n, U, V, W, T, chi, np.random.default_rng(seed)); every = max(1, min(int(K.tau / 8), 2 * len(U)))
    errs, work, steps = [], [], []
    for it in range(int(max_tau * K.tau) + 1):
        if it % every == 0:
            xk = K.potentials(); errs.append(np.sqrt((xk - xs) @ (L @ (xk - xs))) / nrm); work.append(K.ds.touches); steps.append(it)
            if errs[-1] < 1e-6 or K.ds.touches > max_work: break
        K.step()
    return np.array(steps), np.array(errs), np.array(work), K.tau

if '3' in FIGS:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={'width_ratios': [1, 1, 0.85]})
    rate = []
    for col, (title, G, trees, xmax, tmax) in enumerate([
            ('64$\\times$64 grid, unit weights', grid(64), 'unweighted', 3000, 9),
            ('32$\\times$32 grid, weights $e^{U(-5,5)}$', grid(32, np.random.default_rng(0), (-5, 5)), 'weighted', 1200, 14)]):
        n, U, V, W = G; L = laplacian(n, U, V, W); m = len(U); chi = np.zeros(n); chi[0], chi[-1] = 1, -1
        xs = ref_solution(L, chi); nrm = np.sqrt(xs @ (L @ xs))
        lmax = sp.linalg.eigsh(L, 1, which='LA', return_eigenvectors=False)[0]
        if trees == 'unweighted': TT = [('BFS', bfs_tree(n, U, V, 0), '--'), ('recursive', grid_lowstretch_tree(64), '-')]
        else: TT = [('BFS', bfs_tree(n, U, V, 0), '--'), ('max-weight', max_weight_tree(n, U, V, W), '-')]
        eg = gd(L, chi, xs, xmax, 1 / lmax) / nrm; ec = pcg(L, chi, xs, xmax) / nrm
        Tp = TT[1][1]; trp = RootedTree(n, U, V, W, Tp); ep = pcg(L, chi, xs, xmax, prec=lambda r: trp.solve(r)[0]) / nrm
        ax = axes[col]
        ax.semilogy(np.arange(len(eg)), eg, color=GREY, label='gradient descent')
        ax.semilogy(np.arange(len(ec)), ec, color=BLUE, label='conjugate gradient')
        ax.semilogy(np.arange(len(ep)) * (m + n) / m, ep, color=GREEN, label=f'PCG, {TT[1][0]} tree')
        it6 = lambda e: int(np.argmax(e < 1e-6)) if (e < 1e-6).any() else None
        print(f'fig3 [{trees}] n={n} m={m} lambda_max={lmax:.3g}: GD {it6(eg)}, CG {it6(ec)}, PCG {it6(ep)} its to 1e-6')
        for name, T, ls in TT:
            s_, e_, w_, tau = kosz_run(n, U, V, W, T, chi, xs, L, nrm, tmax, 1.05 * xmax * m)
            ax.semilogy(w_ / m, e_, color=ORANGE, ls=ls, label=f'KOSZ, {name} tree ($\\tau={tau/m:.0f}m$)' if tau / m > 10 else f'KOSZ, {name} tree ($\\tau={tau/m:.2g}m$)')
            if tau / m < 100: rate.append((f'{trees}, {name}', s_ / tau, e_, ls, col))
            j = np.argmax(e_ < 1e-3) if (e_ < 1e-3).any() else len(e_) - 1
            print(f'   KOSZ {name}: tau={tau:.0f} = {tau/m:.2f} m; error {e_[-1]:.1e} after {s_[-1]/tau:.1f} tau = {w_[-1]/m:.0f} m work; '
                  f'1e-3 at {w_[j]/m:.0f} m work ({s_[j]/tau:.1f} tau)')
        ax.set_xlim(0, xmax); ax.set_ylim(1e-6, 30); ax.set_xlabel('work / $m$   (matvec $= m$; KOSZ: data-structure touches)', fontsize=11)
        if col == 0: ax.set_ylabel(r'$\|x-x^*\|_L/\|x^*\|_L$')
        ax.legend(fontsize=9.5, frameon=False, loc='lower right' if col == 0 else 'upper right'); ax.set_title(title, fontsize=12)
    ax = axes[2]
    for lab, kk, ee, ls, col in rate: ax.semilogy(kk, ee, color=[ORANGE, PURPLE][col], ls=ls, label=lab)
    kk = np.linspace(0, 14, 50); ax.semilogy(kk, np.exp(-kk / 2), color=GREY, ls=':', label=r'slope of the guarantee: $e^{-k/2\tau}$')
    ax.set_xlabel(r'iterations $k/\tau$'); ax.set_ylim(1e-6, 3); ax.set_xlim(0, 14); ax.legend(fontsize=9.5, frameon=False)
    ax.set_title('KOSZ runs against iterations / $\\tau$', fontsize=12)
    plt.tight_layout(); plt.savefig(OUT + 'grid-convergence.svg'); plt.close()

# ---------------------------------------------------------------- Fig 4: exact vs approximate elimination
if '4' in FIGS:
    k = 24; n, U, V, W = grid(k); m = len(U)
    _, _, tr_ex = approx_cholesky(n, U, V, W, 1, np.random.default_rng(0), exact=True)
    L = laplacian(n, U, V, W, dense=True); w_, Q = np.linalg.eigh(L); Q = Q[:, 1:]; Lih = Q @ np.diag(w_[1:] ** -0.5) @ Q.T
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    ax = axes[0]; ax.plot(np.arange(len(tr_ex)), tr_ex / m, color=RED, label='exact Cholesky (fill-in)')
    for rho, c in [(1, BLUE), (8, GREEN)]:
        _, _, trc = approx_cholesky(n, U, V, W, rho, np.random.default_rng(1)); ax.plot(np.arange(len(trc)), trc / m, color=c, label=f'approximate, $\\rho={rho}$')
    ax.set_yscale('log'); ax.set_ylim(0.1, 40); ax.set_xlabel('vertices eliminated (random order)'); ax.set_ylabel('(multi-)edges remaining / $m$')
    ax.legend(fontsize=10, frameon=False); ax.set_title(f'{k}$\\times${k} grid: size of the remaining graph', fontsize=12)
    print(f'fig4: exact peak {tr_ex.max()/m:.1f} m')
    ax = axes[1]; rhos = [1, 2, 4, 8, 16, 32, 64]; lo, hi = [], []
    for rho in rhos:
        a, b = [], []
        for s_ in range(4):
            C, dv, _ = approx_cholesky(n, U, V, W, rho, np.random.default_rng(10 + s_))
            ev = np.linalg.eigvalsh(Lih @ (C @ sp.diags(dv) @ C.T).toarray() @ Lih)[1:]; a.append(ev.min()); b.append(ev.max())
        lo.append(np.mean(a)); hi.append(np.mean(b))
    ax.fill_between(rhos, lo, hi, color=BLUE, alpha=.25); ax.plot(rhos, lo, 'o-', color=BLUE); ax.plot(rhos, hi, 'o-', color=BLUE)
    ax.axhline(1, color=GREY, lw=1); ax.set_xscale('log', base=2); ax.set_xlabel(r'split factor $\rho$')
    ax.set_ylabel(r'spectrum of $L^{+/2}\,\tilde L\,L^{+/2}$'); ax.set_title('approximation quality', fontsize=12)
    print('fig4: rho, [lo, hi]:', [(r, round(a, 2), round(b, 2)) for r, a, b in zip(rhos, lo, hi)])
    plt.tight_layout(); plt.savefig(OUT + 'approx-elimination.svg'); plt.close()

# ---------------------------------------------------------------- Fig 5: continuous analogue (Poisson on a square, Neumann)
if '5' in FIGS:
    srcs = [((.3, .5), +1), ((.7, .5), -1)]
    def refl(c, i): return i + c if i % 2 == 0 else i + 1 - c
    def field(X, Y):
        u = np.zeros_like(X); gx = np.zeros_like(X); gy = np.zeros_like(X)
        for (px, py), q in srcs:
            for i in range(-4, 5):
                for j in range(-4, 5):
                    dx, dy = X - refl(px, i), Y - refl(py, j); r2 = dx**2 + dy**2 + 1e-4
                    u += -q * np.log(r2) / (4 * np.pi); gx += -q * dx / (2 * np.pi * r2); gy += -q * dy / (2 * np.pi * r2)
        return u, gx, gy
    g = np.linspace(0, 1, 300); X, Y = np.meshgrid(g, g); u, gx, gy = field(X, Y)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cs = ax.contourf(X, Y, np.clip(u, -.45, .45), levels=21, cmap='RdBu_r', alpha=.85)
    ax.contour(X, Y, np.clip(u, -.45, .45), levels=21, colors='white', linewidths=.5)
    # current phi = -grad u, i.e. sigma = 1; streamlines from the source
    ax.streamplot(g, g, -gx, -gy, color='#222', density=.8, linewidth=1.1, arrowsize=1.2)
    ax.scatter([.3], [.5], s=90, color=RED, zorder=5); ax.scatter([.7], [.5], s=90, color=BLUE, zorder=5)
    ax.annotate(r'source $\rho>0$', (.3, .8), ha='center', fontsize=12, color=RED, bbox=dict(fc='white', ec='none', alpha=.85, pad=2))
    ax.annotate(r'sink $\rho<0$', (.7, .8), ha='center', fontsize=12, color=BLUE, bbox=dict(fc='white', ec='none', alpha=.85, pad=2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True)
    ax.set_title(r'potential $u$ (colour, level sets) and flow $\phi=-\sigma\nabla u$ (streamlines)', fontsize=11.5)
    plt.tight_layout(); plt.savefig(OUT + 'poisson-square.svg'); plt.close()
    print('fig5: poisson-square written')
