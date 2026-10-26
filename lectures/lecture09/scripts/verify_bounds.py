"""Numerical checks for every claim/number on the Lecture 9 slides (run: python verify_bounds.py)."""
import numpy as np, time
from core import *

rng = np.random.default_rng(0)
def pinv(L): return np.linalg.pinv(L)
def kappa(L):
    w = np.linalg.eigvalsh(L); return w[-1] / w[1], w[1], w[-1]

print("=" * 70, "\n1. Effective resistance, hitting and commute times")
n, U, V, W = random_regular(40, 3, rng); W = np.exp(rng.uniform(-1, 1, len(U)))
L = laplacian(n, U, V, W, dense=True); Lp = pinv(L); d = np.diag(L).copy(); vol = d.sum()
s, t = 0, 17; bst = np.zeros(n); bst[s], bst[t] = 1, -1
Reff = bst @ Lp @ bst
x = Lp @ bst; f = W * (x[U] - x[V]); print("Thomson: energy of electrical flow =", round(f @ (f / W), 10), " Reff =", round(Reff, 10))
# hitting times by a linear system vs by one Laplacian solve
P = (np.diag(1 / d) @ (np.diag(d) - L))
def hit(tgt):
    idx = [u for u in range(n) if u != tgt]
    h = np.zeros(n); h[idx] = np.linalg.solve(np.eye(n - 1) - P[np.ix_(idx, idx)], np.ones(n - 1)); return h
h_t = hit(t); y = Lp @ (d - vol * np.eye(n)[t]); h_solve = y - y[t]
print("hitting times via L h = d - vol e_t: max error", np.abs(h_t - h_solve).max())
print("commute time identity: h(s,t)+h(t,s) =", round(h_t[s] + hit(s)[t], 8), " vol*Reff =", round(vol * Reff, 8))
lev = np.array([W[e] * (np.eye(n)[U[e]] - np.eye(n)[V[e]]) @ Lp @ (np.eye(n)[U[e]] - np.eye(n)[V[e]]) for e in range(len(U))])
print("Foster: sum_e w_e Reff(e) =", round(lev.sum(), 10), " n-1 =", n - 1)

print("=" * 70, "\n2. Condition numbers (slide table)")
for name, G in [("path n=1000", path(1000)), ("grid 32x32", grid(32)), ("3-regular random n=1024", random_regular(1024, 3, rng)),
                ("complete n=1000", None)]:
    if G is None: k, l2, ln = 1.0, 1000, 1000
    else: k, l2, ln = kappa(laplacian(*G, dense=True))
    print(f"  {name:26s} lambda2={l2:.3g} lambdan={ln:.3g} kappa={k:.4g}")

print("=" * 70, "\n3. GD vs CG iterations to 1e-6 (L-norm, b = e_s - e_t far apart)")
def iters_to(errs, eps=1e-6):
    idx = np.where(errs <= eps * errs[0])[0]; return int(idx[0]) if len(idx) else None
res3 = {}
for name, G, s_, t_ in [("path n=200", path(200), 0, 199), ("grid 32x32", grid(32), 0, 1023), ("3-reg n=1024", random_regular(1024, 3, rng), 0, 1)]:
    n, U, V, W = G; L = laplacian(n, U, V, W); Ld = L.toarray(); b = np.zeros(n); b[s_], b[t_] = 1, -1
    xs = pinv(Ld) @ b; k, l2, ln = kappa(Ld)
    eg = gd(L, b, xs, 400000 if "path" in name else 60000, 1 / ln); ec = pcg(L, b, xs, 5000)
    res3[name] = (iters_to(eg), iters_to(ec), k)
    print(f"  {name:14s} kappa={k:9.1f}  GD:{iters_to(eg)}  (kappa*ln(1e6)={k*np.log(1e6):.0f})   CG:{iters_to(ec)}  (sqrt(kappa)/2*ln(2e6)={np.sqrt(k)/2*np.log(2e6):.0f})")
# Krylov locality: on the path, x_k is supported on k-neighbourhood of supp(b)
n, U, V, W = path(200); L = laplacian(n, U, V, W); b = np.zeros(n); b[0], b[-1] = 1, -1
xs = pinv(L.toarray()) @ b; e99 = pcg(L, b, xs, 99)
print("  path n=200, CG error after 99 its (cannot reach centre yet):", round(e99[-1] / e99[0], 4), "  after 100+: ", end="")
print(round(pcg(L, b, xs, 101)[-1] / e99[0], 12))

print("=" * 70, "\n4. Tree preconditioning: L_T <= L_G <= st_T(G) L_T; PCG iterations")
for k in [16, 32, 64]:
    n, U, V, W = grid(k); L = laplacian(n, U, V, W); Ld = L.toarray()
    for tname, T in [("BFS", bfs_tree(n, U, V, 0)), ("low-stretch", grid_lowstretch_tree(k))]:
        tr = RootedTree(n, U, V, W, T); allst = np.array([tr.dist(U[e], V[e]) * W[e] for e in range(len(U))])
        LT = laplacian(n, U[T], V[T], W[T], dense=True)
        if k <= 32:
            import scipy.linalg as sla; ev = np.sort(sla.eigh(Ld, LT + np.ones((n, n)) / n, eigvals_only=True))[1:]   # pencil (L_G, L_T + J/n)
            evs = f"eig(LT^+ LG) in [{ev.min():.3f}, {ev.max():.1f}]"
        else: evs = ""
        b = np.zeros(n); b[0], b[-1] = 1, -1; xs = pinv(Ld) @ b
        e = pcg(L, b, xs, 3000, prec=lambda r: tr.solve(r)[0])
        print(f"  grid {k}x{k} {tname:11s} st_T(G)={allst.sum():9.0f} (= {allst.sum()/len(U):6.2f} per edge)  {evs}  PCG its to 1e-6: {iters_to(e)}")

# 4b. weighted grid (Fig. 3, middle panel)
n, U, V, W = grid(32, np.random.default_rng(0), (-5, 5)); L = laplacian(n, U, V, W); Ld = L.toarray()
b = np.zeros(n); b[0], b[-1] = 1, -1; xs = pinv(Ld) @ b; k_, _, _ = kappa(Ld)
T = max_weight_tree(n, U, V, W); tr = RootedTree(n, U, V, W, T)
allst = np.array([tr.dist(U[e], V[e]) * W[e] for e in range(len(U))]); tau_ = allst.sum() - (n - 1) + (len(U) - n + 1)
ec = pcg(L, b, xs, 5000); ep = pcg(L, b, xs, 500, prec=lambda r: tr.solve(r)[0])
print(f"  weighted 32x32 grid e^U(-5,5): kappa={k_:.3g}; max-weight tree st/m={allst.sum()/len(U):.3f}, tau/m={tau_/len(U):.3f}; "
      f"CG its to 1e-3: {iters_to(ec, 1e-3)}, to 1e-6: {iters_to(ec)}; PCG(tree) to 1e-6: {iters_to(ep)}")

print("=" * 70, "\n5. Randomized coordinate descent on quadratics: exact expected one-step contraction")
worst = 0
for trial in range(300):
    N = 8; A = rng.standard_normal((N, N)) * np.exp(rng.uniform(-2, 2, N)); M = A.T @ A + 0.01 * np.eye(N)
    p_ = rng.standard_normal(N); zs = np.linalg.solve(M, p_); F = lambda z: 0.5 * z @ M @ z - p_ @ z
    z = rng.standard_normal(N); g = M @ z - p_; probs = np.diag(M) / np.trace(M)
    EF = sum(probs[i] * F(z - g[i] / M[i, i] * np.eye(N)[i]) for i in range(N))
    ratio = (EF - F(zs)) / (F(z) - F(zs)); bound = 1 - np.linalg.eigvalsh(M)[0] / np.trace(M)
    worst = max(worst, ratio / bound)
print("  max over 300 instances of (E[F+]-F*)/(F-F*) / (1 - mu/trM) =", round(worst, 6), "(must be <= 1)")

print("=" * 70, "\n6. KOSZ: data structure, gap identity, initial energy, convergence")
# 6a. four-point identity / Psi formula vs dense L_T^+
n, U, V, W = grid(8, rng, (-1, 1)); T = bfs_tree(n, U, V, 5); tr = RootedTree(n, U, V, W, T); ds = CentroidDS(tr)
q = rng.standard_normal(n); q -= q.mean()
for v in range(n): ds.add(v, q[v])
psi = np.array([ds.psi(u) for u in range(n)]); LTp = pinv(laplacian(n, U[T], V[T], W[T], dense=True))
lhs = -0.5 * psi; lhs -= lhs.mean()
print("  -1/2 Psi = L_T^+ q (centred):  max error", np.abs(lhs - LTp @ q).max(), "  centroid depth", ds.depth, " log2 n =", round(np.log2(n), 2))
psi_naive = np.array([sum(q[v] * tr.dist(u, v) for v in range(n)) for u in range(n)])
print("  Psi via centroid DS vs naive sum:", np.abs(psi - psi_naive).max())
# 6b. KOSZ with and without the data structure agree; gap identity; energy monotone
n, U, V, W = grid(10, rng, (-1.5, 1.5)); L = laplacian(n, U, V, W, dense=True); chi = np.zeros(n); chi[0], chi[-1] = 1, -1
xs = pinv(L) @ chi; Estar = chi @ xs; T = bfs_tree(n, U, V, 0)
K1 = KOSZ(n, U, V, W, T, chi, np.random.default_rng(1), True); K2 = KOSZ(n, U, V, W, T, chi, np.random.default_rng(1), False)
st_total = sum(K1.tree.dist(U[e], V[e]) * W[e] for e in range(len(U)))
E0 = K1.energy(); print(f"  E(f0)/E* = {E0/Estar:.3f} <= st_T(G) = {st_total:.1f}")
Es = [E0]
for it in range(300):
    i = K1.rng.choice(len(K1.off), p=K1.p); K2.rng.choice(len(K2.off), p=K2.p); K1.step(i); K2.step(i); Es.append(K1.energy())
print("  DS vs naive potentials: max diff", np.abs(K1.potentials() - K2.potentials()).max())
f = K1.flow(); x = K1.potentials()
dual = 2 * chi @ x - x @ L @ x
print("  gap identity: E(f)-D(x_T) =", round(K1.energy() - dual, 12), " sum Delta^2/r =", round(K1.gap(), 12))
print("  ||x_T - x*||_L^2 =", round((x - xs) @ L @ (x - xs), 12), "<= gap; energy monotone:", bool(np.all(np.diff(Es) <= 1e-12)))
print("  flow feasible: |B^T f - chi| =", np.abs(np.bincount(U, f, n) - np.bincount(V, f, n) - chi).max())
# 6c. expected convergence vs (1-1/tau)^k bound, averaged over runs
k_ = 12; n, U, V, W = grid(k_); L = laplacian(n, U, V, W, dense=True); chi = np.zeros(n); chi[0], chi[-1] = 1, -1
xs = pinv(L) @ chi; Estar = chi @ xs
for tname, T in [("BFS", bfs_tree(n, U, V, 0)), ("low-stretch", grid_lowstretch_tree(k_))]:
    runs = []; Kk = None
    for rr in range(40):
        Kk = KOSZ(n, U, V, W, T, chi, np.random.default_rng(100 + rr), True); tr_ = []
        for it in range(int(3 * Kk.tau)):
            if it % int(Kk.tau / 4) == 0: tr_.append(Kk.energy() - Estar)
            Kk.step()
        runs.append(tr_)
    avg = np.mean(runs, 0); E0 = avg[0]; ks = np.arange(len(avg)) * int(Kk.tau / 4)
    bnd = (1 - 1 / Kk.tau) ** ks
    print(f"  grid {k_}x{k_} {tname:11s}: tau={Kk.tau:.0f}, m={len(U)}; avg (E-E*)/(E0-E*) at k=tau,2tau: "
          f"{avg[4]/E0:.2e}, {avg[8]/E0:.2e}  vs bound {bnd[4]:.3f}, {bnd[8]:.3f}")

print("=" * 70, "\n7. Total stretch on grids: BFS vs recursive tree (n log n vs n^1.5)")
for k in [16, 32, 64, 128]:
    n, U, V, W = grid(k)
    out = []
    for T in [bfs_tree(n, U, V, 0), grid_lowstretch_tree(k)]:
        tr = RootedTree(n, U, V, W, T); out.append(sum(tr.dist(U[e], V[e]) for e in range(len(U))) / len(U))
    print(f"  k={k:4d} n={n:6d}: avg stretch BFS {out[0]:8.2f}  recursive {out[1]:6.2f}   (sqrt n = {np.sqrt(n):.0f}, log2 n = {np.log2(n):.1f})")

print("=" * 70, "\n8. Approximate Gaussian elimination (Kyng-Sachdeva)")
k = 20; n, U, V, W = grid(k); L = laplacian(n, U, V, W, dense=True)
w_, Q = np.linalg.eigh(L); Q = Q[:, 1:]; Lih = Q @ np.diag(w_[1:] ** -0.5) @ Q.T
C, dv, tr_ex = approx_cholesky(n, U, V, W, 1, np.random.default_rng(0), exact=True)
Lex = (C @ sp.diags(dv) @ C.T).toarray(); print("  exact elimination reproduces L:", np.abs(Lex - L).max(), " peak #edges", tr_ex.max(), " (m =", len(U), ")")
for rho in [1, 4, 16, 64]:
    evs = []; peaks = []
    for s_ in range(5):
        C, dv, trc = approx_cholesky(n, U, V, W, rho, np.random.default_rng(s_))
        Lap = (C @ sp.diags(dv) @ C.T).toarray(); ev = np.linalg.eigvalsh(Lih @ Lap @ Lih)[1:]; evs.append((ev.min(), ev.max())); peaks.append(trc.max())
    evs = np.array(evs)
    print(f"  rho={rho:3d}: eig(L^-1/2 U^T U L^-1/2) in [{evs[:,0].mean():.3f}, {evs[:,1].mean():.3f}] (avg of 5); peak multi-edges {np.mean(peaks):.0f} = {np.mean(peaks)/len(U)/rho:.2f} rho m")
# unbiasedness of clique sampling for one star
ws = np.array([1., 2., 3., 4.]); dd = ws.sum(); Exp = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        if i != j: Exp[i, j] += ws[j] / dd * ws[i] * ws[j] / (ws[i] + ws[j])
Exp = Exp + Exp.T; Cl = np.outer(ws, ws) / dd; np.fill_diagonal(Cl, 0)
print("  clique sampling unbiased (expected weights = w_i w_j / d):", np.abs(Exp - Cl).max())
