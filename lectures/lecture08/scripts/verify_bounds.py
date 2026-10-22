"""Numerical sanity checks for the Lecture 8 bounds (run: python verify_bounds.py)."""
import numpy as np, itertools
rng = np.random.default_rng(0)

def topk(M, k):
    w, U = np.linalg.eigh(M); return w[::-1], U[:, ::-1][:, :k]

def sin_theta(V, Vh):
    s = np.linalg.svd(Vh.T @ V, compute_uv=False)
    s = np.clip(s, 0, 1); sin = np.sqrt(1 - s**2)
    return sin.max(), np.sqrt((sin**2).sum())

def procrustes(V, Vh):
    U, _, Wt = np.linalg.svd(Vh.T @ V); return U @ Wt

# ---------- 1. Davis-Kahan on random instances ----------
worst = 0
for t in range(2000):
    n, k = 8, 3
    A = rng.standard_normal((n, n)); A = A + A.T
    E = 0.3 * rng.standard_normal((n, n)); E = E + E.T
    w, V = topk(A, k); delta0 = w[k-1] - w[k]
    _, Vh = topk(A + E, k)
    s, sF = sin_theta(V, Vh)
    worst = max(worst, s / (2*np.linalg.norm(E, 2)/delta0))
    O = procrustes(V, Vh)
    assert np.linalg.norm(Vh@O - V) <= np.sqrt(2)*sF + 1e-9
print("DK Cor 3.3: max ratio actual/bound =", round(worst, 3))

# ---------- 2. 2x2 sharpness example ----------
for eps in [0.01, 0.1, 0.3]:
    d = 1.0; A = np.diag([d, 0.]); E = eps*np.array([[0, 1], [1, 0.]])
    w, U = np.linalg.eigh(A+E); v = U[:, -1]
    sin = abs(v[1]); disc = w[0]; dpert = d - disc
    print(f"2x2 eps={eps}: sin={sin:.5f}  eps/delta_pert={eps/dpert:.5f}  eps/delta0={eps/d:.5f}")

# ---------- 3. Planted clusters: self-loop trick ----------
def expander(m, rng, deg=6):
    W = np.zeros((m, m))
    for _ in range(deg//2):
        p = rng.permutation(m)
        for i in range(m): a, b = p[i], p[(i+1) % m]; W[a, b] += 1; W[b, a] += 1
    return W

def planted(k, m, ncross, mode, rng):
    n = k*m; W = np.zeros((n, n)); lab = np.repeat(np.arange(k), m)
    for j in range(k): W[j*m:(j+1)*m, j*m:(j+1)*m] = expander(m, rng)
    for _ in range(ncross):
        j, l = rng.choice(k, 2, replace=False)
        if mode == 'spread': u = j*m + rng.integers(m); v = l*m + rng.integers(m)
        else: u = j*m + rng.integers(2); v = l*m + rng.integers(2)   # hubs: 2 per cluster
        W[u, v] += 1; W[v, u] += 1
    return W, lab

def analyze_planted(W, lab, k):
    n = len(W); d = W.sum(1); Dm = np.diag(d**-0.5)
    same = lab[:, None] == lab[None, :]
    Aout = W * (~same); dout = Aout.sum(1)
    A0 = W*same + np.diag(dout)
    N = Dm @ W @ Dm; N0 = Dm @ A0 @ Dm; E = N - N0
    Lout = np.diag(dout) - Aout
    assert np.allclose(E, -Dm @ Lout @ Dm)
    normE = np.linalg.norm(E, 2); eps_out = (dout/d).max()
    assert normE <= 2*eps_out + 1e-12
    w0 = np.linalg.eigvalsh(N0)[::-1]; delta0 = w0[k-1] - w0[k]
    # delta0 = min_j lambda_2 of block with loops
    lam2 = []
    for j in range(k):
        idx = lab == j; Nj = N0[np.ix_(idx, idx)]; lam2.append(1 - np.linalg.eigvalsh(Nj)[-2])
    assert abs(delta0 - min(lam2)) < 1e-9 and abs(w0[k-1]-1) < 1e-9
    vol = np.array([d[lab == j].sum() for j in range(k)])
    V = np.zeros((n, k))
    for j in range(k): V[lab == j, j] = np.sqrt(d[lab == j]/vol[j])
    wN, Vh = topk(N, k); mu_k1 = 1 - wN[k]
    s, sF = sin_theta(V, Vh)
    Phi = np.array([Aout[lab == j].sum()/vol[j] for j in range(k)])
    # energy bound check: chi_j^T (I-N) chi_j = Phi(S_j)
    for j in range(k): assert abs(V[:, j] @ (V[:, j] - N @ V[:, j]) - Phi[j]) < 1e-9
    energy = np.sqrt(Phi.sum()/mu_k1)
    dkF = min(1, 2*np.sqrt(k)*normE/delta0) if normE < delta0/2 else np.nan
    # bad-node bound (degree-weighted, random-walk embedding)
    O = procrustes(V, Vh); Y = Dm @ Vh @ O; X = Dm @ V
    err2 = ((Y - X)**2).sum(1)
    bad = err2 >= 1/(4*vol[lab])
    frac = sum(d[bad & (lab == j)].sum()/vol[j] for j in range(k))
    assert frac <= 4*np.linalg.norm(Vh@O - V)**2 + 1e-9
    return dict(eps_out=eps_out, normE=normE, delta0=delta0, sinF=sF, DK_F=dkF,
                DK_F_eps=2*np.sqrt(k)*2*eps_out/delta0, energy=energy, frac_bad=frac)

for mode in ['spread', 'hubs']:
    for nc in [5, 20, 60]:
        W, lab = planted(3, 60, nc, mode, np.random.default_rng(1))
        r = analyze_planted(W, lab, 3)
        assert r['sinF'] <= r['energy'] + 1e-9
        print(mode, nc, {a: round(b, 4) for a, b in r.items()})

# ---------- 4. SBM ----------
def sbm(n, k, p, q, rng):
    lab = np.repeat(np.arange(k), n//k)
    P = np.where(lab[:, None] == lab[None, :], p, q); np.fill_diagonal(P, 0)
    U = rng.random((n, n)); A = np.triu((U < P).astype(float), 1); A = A + A.T
    return A, P, lab

def kmeans_err(Y, lab, k, rng, restarts=5):
    best = None
    for _ in range(restarts):
        C = Y[rng.choice(len(Y), k, replace=False)]
        for _ in range(50):
            z = ((Y[:, None]-C[None])**2).sum(2).argmin(1)
            C = np.array([Y[z == j].mean(0) if (z == j).any() else C[j] for j in range(k)])
        cost = ((Y-C[z])**2).sum()
        if best is None or cost < best[0]: best = (cost, z)
    z = best[1]
    return min((np.array(pm)[z] != lab).sum() for pm in itertools.permutations(range(k)))

n, k = 400, 2
for p, q in [(0.10, 0.02), (0.06, 0.03), (0.04, 0.03)]:
    A, P, lab = sbm(n, k, p, q, np.random.default_rng(2))
    E = A - P; normE = np.linalg.norm(E, 2)
    dmax = P.sum(1).max(); bern = 2*np.sqrt(dmax*np.log(n)) + 4/3*np.log(n)
    w = np.linalg.eigvalsh(P)[::-1]; delta0 = w[k-1]-w[k]
    pred = [n*(p-q)/k + n*q - p, n*(p-q)/k - p]
    V = np.zeros((n, k)); 
    for j in range(k): V[lab == j, j] = np.sqrt(k/n)
    _, Vh = topk(A, k); s, sF = sin_theta(V, Vh)
    O = procrustes(V, Vh); err2 = ((Vh@O - V)**2).sum(1); Delta2 = 2*k/n
    bad = (err2 >= Delta2/4).sum(); badbound = 16*n*normE**2/delta0**2
    km = kmeans_err(Vh, lab, k, np.random.default_rng(0))
    print(f"SBM p={p} q={q}: eig {w[:3].round(2)} pred {np.round(pred,2)} delta0={delta0:.2f} (n(p-q)/k={n*(p-q)/k:.2f})"
          f" |E|={normE:.2f} 2sqrt(d)={2*np.sqrt(dmax):.2f} Bernstein={bern:.2f} sin={s:.3f} DK={2*normE/delta0:.3f}"
          f" bad={bad} bound={badbound:.0f} kmeans_err={km}")
