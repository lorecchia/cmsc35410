"""Numerical core for Lecture 9 (Laplacian solvers). Imported by verify_bounds.py and make_figures.py.

Conventions (as on the slides)
  graph: n vertices, edge arrays U, V (oriented U[e] -> V[e]), conductances W, resistances r = 1/W
  b_e = e_u - e_v,  L = sum_e w_e b_e b_e^T,  flow f routes chi iff  B^T f = chi  (net outflow = chi)
  energy E(f) = sum_e r_e f_e^2,  electrical potentials x* = L^+ chi,  f*_e = w_e (x_u - x_v)
"""
import numpy as np
from collections import deque
import scipy.sparse as sp

# ----------------------------------------------------------------- graphs
def grid(k, rng=None, wrange=None):
    U, V = [], []
    for i in range(k):
        for j in range(k):
            u = i * k + j
            if j + 1 < k: U.append(u); V.append(u + 1)
            if i + 1 < k: U.append(u); V.append(u + k)
    U, V = np.array(U), np.array(V)
    W = np.ones(len(U)) if wrange is None else np.exp(rng.uniform(*wrange, len(U)))
    return k * k, U, V, W

def path(n):
    return n, np.arange(n - 1), np.arange(1, n), np.ones(n - 1)

def cycle(n):
    return n, np.arange(n), (np.arange(n) + 1) % n, np.ones(n)

def random_regular(n, d, rng):
    import networkx as nx
    G = nx.random_regular_graph(d, n, seed=int(rng.integers(1 << 30)))
    while not nx.is_connected(G):
        G = nx.random_regular_graph(d, n, seed=int(rng.integers(1 << 30)))
    E = np.array(G.edges())
    return n, E[:, 0], E[:, 1], np.ones(len(E))

def laplacian(n, U, V, W, dense=False):
    L = sp.coo_matrix((np.r_[W, W, -W, -W], (np.r_[U, V, U, V], np.r_[U, V, V, U])), shape=(n, n)).tocsr()
    return L.toarray() if dense else L

def adjacency_lists(n, U, V):
    adj = [[] for _ in range(n)]
    for e, (u, v) in enumerate(zip(U, V)):
        adj[u].append((v, e)); adj[v].append((u, e))
    return adj

# ----------------------------------------------------------------- spanning trees
def bfs_tree(n, U, V, root=0):
    adj = adjacency_lists(n, U, V); seen = np.zeros(n, bool); seen[root] = True
    T = []; q = deque([root])
    while q:
        u = q.popleft()
        for v, e in adj[u]:
            if not seen[v]: seen[v] = True; T.append(e); q.append(v)
    return np.array(T)

def max_weight_tree(n, U, V, W):
    """Maximum-conductance spanning tree (= minimum spanning tree w.r.t. resistances), Kruskal."""
    par = list(range(n))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    T = []
    for e in np.argsort(-W):
        a, b = find(U[e]), find(V[e])
        if a != b: par[a] = b; T.append(e)
    return np.array(T)

def grid_lowstretch_tree(k):
    """Recursive 'cross' tree on the k x k grid: middle row + middle column of each block, recurse into
    the four quadrants, attach each quadrant by one edge. Total stretch O(n log n) (vs Theta(n^1.5) for BFS)."""
    idx = {}
    n, U, V, W = grid(k)
    for e, (u, v) in enumerate(zip(U, V)): idx[(u, v)] = e; idx[(v, u)] = e
    T = []
    def ed(a, b): T.append(idx[(a[0] * k + a[1], b[0] * k + b[1])])
    def build(r0, r1, c0, c1):
        if r1 <= r0 or c1 <= c0: return
        if r1 - r0 <= 2 or c1 - c0 <= 2:                      # comb: rows + first column
            for r in range(r0, r1):
                for c in range(c0, c1 - 1): ed((r, c), (r, c + 1))
            for r in range(r0, r1 - 1): ed((r, c0), (r + 1, c0))
            return
        rm, cm = (r0 + r1) // 2, (c0 + c1) // 2
        for c in range(c0, c1 - 1): ed((rm, c), (rm, c + 1))
        for r in range(r0, r1 - 1):
            if r != rm - 1 and r != rm: ed((r, cm), (r + 1, cm))
        if rm - 1 >= r0: ed((rm - 1, cm), (rm, cm))
        if rm + 1 < r1: ed((rm, cm), (rm + 1, cm))
        for (a0, a1, b0, b1, conn) in [(r0, rm, c0, cm, ((rm - 1, cm - 1), (rm, cm - 1))),
                                       (r0, rm, cm + 1, c1, ((rm - 1, cm + 1), (rm, cm + 1))),
                                       (rm + 1, r1, c0, cm, ((rm + 1, cm - 1), (rm, cm - 1))),
                                       (rm + 1, r1, cm + 1, c1, ((rm + 1, cm + 1), (rm, cm + 1)))]:
            if a1 > a0 and b1 > b0: build(a0, a1, b0, b1); ed(*conn)
    build(0, k, 0, k)
    T = np.array(sorted(set(T)))
    assert len(T) == n - 1
    return T

class RootedTree:
    """Static spanning tree: parent pointers, resistance-depths, LCA by climbing (fine at demo sizes)."""
    def __init__(self, n, U, V, W, T, root=0):
        self.n = n; self.r = 1.0 / W
        adj = [[] for _ in range(n)]
        for e in T: adj[U[e]].append((V[e], e)); adj[V[e]].append((U[e], e))
        self.adj = adj
        self.par = -np.ones(n, int); self.pe = -np.ones(n, int); self.dep = np.zeros(n, int); self.h = np.zeros(n)
        order = [root]; seen = np.zeros(n, bool); seen[root] = True
        for u in order:
            for v, e in adj[u]:
                if not seen[v]:
                    seen[v] = True; self.par[v] = u; self.pe[v] = e
                    self.dep[v] = self.dep[u] + 1; self.h[v] = self.h[u] + self.r[e]; order.append(v)
        assert len(order) == n, "T is not spanning"
        self.order = np.array(order); self.root = root
        self.U, self.V = U, V

    def lca(self, a, b):
        while self.dep[a] > self.dep[b]: a = self.par[a]
        while self.dep[b] > self.dep[a]: b = self.par[b]
        while a != b: a = self.par[a]; b = self.par[b]
        return a

    def dist(self, a, b):                                   # tree resistance R_T(a,b)
        return self.h[a] + self.h[b] - 2 * self.h[self.lca(a, b)]

    def path_edges(self, a, b):
        c = self.lca(a, b); P = []
        while a != c: P.append(self.pe[a]); a = self.par[a]
        Q = []
        while b != c: Q.append(self.pe[b]); b = self.par[b]
        return P + Q[::-1]

    def solve(self, chi):
        """O(n) tree solve: unique tree flow routing chi, and potentials x with x_root = 0 (then centred)."""
        s = chi.astype(float).copy(); fup = np.zeros(self.n)      # flow child -> parent on edge pe[child]
        for v in self.order[::-1]:
            if v != self.root: fup[v] = s[v]; s[self.par[v]] += s[v]
        x = np.zeros(self.n)
        for v in self.order[1:]: x[v] = x[self.par[v]] + self.r[self.pe[v]] * fup[v]
        return x - x.mean(), fup

def stretches(tree, U, V, W, offtree):
    return np.array([tree.dist(U[e], V[e]) * W[e] for e in offtree])

# ----------------------------------------------------------------- classical iterative methods
def L_err(L, x, xs):
    d = x - xs; return np.sqrt(max(d @ (L @ d), 0.0))

def gd(L, b, xs, iters, eta):
    x = np.zeros_like(b); out = [L_err(L, x, xs)]
    for _ in range(iters):
        x = x - eta * (L @ x - b); out.append(L_err(L, x, xs))
    return np.array(out)

def pcg(L, b, xs, iters, prec=None):
    x = np.zeros_like(b); r = b - L @ x
    z = prec(r) if prec else r.copy(); p = z.copy(); rz = r @ z; out = [L_err(L, x, xs)]
    for _ in range(iters):
        Lp = L @ p; pLp = p @ Lp
        if pLp <= 0 or rz <= 0: break
        a = rz / pLp; x = x + a * p; r = r - a * Lp
        z = prec(r) if prec else r.copy(); rz2 = r @ z
        p = z + (rz2 / rz) * p; rz = rz2; out.append(L_err(L, x, xs))
        if out[-1] < 1e-13 * out[0]: break
    return np.array(out)

# ----------------------------------------------------------------- KOSZ data structure
class CentroidDS:
    """Maintains vertex charges q (sum q = 0 whenever queried) and answers
        Psi(u) = sum_v q_v R_T(u, v)
    in O(depth) = O(log n) time per add/query, via a centroid decomposition of the static tree."""
    def __init__(self, tree):
        n = tree.n; adj = tree.adj; r = tree.r
        removed = np.zeros(n, bool); self.anc = [[] for _ in range(n)]    # per vertex: [(centroid, R(u,c))], top-down
        self.cpar = -np.ones(n, int)
        stack = [(tree.root, -1)]
        while stack:
            s, cp = stack.pop()
            comp = [s]; par = {s: -1}                                   # collect component of s
            for u in comp:
                for v, e in adj[u]:
                    if not removed[v] and v != par[u]: par[v] = u; comp.append(v)
            size = {u: 1 for u in comp}
            for u in comp[::-1]:
                if par[u] != -1: size[par[u]] += size[u]
            N = len(comp); c = s                                        # walk to centroid
            while True:
                nxt = None
                for v, e in adj[c]:
                    if not removed[v] and v != par[c] and size[v] > N // 2: nxt = v; break
                if nxt is None: break
                c = nxt
            self.cpar[c] = cp
            dist = {c: 0.0}; q = [c]                                    # distances from c inside the component
            for u in q:
                for v, e in adj[u]:
                    if not removed[v] and v not in dist: dist[v] = dist[u] + r[e]; q.append(v)
            for u in comp: self.anc[u].append((c, dist[u]))
            removed[c] = True
            for v, e in adj[c]:
                if not removed[v]: stack.append((v, c))
        for u in range(n): self.anc[u].reverse()                        # bottom-up: anc[u][0] = (u, 0)
        self.anc = [[(int(c), float(d)) for c, d in A] for A in self.anc]
        self.S = [0.0] * n; self.D = [0.0] * n; self.Dp = [0.0] * n     # python lists: fast scalar access
        self.depth = max(len(a) for a in self.anc); self.touches = 0

    def add(self, a, alpha):
        A = self.anc[a]
        for i, (c, d) in enumerate(A):
            self.S[c] += alpha; self.D[c] += alpha * d
            if i + 1 < len(A): self.Dp[c] += alpha * A[i + 1][1]
        self.touches += len(A)

    def psi(self, u):
        A = self.anc[u]; tot = 0.0; prev = None
        for (c, d) in A:
            Sc, Dc = self.S[c], self.D[c]
            if prev is not None: Sc -= self.S[prev]; Dc -= self.Dp[prev]
            tot += Dc + d * Sc; prev = c
        self.touches += len(A)
        return tot

class KOSZ:
    """Cycle toggling = randomized coordinate descent on the (rescaled) cycle space."""
    def __init__(self, n, U, V, W, T, chi, rng, use_ds=True):
        self.n, self.U, self.V, self.W, self.r = n, U, V, W, 1.0 / W
        self.tree = RootedTree(n, U, V, W, T)
        inT = np.zeros(len(U), bool); inT[T] = True
        self.off = np.where(~inT)[0]; self.T = T
        self.chi = chi; self.rng = rng
        self.x0, self.fup0 = self.tree.solve(chi)                      # tree flow f_0 and its potentials
        self.RT = np.array([self.tree.dist(U[e], V[e]) for e in self.off])
        self.Rcyc = self.r[self.off] + self.RT                          # cycle resistances R_e
        self.tau_e = self.Rcyc / self.r[self.off]                       # 1 + stretch
        self.tau = self.tau_e.sum(); self.p = self.tau_e / self.tau
        self.foff = np.zeros(len(self.off))                              # flow on off-tree edges
        self.q = np.zeros(n)                                             # charges pushed into the tree
        self.use_ds = use_ds
        if use_ds: self.ds = CentroidDS(self.tree)
        self.steps = 0

    def potential(self, u):                                              # tree-induced potential (up to a constant)
        if self.use_ds: return self.x0[u] - 0.5 * self.ds.psi(u)
        return self.x0[u] - 0.5 * sum(self.q[v] * self.tree.dist(u, v) for v in np.nonzero(self.q)[0])

    def sample(self):
        if not getattr(self, '_buf', None): self._buf = list(self.rng.choice(len(self.off), size=20000, p=self.p))[::-1]
        return self._buf.pop()

    def step(self, i=None):
        if i is None: i = self.sample()
        e = self.off[i]; a, b = self.U[e], self.V[e]
        delta = self.r[e] * self.foff[i] + self.potential(b) - self.potential(a)   # KVL violation around C_e
        alpha = -delta / self.Rcyc[i]
        self.foff[i] += alpha
        self.q[b] += alpha; self.q[a] -= alpha                          # push alpha along tree path b -> a
        if self.use_ds: self.ds.add(b, alpha); self.ds.add(a, -alpha)
        self.steps += 1
        return delta

    def flow(self):
        """Full flow vector (tree part recomputed from chi minus off-tree divergence)."""
        f = np.zeros(len(self.U)); f[self.off] = self.foff
        dem = self.chi.astype(float).copy()
        np.add.at(dem, self.U[self.off], -self.foff); np.add.at(dem, self.V[self.off], self.foff)
        _, fup = self.tree.solve(dem)
        for v in range(self.n):
            if v != self.tree.root:
                e = self.tree.pe[v]; f[e] = fup[v] if self.U[e] == v else -fup[v]
        return f

    def potentials(self):
        """All tree potentials (for measuring the error). Not charged to the solver's work counter."""
        t0 = self.ds.touches if self.use_ds else 0
        x = np.array([self.potential(u) for u in range(self.n)])
        if self.use_ds: self.ds.touches = t0
        return x - x.mean()

    def energy(self):
        f = self.flow(); return f @ (self.r * f)

    def gap(self):
        """E(f) - D(x_T) = sum over off-tree e of Delta_e^2 / r_e."""
        x = self.potentials(); tot = 0.0
        for i, e in enumerate(self.off):
            d = self.r[e] * self.foff[i] + x[self.V[e]] - x[self.U[e]]; tot += d * d / self.r[e]
        return tot

# ----------------------------------------------------------------- Kyng-Sachdeva approximate Cholesky
def approx_cholesky(n, U, V, W, rho, rng, exact=False):
    """Eliminate vertices in random order. exact=True: true Schur complements (fill-in).
    exact=False: KS16 clique sampling on rho-fold split multi-edges. Returns (C, d, edgecount trace)
    with  L_approx = sum_v d_v c_v c_v^T,  c_v = e_v - sum_u (w_vu/d_v) e_u."""
    nbr = [dict() for _ in range(n)] if exact else [[] for _ in range(n)]
    for u, v, w in zip(U, V, W):
        if exact:
            nbr[u][v] = nbr[u].get(v, 0) + w; nbr[v][u] = nbr[v].get(u, 0) + w
        else:
            for _ in range(rho): nbr[u].append((v, w / rho)); nbr[v].append((u, w / rho))
    alive = np.ones(n, bool); order = rng.permutation(n)
    rows, cols, vals, dvec = [], [], [], np.zeros(n); trace = []
    m_now = len(U) if exact else rho * len(U)
    for v in order[:-1]:
        alive[v] = False
        if exact:
            E = [(u, w) for u, w in nbr[v].items() if alive[u]]
        else:
            E = [(u, w) for (u, w) in nbr[v] if alive[u]]
        d = sum(w for _, w in E); dvec[v] = d
        rows.append(v); cols.append(v); vals.append(1.0)
        if d == 0: trace.append(m_now); continue
        agg = {}
        for u, w in E: agg[u] = agg.get(u, 0) + w
        for u, w in agg.items(): rows.append(u); cols.append(v); vals.append(-w / d)
        m_now -= len(E)
        if exact:
            for u in agg: nbr[u].pop(v, None)
            us = list(agg)
            for i in range(len(us)):
                for j in range(i + 1, len(us)):
                    a, b = us[i], us[j]; w = agg[a] * agg[b] / d
                    if b not in nbr[a]: m_now += 1
                    nbr[a][b] = nbr[a].get(b, 0) + w; nbr[b][a] = nbr[b].get(a, 0) + w
        else:
            ws = np.array([w for _, w in E]); us = [u for u, _ in E]
            pick = rng.choice(len(E), size=len(E), p=ws / d)
            for i, j in enumerate(pick):
                a, b = us[i], us[j]
                if a == b: continue
                w = ws[i] * ws[j] / (ws[i] + ws[j])
                nbr[a].append((b, w)); nbr[b].append((a, w)); m_now += 1
        trace.append(m_now)
    last = order[-1]; rows.append(last); cols.append(last); vals.append(1.0)
    C = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
    return C, dvec, np.array(trace)
