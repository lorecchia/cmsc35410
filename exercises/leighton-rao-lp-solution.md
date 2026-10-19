# Solution: Leighton–Rao as a Linear Program

Notation as in the statement: $c_{ij}=d_id_j/\mathrm{vol}(V)$, $d(G)=\sum_Ew_{ij}d_{ij}$, $d(K_G)=\sum_{i<j}c_{ij}d_{ij}$.

## (a) It is a relaxation

For a cut $S$: $\delta^S(G)=w(S,\bar S)$ and $\delta^S(K_G)=\sum_{i\in S,j\notin S}\frac{d_id_j}{\mathrm{vol}(V)}=\frac{\mathrm{vol}(S)\mathrm{vol}(\bar S)}{\mathrm{vol}(V)}$. So
$$
\frac{\delta^S(G)}{\delta^S(K_G)}=\frac{w(S,\bar S)\,\mathrm{vol}(V)}{\mathrm{vol}(S)\mathrm{vol}(\bar S)}=\bar\phi(S).
$$
**Triangle inequality.** $\delta^S_{ij}\le\delta^S_{ik}+\delta^S_{kj}$: if $\delta^S_{ij}=1$, then $i,j$ are on different sides, and $k$ is on a different side from at least one of them, so the right side is $\ge1$. If $\delta^S_{ij}=0$ there is nothing to prove.

So $\delta^S/\delta^S(K_G)$ is feasible, with value $\bar\phi(S)$, and $\mathrm{LR}(G)\le\min_S\bar\phi(S)=\bar\phi(G)$.

**Ratio form.** The metrics form a cone, and $d(G),d(K_G)$ are linear. Any metric $d$ with $d(K_G)>0$ can be scaled to $d/d(K_G)$, which is feasible with objective $d(G)/d(K_G)$. A metric with $d(K_G)=0$ is zero on all pairs (all $c_{ij}>0$), so it is excluded. Hence $\mathrm{LR}=\min_dd(G)/d(K_G)$.

## (b) $K_{2,3}$

Vertices $a_1,a_2$ (degree $3$), $b_1,b_2,b_3$ (degree $2$), $\mathrm{vol}(V)=12$, $6$ edges.

**1. $\bar\phi(G)$.** Using $\bar\phi(S)=\dfrac{12\,w(S,\bar S)}{\mathrm{vol}(S)\mathrm{vol}(\bar S)}$:

| $S$ | cut | $\mathrm{vol}(S)$ | $\bar\phi(S)$ |
|---|---|---|---|
| $\{a_1\}$ | 3 | 3 | $\frac{36}{27}=1.333$ |
| $\{b_1\}$ | 2 | 2 | $\frac{24}{20}=1.2$ |
| $\{a_1,a_2\}$ | 6 | 6 | $\frac{72}{36}=2$ |
| $\{a_1,b_1\}$ | 3 | 5 | $\frac{36}{35}\approx1.0286$ |
| $\{b_1,b_2\}$ | 4 | 4 | $\frac{48}{32}=1.5$ |

(the remaining cut types are complements of these, or are worse), so $\bar\phi(G)=36/35$ (checked by brute force below).

**2. Shortest-path metric.** $d(G)=6\cdot1=6$. Pairs, with weights $c_{ij}=d_id_j/12$:

- $a$–$a$: one pair, $c=9/12$, $d=2$: contributes $1.5$;
- $b$–$b$: three pairs, $c=4/12$, $d=2$: contributes $3\cdot\frac13\cdot2=2$;
- $a$–$b$: six pairs, $c=6/12$, $d=1$: contributes $6\cdot\frac12=3$.

So $d(K_G)=6.5$ and $d(G)/d(K_G)=\dfrac{6}{6.5}=\dfrac{12}{13}\approx0.923<\dfrac{36}{35}$. Hence $\mathrm{LR}(G)\le12/13<\bar\phi(G)$. (The LP optimum is exactly $12/13$, see the table.)

**3. Not $\ell_1$.** If $d$ were $\ell_1$-embeddable, the remark after Lemma 1 would give a cut with $\bar\phi(S)\le d(G)/d(K_G)=12/13<36/35=\bar\phi(G)$, a contradiction.

## (c) Solving the LP

```python
import numpy as np, itertools, networkx as nx
from scipy.optimize import linprog

def stats(G):
    n = G.number_of_nodes(); nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes); d = A.sum(1); vol = d.sum()
    S = A/np.sqrt(np.outer(d, d)); lam = np.linalg.eigvalsh(np.eye(n) - S)[1]   # lambda_2(L_rw)
    Phi = phibar = np.inf
    for mask in range(1, 2**(n-1)):                       # brute force over cuts
        x = np.array([(mask >> i) & 1 for i in range(n)], float)
        cut = x @ A @ (1 - x); v = x @ d
        phibar = min(phibar, cut*vol/(v*(vol - v))); Phi = min(Phi, cut/min(v, vol - v))
    pairs = list(itertools.combinations(range(n), 2)); pid = {p: i for i, p in enumerate(pairs)}
    P = lambda i, j: pid[(min(i, j), max(i, j))]
    obj = np.array([A[i, j] for i, j in pairs])           # d(G)
    eq = np.array([d[i]*d[j]/vol for i, j in pairs])      # d(K_G)
    rows = []
    for i, j, k in itertools.permutations(range(n), 3):
        if j < k:
            r = np.zeros(len(pairs)); r[P(j, k)] = 1; r[P(i, j)] -= 1; r[P(i, k)] -= 1
            rows.append(r)                                 # d_jk <= d_ij + d_ik
    res = linprog(obj, A_ub=np.array(rows), b_ub=np.zeros(len(rows)),
                  A_eq=eq[None, :], b_eq=[1], bounds=(0, None), method="highs")
    return lam, Phi, phibar, res.fun

graphs = {"K_4": nx.complete_graph(4), "C_8": nx.cycle_graph(8), "C_10": nx.cycle_graph(10),
          "Q_3": nx.hypercube_graph(3), "tree h=3": nx.balanced_tree(2, 3),
          "Petersen": nx.petersen_graph(), "barbell": nx.barbell_graph(4, 0),
          "K_{2,3}": nx.complete_bipartite_graph(2, 3), "K_{3,3}": nx.complete_bipartite_graph(3, 3)}
for name, G in graphs.items(): print(name, *np.round(stats(G), 4))
```

| graph | $\lambda$ | $\Phi$ | $\bar\phi$ | $\mathrm{LR}$ |
|---|---|---|---|---|
| $K_4$ | 1.3333 | 0.6667 | 1.3333 | 1.3333 |
| $C_8$ | 0.2929 | 0.2500 | 0.5000 | 0.5000 |
| $C_{10}$ | 0.1910 | 0.2000 | 0.4000 | 0.4000 |
| $Q_3$ | 0.6667 | 0.3333 | 0.6667 | 0.6667 |
| binary tree, $h=3$ | 0.0572 | 0.0769 | 0.1436 | 0.1436 |
| Petersen | 0.6667 | 0.3333 | 0.6667 | 0.6667 |
| barbell | 0.1134 | 0.0769 | 0.1538 | 0.1538 |
| $K_{2,3}$ | 1.0000 | 0.6000 | 1.0286 | **0.9231** |
| $K_{3,3}$ | 1.0000 | 0.5556 | 1.0000 | **0.8571** |

**Observations.**

- $\mathrm{LR}=\bar\phi$ on the complete graph, the cycles, the cube, the tree, Petersen and the barbell. The relaxation is strictly smaller on the complete bipartite graphs: $12/13<36/35$ for $K_{2,3}$ (matches (b)) and $6/7<1$ for $K_{3,3}$.
- $\lambda$ and $\mathrm{LR}$ are **incomparable**. On $C_{10}$, $\lambda=0.19<\mathrm{LR}=0.40$: the spectral relaxation is weaker (it allows $\ell_2^2$ solutions that violate the triangle inequality). On $K_{2,3}$ and $K_{3,3}$, $\lambda>\mathrm{LR}$: forbidding non-$\ell_2^2$ metrics can help. Both are $\le\bar\phi$ (cut metrics are in both feasible sets). This is the Venn diagram of slide 18, with the intersection (metrics that are also $\ell_2^2$) being the feasible set of ARV.
- Also: $\bar\phi(C_n)=4/n$ for even $n$, from the half-cycle cut.

## (d) Trees

**Step 1.** Let $d$ be an optimal metric, with ratio $r=d(G)/d(K_G)$. Let $\ell_{ij}=d_{ij}$ for $ij\in E$ and let $\hat d$ be the shortest-path metric of $\ell$ on $G$. For every pair $u,v$ and every path $u=x_0,\dots,x_m=v$ in $G$, the triangle inequality for $d$ gives $d_{uv}\le\sum_td_{x_{t-1}x_t}=\sum_t\ell_{x_{t-1}x_t}$; minimizing over paths, $d_{uv}\le\hat d_{uv}$. On edges, $\hat d_{ij}\le\ell_{ij}=d_{ij}$. Hence
$$
\hat d(G)\le d(G),\qquad \hat d(K_G)\ge d(K_G),\qquad\text{so}\qquad \frac{\hat d(G)}{\hat d(K_G)}\le r .
$$
Since $r$ is the minimum, $\hat d$ is also optimal (and a metric).

**Step 2.** If $G$ is a tree and $e$ is an edge, removing $e$ splits $V$ into $S_e,\bar S_e$. The unique $u$–$v$ path uses $e$ iff $S_e$ separates $u$ and $v$. So for any edge lengths,
$$
\hat d_{uv}=\sum_{e\in\text{path}(u,v)}\ell_e=\sum_{e\in E}\ell_e\,\delta^{S_e}_{uv},
$$
a nonnegative combination of cut metrics, i.e. $\hat d$ is $\ell_1$-embeddable. By the remark after Lemma 1 there is a cut $S$ with $\bar\phi(S)\le\hat d(G)/\hat d(K_G)=\mathrm{LR}(G)$. Together with (a): $\mathrm{LR}(G)=\bar\phi(G)$. $\square$

(The same argument works whenever the shortest-path metric of *every* choice of edge lengths is $\ell_1$, e.g. for cycles. It fails for $K_{2,3}$ by (b).)

## (e) The dual is a flow problem

**1. Weak duality.** Suppose a flow sends $\alpha c_{uv}$ units between every pair, with congestion $\le w_e$ on each edge $e$. For a cut $S$, every unit of flow between a separated pair crosses an edge of $(S,\bar S)$. The total amount of such flow is $\alpha\sum_{i\in S,j\notin S}c_{ij}=\alpha\,\delta^S(K_G)$, and the total capacity of the cut is $w(S,\bar S)=\delta^S(G)$. So $\alpha\,\delta^S(K_G)\le\delta^S(G)$, that is, $\alpha\le\bar\phi(S)$ for every $S$. Hence $\alpha^*\le\bar\phi(G)$.

**2. Strong duality.** Let $\mathcal P_{uv}$ be the set of $u$–$v$ paths. The flow problem is
$$
\max\ \alpha\quad\text{s.t.}\quad\sum_{P\ni e}f_P\le w_e\ \ (e\in E),\qquad\alpha c_{uv}-\sum_{P\in\mathcal P_{uv}}f_P\le0\ \ (u<v),\qquad f\ge0.
$$
Attach multipliers $\ell_e\ge0$ to the first group and $z_{uv}\ge0$ to the second. The dual is
$$
\min\ \sum_ew_e\ell_e\quad\text{s.t.}\quad\sum_{u<v}c_{uv}z_{uv}\ge1,\qquad \sum_{e\in P}\ell_e\ge z_{uv}\ \ (P\in\mathcal P_{uv}),\qquad\ell,z\ge0.
$$
(The coefficient of $\alpha$ gives the first constraint, the coefficient of $f_P$ gives the second.) For fixed $\ell$, the best choice of $z$ is $z_{uv}=\mathrm{dist}_\ell(u,v)$ (the shortest-path distance, largest allowed value, which makes the first constraint easiest to satisfy). So the dual is
$$
\min_{\ell\ge0}\Big\{\textstyle\sum_ew_e\ell_e:\ \ \sum c_{uv}\mathrm{dist}_\ell(u,v)\ge1\Big\}=\min_\ell\frac{\sum_ew_e\ell_e}{\sum c_{uv}\mathrm{dist}_\ell(u,v)}
$$
(by homogeneity, the constraint is tight at optimum). By Step 1 of (d), the minimum over *metrics* $d$ of $d(G)/d(K_G)$ is attained at some $\hat d=\mathrm{dist}_\ell$, and conversely for every $\ell$ the metric $\mathrm{dist}_\ell$ has $\mathrm{dist}_\ell(G)\le\sum_ew_e\ell_e$. So the dual value equals $\min_d d(G)/d(K_G)=\mathrm{LR}(G)$. The primal is feasible and bounded, so strong duality gives $\alpha^*=\mathrm{LR}(G)$.

**3. Canonical paths.** Route $c_{uv}$ units from $u$ to $v$ along a fixed path $\gamma_{uv}$, and let
$$
\kappa'=\max_{e\in E}\ \frac1{w_e}\sum_{uv:\ e\in\gamma_{uv}}c_{uv}
$$
be the resulting congestion. Scaling the flow by $1/\kappa'$ gives a feasible concurrent flow with $\alpha=1/\kappa'$, so by 1, $\bar\phi(G)\ge 1/\kappa'$: **every cut is large**, with no eigenvalue involved. The Lecture 5 certificate, $\lambda_2(L)\ge2n/\kappa$, bounds the *spectral* relaxation; to turn it into a statement about cuts we then use the easy side of Cheeger, $\Phi\ge\lambda/2$. Here the same kind of data (a routing of one unit between every pair) bounds $\bar\phi$ directly, by a one-line cut argument, with no eigenvalue involved. The two certificates differ in how they are paid for in the other direction: the spectral one is rounded with a square-root loss in the value (Cheeger), the flow one with an $O(\log n)$ loss (part (f)). Note also that the LP *optimizes* the routing, and allows splitting flow over many paths.

## (f) Rounding

Let $d^*$ be an optimal LP solution. By Bourgain, there is $\psi:V\to\ell_1^m$ with $d^*_{ij}\le\|\psi(i)-\psi(j)\|_1\le O(\log n)\,d^*_{ij}$ (and we may take the embedded metric $e_{ij}=\|\psi(i)-\psi(j)\|_1$). Then
$$
\frac{e(G)}{e(K_G)}\le\frac{O(\log n)\,d^*(G)}{d^*(K_G)}=O(\log n)\,\mathrm{LR}(G),
$$
since $e(G)\le O(\log n)d^*(G)$ and $e(K_G)\ge d^*(K_G)$. By the remark after Lemma 1, $e$, being $\ell_1$, has a cut $S$ with $\bar\phi(S)\le e(G)/e(K_G)$. So $\bar\phi(S)\le O(\log n)\,\mathrm{LR}(G)\le O(\log n)\,\bar\phi(G)$, and $S$ can be found by sweeping each coordinate of $\psi$.

**Contrast with Cheeger.** In the Cheeger proof, the squaring map $x\mapsto x^2$ is *not* an embedding with bounded distortion on every pair. It only needs to preserve the single ratio $R_1/R_2$, and Cauchy–Schwarz gives that up to a square root of the *value*. Bourgain needs distortion on every pair (because the LP metric $d^*$ is arbitrary), and loses a $\log n$ in the *dimension* instead. This is why the spectral rounding has no $\log n$ but loses $\sqrt\lambda$, and LR has no square root but loses $\log n$ (ARV improves it to $\sqrt{\log n}$ by using the structure that both constraints give together).
