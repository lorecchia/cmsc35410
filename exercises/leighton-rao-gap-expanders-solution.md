# Solution: The Leighton–Rao Integrality Gap on Expanders

## (a) The shortest-path metric

For a $\delta$-regular graph, $\mathrm{vol}(V)=n\delta$, so $c_{ij}=\frac{d_id_j}{\mathrm{vol}(V)}=\frac{\delta^2}{n\delta}=\frac\delta n$ for all pairs. With $\mathrm{dist}=1$ on each of the $n\delta/2$ edges:
$$
\mathrm{dist}(G)=\frac{n\delta}2,\qquad\mathrm{dist}(K_G)=\frac\delta n\sum_{i<j}\mathrm{dist}(i,j)=\frac\delta n\binom n2\bar\Delta=\frac{\delta(n-1)}2\bar\Delta .
$$
So $\dfrac{\mathrm{dist}(G)}{\mathrm{dist}(K_G)}=\dfrac{n\delta/2}{\delta(n-1)\bar\Delta/2}=\dfrac n{(n-1)\bar\Delta}$, and $\mathrm{LR}(G)\le$ this, as $\mathrm{dist}$ is a metric.

## (b) Balls are small

The number of vertices at distance exactly $i\ge1$ from $v$ is at most $\delta(\delta-1)^{i-1}\le\delta^i$, so $|B(v,r)|\le1+\sum_{i=1}^r\delta^i\le\delta^{r+1}$ (for $\delta\ge2$, $1+\sum_{i=1}^r\delta^i=\frac{\delta^{r+1}-1}{\delta-1}\le\delta^{r+1}$).

Let $r_0=\lfloor\log_\delta(n/4)\rfloor-1$. Then $|B(v,r_0)|\le\delta^{r_0+1}\le n/4$, so at least $n-n/4=\frac34n$ vertices are at distance $\ge r_0+1$ from $v$. Averaging over ordered pairs $(v,u)$, $u\ne v$: the fraction of such $u$ is $\ge\frac{3n/4}{n-1}\ge\frac34$, so
$$
\bar\Delta\ \ge\ \tfrac34(r_0+1)\ \ge\ \tfrac34\Big(\log_\delta\frac n4-1\Big).
$$
Hence $\mathrm{LR}(G)\le\frac{n}{n-1}\cdot\frac{4}{3(\log_\delta(n/4)-1)}=O(1/\log_\delta n)$.

## (c) Expanders are well connected

By the easy side of Cheeger, $\lambda\le\bar\phi(G)$ (cut vectors are feasible for the relaxation). If $\lambda\ge c$ then $\bar\phi(G)\ge c$. Combined with (a):
$$
\frac{\bar\phi(G)}{\mathrm{LR}(G)}\ \ge\ \frac{c}{n/((n-1)\bar\Delta)}=\frac{c\,(n-1)\bar\Delta}{n}\ \ge\ \frac{3c}4\cdot\frac{n-1}{n}\Big(\log_\delta\frac n4-1\Big)=\Omega(c\log_\delta n).
$$

## (d) Numerics

```python
import numpy as np, networkx as nx
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh

for delta in [3, 6]:
    for n in [64, 256, 1024, 2048, 4096]:
        G = nx.random_regular_graph(delta, n, seed=7)
        while not nx.is_connected(G): G = nx.random_regular_graph(delta, n, seed=8)
        A = nx.to_scipy_sparse_array(G, format="csr").astype(float)
        D = shortest_path(A, unweighted=True); avg = D.sum()/(n*(n - 1))        # average distance
        mu = sorted(eigsh(A/delta, k=2, which="LA", return_eigenvectors=False))[0]
        lam = 1 - mu; ratio = n/((n - 1)*avg)
        print(delta, n, round(lam, 4), round(avg, 2), round(ratio, 4), round(lam/ratio, 2))
```

| $\delta$ | $n$ | $\lambda$ | $\bar\Delta$ | $\dfrac{n}{(n-1)\bar\Delta}$ | lower bound on $\bar\phi/\mathrm{LR}$ |
|---|---|---|---|---|---|
| 3 | 64 | 0.0854 | 4.18 | 0.2429 | 0.35 |
| 3 | 256 | 0.0740 | 6.05 | 0.1659 | 0.45 |
| 3 | 1024 | 0.0584 | 8.08 | 0.1239 | 0.47 |
| 3 | 2048 | 0.0573 | 9.11 | 0.1098 | 0.52 |
| 3 | 4096 | 0.0586 | 10.09 | 0.0992 | 0.59 |
| 6 | 64 | 0.3070 | 2.47 | 0.4112 | 0.75 |
| 6 | 256 | 0.2751 | 3.34 | 0.3005 | 0.92 |
| 6 | 1024 | 0.2605 | 4.20 | 0.2382 | **1.09** |
| 6 | 2048 | 0.2584 | 4.63 | 0.2163 | **1.19** |
| 6 | 4096 | 0.2555 | 5.06 | 0.1976 | **1.29** |

For $\delta=6$, the lower bound exceeds $1$ already at $n=1024$, and keeps growing like $\log n$ (it increases by about $0.1$ for each doubling of $n$ — each doubling adds $\approx0.45$ to $\bar\Delta\approx\log_5n$). For $\delta=3$, $\lambda$ is much smaller ($\approx0.058$, matching $1-2\sqrt2/3=0.057$), so the constant is poor and the bound would exceed $1$ only for astronomically larger $n$ ($\bar\Delta\gtrsim1/0.058\approx17$). The numbers show the *mechanism*: $\lambda$ is flat, $\bar\Delta$ grows like $\log n$. (Note that $\bar\phi/\mathrm{LR}\ge\lambda/\text{ratio}$ is only a *lower* bound on the gap, since $\mathrm{LR}\le$ ratio and $\bar\phi\ge\lambda$ are both one-sided.)

## (e) What this says about the two relaxations

On bounded-degree expanders, $\lambda$ is a constant, $\bar\phi\ge\lambda$ is a constant, and $\bar\phi\le2\Phi\le2\sqrt{2\lambda}$ by Cheeger: so $\lambda\le\bar\phi\le O(\sqrt\lambda)=O(1)$ with all quantities within constants of each other. The spectral relaxation is **accurate up to a constant**, while $\mathrm{LR}=O(1/\log n)$ is off by $\Omega(\log n)$.

On the cycle, we have the opposite behavior (from the table of the LP exercise: $\mathrm{LR}=\bar\phi=4/n$, and $\lambda\approx2\pi^2/n^2$):

| family | $\bar\phi$ | $\lambda$ (spectral) | $\mathrm{LR}$ | which relaxation is accurate? |
|---|---|---|---|---|
| bounded-degree expander | $\Theta(1)$ | $\Theta(1)$ | $O(1/\log n)$ | spectral ($\lambda\approx\bar\phi$), LR is off by $\Omega(\log n)$ |
| cycle $C_n$ | $4/n$ | $\approx2\pi^2/n^2$ | $4/n$ | LR (exact), spectral is off by a factor $\approx n$ in value ($\approx\sqrt{\bar\phi}$ in the Cheeger sense) |

Each relaxation fails on a different family: the spectral one on long thin graphs (it allows an $\ell_2^2$ solution that violates the triangle inequality), and the LR one on expanders (it allows a metric that is far from $\ell_1$, and there is plenty of room for it to have a small average-case ratio). A relaxation that keeps **both** the $\ell_2^2$ structure and the triangle inequality rules out both bad solutions: this is the SDP of Arora–Rao–Vazirani, which gives $O(\sqrt{\log n})$ (and the Venn diagram of the lecture).
