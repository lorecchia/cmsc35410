# Solution: The Cut Cone and Hypermetric Inequalities

## (a) Points on a line

Sorted, the points are $x_1\le\dots\le x_5$ with gaps $0.7,\ 0.9,\ 0.2,\ 1.4$. With $S_t=\{1,\dots,t\}$,
$$
|x_i-x_j|=0.7\,\delta^{S_1}_{ij}+0.9\,\delta^{S_2}_{ij}+0.2\,\delta^{S_3}_{ij}+1.4\,\delta^{S_4}_{ij}.
$$
(For $i<j$, $\delta^{S_t}_{ij}=1$ iff $i\le t<j$, so the right side is $\sum_{t=i}^{j-1}(x_{t+1}-x_t)=x_j-x_i$.) In general $n-1$ cuts suffice, and they are nested.

## (b) Plane and cycle

$\|p_i-p_j\|_1=|p_i^{(1)}-p_j^{(1)}|+|p_i^{(2)}-p_j^{(2)}|$, and each coordinate is a line metric, which by (a) is a sum of $\le n-1$ cut metrics. Hence at most $2(n-1)$ cuts.

For $C_4$ with vertices $v_0,v_1,v_2,v_3$ in cyclic order, map $v_0\mapsto00$, $v_1\mapsto01$, $v_2\mapsto11$, $v_3\mapsto10$. The Hamming distance equals $d_{C_4}$ on all pairs (adjacent pairs differ in one bit, antipodal pairs in two). The first coordinate is the cut $\{v_0,v_1\}|\{v_2,v_3\}$, the second is $\{v_0,v_3\}|\{v_1,v_2\}$. So $d_{C_4}=\delta^{\{v_0,v_1\}}+\delta^{\{v_0,v_3\}}$. (The same works for all cycles: $d_{C_n}$ is a sum of the $n/2$ "diameter" cuts, for even $n$.)

## (c) Hypermetric inequalities

For a fixed cut $S$, put $b(S)=\sum_{i\in S}b_i\in\mathbb Z$, so $b(\bar S)=1-b(S)$. Then
$$
\sum_{i<j}b_ib_j\,\delta^S_{ij}=\sum_{i\in S,\,j\notin S}b_ib_j=b(S)\,b(\bar S)=b(S)\big(1-b(S)\big).
$$
For an integer $x$, $x(1-x)\le0$ (it is $0$ at $x\in\{0,1\}$ and negative otherwise). So the inequality holds for every cut metric, and, being linear in $d$, for every nonnegative combination of cut metrics, i.e. for every $\ell_1$ metric.

For $b=(1,1,-1)$: $\sum b_ib_jd_{ij}=d_{12}-d_{13}-d_{23}\le0$, which is the triangle inequality. The hypermetric inequalities are a family of generalizations of the triangle inequality (which is why all $\ell_1$ metrics are metrics), and they are *strictly* stronger.

## (d) $K_{2,3}$ and $K_{3,3}$

For $K_{2,3}$, $d=1$ on the six edges $a_ib_j$, $d=2$ for $a_1a_2$ and for the three pairs $b_ib_j$. With $b=+1$ on $b_1,b_2,b_3$ and $-1$ on $a_1,a_2$ (so $\sum b_i=3-2=1$):

| pairs | count | $b_ib_j$ | $d$ | contribution |
|---|---|---|---|---|
| $b$–$b$ | 3 | $+1$ | 2 | $+6$ |
| $a$–$a$ | 1 | $+1$ | 2 | $+2$ |
| $a$–$b$ | 6 | $-1$ | 1 | $-6$ |

Total: $+2>0$. This violates the hypermetric inequality, so $d_{K_{2,3}}\notin\mathrm{CUT}_V$ and is not $\ell_1$-embeddable.

For $K_{3,3}$, take $b=+1$ on $a_1,a_2,a_3$, $-1$ on $b_1,b_2$, $0$ on $b_3$ ($\sum b_i=1$). The pairs: $a$–$a$ (3 pairs, $d=2$, $b_ib_j=+1$): $+6$; $b_1b_2$ ($d=2$, $+1$): $+2$; $a$–$b_{1,2}$ (6 pairs, $d=1$, $b_ib_j=-1$): $-6$; pairs involving $b_3$ contribute $0$. Total $+2>0$, so $d_{K_{3,3}}$ is not $\ell_1$ either.

**Consistency with the LP exercise.** There, the shortest-path metric of $K_{2,3}$ had ratio $12/13<\bar\phi=36/35$, which by the remark after Lemma 1 is only possible if the metric is not $\ell_1$. The hypermetric inequality proves that fact directly, with no cut enumeration.

## (e) Computation

```python
import numpy as np, itertools, networkx as nx
from scipy.optimize import linprog

def in_cut_cone(D):
    n = len(D); pairs = list(itertools.combinations(range(n), 2))
    cuts = range(1, 2**(n - 1))            # each cut once (vertex n-1 always on the 0 side)
    M = np.array([[float(((m >> i) & 1) != ((m >> j) & 1)) for m in cuts] for i, j in pairs])
    b = np.array([D[i][j] for i, j in pairs])
    return linprog(np.zeros(M.shape[1]), A_eq=M, b_eq=b, bounds=(0, None), method="highs").status == 0

def sp_metric(G):
    n = G.number_of_nodes(); L = dict(nx.all_pairs_shortest_path_length(G))
    return [[L[i][j] for j in range(n)] for i in range(n)]
```

Results:

| graph | in the cut cone? |
|---|---|
| $C_5$ | yes |
| $C_6$ | yes |
| $K_4$ | yes |
| Petersen | yes |
| $K_{2,3}$ | **no** |
| $K_{3,3}$ | **no** |

This agrees with (b) (cycles are $\ell_1$) and (d). It also agrees with the table of the Leighton–Rao exercise: $\mathrm{LR}=\bar\phi$ for $K_4$, the cycles and Petersen, and $\mathrm{LR}<\bar\phi$ for $K_{2,3}$ and $K_{3,3}$ (the shortest-path metric being optimal there). The Petersen graph is a good example of a metric that is $\ell_1$ although it does not embed isometrically into a hypercube.

## (f) Why this matters for rounding

If the optimal LP metric $d^*$ lies in $\mathrm{CUT}_V$, then $d^*=\sum_S\alpha_S\delta^S$ with $\alpha\ge0$, and the mediant argument of Lemma 1 gives a cut $S$ with $\bar\phi(S)\le d^*(G)/d^*(K_G)=\mathrm{LR}(G)$: rounding is **lossless**, and $\mathrm{LR}=\bar\phi$. If $d^*\notin\mathrm{CUT}_V$ (as for $K_{2,3}$), the argument does not apply, and in fact there may be no cut with $\bar\phi(S)\le\mathrm{LR}(G)$ (an *integrality gap*). One must first replace $d^*$ by a nearby $\ell_1$ metric: Bourgain's theorem gives $e\in\mathrm{CUT}_V$ with $d^*\le e\le O(\log n)\,d^*$ pairwise, and then applying the lossless rounding to $e$ costs an extra $O(\log n)$ factor. The hypermetric inequalities (and the more refined negative-type inequalities that the SDP of ARV adds) are exactly the constraints that cut away some of the non-$\ell_1$ metrics.
