<!-- Moved from lectures/lecture09/lecture9.qmd on 2026-10-28 for the end of Lecture 12 (spectral sparsification).
Figure paths are relative to lecture09: copy figures/approx-elimination.svg or fix the path.
Also moved: Lecture 9 takeaway and discussion question below. -->

# Other routes to nearly-linear time

## Spielman–Teng and recursive preconditioning

::: {.defn}
**Ultra-sparsifier.** $H$ = spanning tree + $\tilde O(m/k)$ extra edges, with $\ L_H\preceq L_G\preceq k\,L_H$.
:::

**One level of the recursion:**

1. PCG (or Chebyshev) on $G$ with preconditioner $H$: $O(\sqrt k)$ iterations.
2. To solve with $L_H$: **eliminate** vertices of degree 1 and 2. A degree-1 vertex is a leaf; a degree-2 vertex is an edge in series ($r\leftarrow r_1+r_2$). No fill-in. What remains has $\tilde O(m/k)$ vertices and edges.
3. Solve the remaining system **recursively**, approximately.

- Work: $T(m)\approx\sqrt k\,\big(m+T(\tilde O(m/k))\big)$, nearly linear for a suitable $k=\mathrm{polylog}(n)$.
- Ingredients: low-stretch trees and **spectral sparsifiers** (this lecture).

| | running time ($\times\log(1/\varepsilon)$) |
|:--|:--|
| Spielman–Teng 2004 | $m\log^{c}n$ |
| Koutis–Miller–Peng 2010, 2011 | $m\log^2n$, then $m\log n\,(\log\log n)^2$ |
| Kelner–Orecchia–Sidford–Zhu 2013 | $m\log^2n\,\log\log n$ (no recursion) |
| Lee–Sidford 2013 | $m\log^{3/2}n\,\sqrt{\log\log n}$ |
| Cohen–Kyng–Miller–Pachocki–Peng–Rao–Xu 2014 | $m\sqrt{\log n}\,(\log\log n)^{O(1)}$ |
| Kyng–Sachdeva 2016 | $m\log^3n$ (sampling only) |

::: notes
~4 min. The degree-1 and degree-2 eliminations are exact Gaussian elimination steps that create no new edges: this is "partial Cholesky". The table is a guide, not a precise statement: polyloglog factors and the exact dependence on ε vary between papers (KOSZ has log(n/ε) in our analysis). KMP: Koutis–Miller–Peng, "Approaching optimality..." (FOCS 2010) and "A nearly m log n time solver..." (FOCS 2011).
:::

## Gaussian elimination is a graph operation

Eliminating vertex $v$ (one step of Cholesky) writes $L=d_v\,c_vc_v^\top+L'$, with $c_v=\mathbf 1_v-\sum_u\frac{w_{vu}}{d_v}\mathbf 1_u$, and the **Schur complement**
$$L'=L-\mathrm{star}(v)+\mathrm{clique}(v),\qquad \mathrm{clique}(v)=\sum_{\{u,u'\}\subseteq N(v)}\frac{w_{vu}w_{vu'}}{d_v}\,b_{uu'}b_{uu'}^\top .$$

- **The Schur complement of a Laplacian is a Laplacian**, on $V-v$. It is the electrical network seen from $V-v$ (star–mesh transform).
- Eliminating all vertices gives $L=\mathcal L\mathcal L^\top$ exactly. The cost is the **fill-in**: a vertex of degree $d$ creates up to $\binom d2$ edges.
- Trees (degree 1) and series edges (degree 2) create no fill. Grids do: nested dissection gives $O(n^{1.5})$ time for planar graphs (George 1973; Lipton–Rose–Tarjan 1979); expanders are hopeless.

::: notes
~3 min. Check of the clique weights: d_v c_v c_v^T has off-diagonal entries w_vu w_vu'/d_v, which must be added back to keep the star's contribution correct. Star–mesh: eliminating the center of a 3-star is the Y–Δ transform from physics.
:::

## Approximate Gaussian elimination (Kyng–Sachdeva 2016)

::: {.defn}
**Clique sampling.** When eliminating $v$, for each (multi-)edge $(v,u)$ sample one neighbor $u'$ with probability $w_{vu'}/d_v$ and add the edge $(u,u')$ with weight $\dfrac{w_{vu}w_{vu'}}{w_{vu}+w_{vu'}}$ (nothing if $u'=u$).
:::

- **Unbiased:** the expected weight on $\{u,u'\}$ is $\frac{w_{vu'}}{d_v}\cdot\frac{w_{vu}w_{vu'}}{w_{vu}+w_{vu'}}+\frac{w_{vu}}{d_v}\cdot\frac{w_{vu}w_{vu'}}{w_{vu}+w_{vu'}}=\frac{w_{vu}w_{vu'}}{d_v}$.
- **Sparse:** eliminating $v$ removes $\deg(v)$ edges and adds at most $\deg(v)$. **The number of edges never grows.**
- **Concentrated:** split every edge into $\rho=O(\log^2n)$ copies and eliminate in a **uniformly random order**. Every sample is then a small piece relative to $L$, and the errors form a **matrix martingale** (matrix Freedman). W.h.p. the result $\tilde L=\tilde{\mathcal L}\tilde{\mathcal L}^\top$ satisfies $\ \tfrac12L\preceq\tilde L\preceq\tfrac32L$.
- Factorization in $O(m\log^3n)$ time, then PCG: $O(m\log^3n\log(1/\varepsilon))$. No trees, no sparsifiers, no expanders.

![](figures/approx-elimination.svg){width="88%"}

<p class="small muted center">$24\times24$ grid, random order. Exact elimination peaks at $18m$ edges. With $\rho=16$, $\ \mathrm{spec}(L^{+/2}\tilde LL^{+/2})\subseteq[0.70,1.30]$; with $\rho=64$, $[0.84,1.18]$ (average over 4 runs).</p>

::: notes
~5 min. The split copies are what make every sample have leverage score ≤ 1/ρ, exactly the condition for matrix concentration in Lecture 8 (matrix Bernstein) and this lecture (sparsification). The random order keeps expected degrees small: a random remaining vertex has expected degree 2·(edges)/(vertices). Modern practice: Gao–Kyng–Spielman (2023) "ApproxChol" in Laplacians.jl, a robust variant of this algorithm used as a PCG preconditioner. Numbers from make_figures.py, Fig. 4; unbiasedness checked in scripts §8.
:::


<!-- Takeaway (was L9 #6) -->
- **Elimination:** the Schur complement of a Laplacian is a Laplacian; sample the cliques to keep it sparse (Kyng–Sachdeva).

<!-- Discussion question (was L9 #5) -->
- Why does Kyng–Sachdeva need a **random** elimination order? Construct an order that makes clique sampling fail on a star.
