# Solution: Which Eigenvector? The Hypercube

## (a) The spectrum

$L_{rw}=I-\frac1dA$, and
$$
(A\chi_S)(z)=\sum_{i=1}^d\chi_S(z\oplus e_i)=\chi_S(z)\sum_{i=1}^d\epsilon_i,\qquad\epsilon_i=\begin{cases}-1&i\in S\\+1&i\notin S\end{cases},
$$
because flipping bit $i$ multiplies $\chi_S$ by $-1$ iff $i\in S$. So $A\chi_S=(d-2|S|)\chi_S$ and $L_{rw}\chi_S=\frac{2|S|}d\chi_S$. The $2^d$ characters are pairwise orthogonal ($\sum_z\chi_S(z)\chi_T(z)=2^d\mathbf 1[S=T]$), so they form a basis. The smallest eigenvalue is $0$ ($S=\emptyset$), the next is $2/d$, with eigenspace spanned by $\chi_1,\dots,\chi_d$ (dimension $d$). So $\lambda=2/d$.

## (b) The good eigenvector

$\chi_i(z)=1-2z_i$ takes two values, so the only nontrivial threshold set is $\{z_i=0\}$ (or its complement $\{z_i=1\}$). It has $n/2$ vertices, $\mathrm{vol}=dn/2$, and $n/2$ cut edges (those in direction $i$). So
$$
\Phi=\frac{n/2}{dn/2}=\frac1d.
$$
By the easy side, $\Phi(Q_d)\ge\lambda/2=1/d$. Hence $\Phi(Q_d)=1/d=\lambda/2$: the lower side of Cheeger is tight.

## (c) The bad eigenvector

$f=\sum_i(1-2z_i)=d-2|z|$ is a function of the Hamming weight only, so its threshold sets are $\{|z|<s\}=B_{s-1}$ and complements. Take $r<d/2$. The edges leaving $B_r$ go from weight $r$ to weight $r+1$, and each weight-$r$ vertex has $d-r$ up-neighbors: $\binom dr(d-r)$ cut edges. $\mathrm{vol}(B_r)=d\sum_{j\le r}\binom dj\le d2^{d-1}=\mathrm{vol}(V)/2$. So
$$
\Phi(B_r)=\frac{\binom dr(d-r)}{d\sum_{j\le r}\binom dj}.
$$

## (d) Every sweep cut of $f$ is bad

**Regime 1: $d-2r\ge\sqrt d$.** For $j\le r$, $\binom d{j-1}/\binom dj=\frac j{d-j+1}\le\frac r{d-r+1}=:q<1$, so $\sum_{j\le r}\binom dj\le\binom dr\sum_{i\ge0}q^i=\binom dr\frac{d-r+1}{d-2r+1}$. Hence
$$
\Phi(B_r)\ge\frac{(d-r)(d-2r+1)}{d(d-r+1)}\ge\frac{d-2r+1}{2d}\ge\frac1{2\sqrt d},
$$
using $\frac{d-r}{d-r+1}\ge\frac12$.

**Regime 2: $d-2r<\sqrt d$**, i.e. $|r-d/2|\le\sqrt d/2$. In this window $\binom dr\ge c_02^d/\sqrt d$ for an absolute constant $c_0>0$ (the binomial coefficients are within a constant factor of the central one, which is $\sim2^d\sqrt{2/(\pi d)}$), $d-r\ge d/2$, and $\sum_{j\le r}\binom dj\le2^{d-1}$. So
$$
\Phi(B_r)\ge\frac{(d/2)\,c_02^d/\sqrt d}{d\,2^{d-1}}=\frac{c_0}{\sqrt d}.
$$
So every sweep cut of $f$ has conductance $\ge c/\sqrt d$ with $c=\min\{\frac12,c_0\}$, while $\Phi(Q_d)=1/d$. In terms of $\Phi=\Phi(Q_d)$: $\Phi(S)\ge c\sqrt{\Phi}$.

**Exact asymptotics.** At $r\approx d/2$: $\binom dr\approx 2^d\sqrt{2/(\pi d)}$, $d-r\approx d/2$, $\sum_{j\le r}\binom dj\approx2^{d-1}$, so
$$
\Phi(B_{\lfloor d/2\rfloor})\approx\frac{(d/2)\,2^d\sqrt{2/(\pi d)}}{d\,2^{d-1}}=\sqrt{\frac{2}{\pi d}}\approx\frac{0.80}{\sqrt d}.
$$
The best ball (computed exactly) satisfies $\sqrt d\cdot\min_r\Phi(B_r)=1.20,1.11,1.06,1.03,0.99,0.96,0.93,0.89$ for $d=4,6,8,10,14,20,30,50$, slowly approaching $\sqrt{2/\pi}=0.80$. Compare with Cheeger's guarantee $\sqrt{2\lambda}=2/\sqrt d$: the rounding $\approx0.8/\sqrt d$ is within a factor $2.5$ of the worst the theorem allows.

## (e) Numerics

```python
import numpy as np, math

def sweep_cube(d, g):
    """Best threshold-set conductance of f(z) = sum_i g_i (1 - 2 z_i) on Q_d."""
    n = 2**d; z = (np.arange(n)[:, None] >> np.arange(d)) & 1
    f = ((1 - 2*z) * g).sum(1)
    order = np.argsort(f); pos = np.empty(n, int); pos[order] = np.arange(n)
    delta = np.zeros(n + 1)
    for b in range(d):                       # difference array: edge cut by prefixes t in (a, c]
        u = np.arange(n); v = u ^ (1 << b); m = u < v
        a = np.minimum(pos[u[m]], pos[v[m]]); c = np.maximum(pos[u[m]], pos[v[m]])
        np.add.at(delta, a + 1, 1); np.add.at(delta, c + 1, -1)
    cut = np.cumsum(delta)[1:n]; t = np.arange(1, n)
    return (cut / (d*np.minimum(t, n - t))).min()
```

For (2), ties must be handled: $f=d-2|z|$ has only $d+1$ distinct values, so use the formula of (c) (or skip prefix boundaries inside ties). With exact ball sets:

| $d$ | $1/d$ | $\chi_1$ | balls (all-ones) | Gaussian $g$ (mean of 10) | $1/\sqrt d$ |
|---|---|---|---|---|---|
| 8 | 0.125 | 0.125 | 0.376 | 0.233 | 0.354 |
| 10 | 0.100 | 0.100 | 0.326 | 0.200 | 0.316 |
| 12 | 0.083 | 0.083 | 0.291* | 0.175 | 0.289 |
| 14 | 0.071 | 0.071 | 0.265 | 0.175 | 0.267 |

(*computed from the formula in (c).) The Gaussian vectors give cuts that are typically $2$–$3\times$ larger than $1/d$ (there is large sample-to-sample variation: individual samples ranged from $1.1\times$ to $2.6\times$ the optimum at $d=12$), and the gap to $1/d$ widens with $d$. The all-ones vector is the worst case of this family, and the Gaussian ones fall between.

(One caveat: if one adds tiny noise to the all-ones vector to break ties, the prefix sets are no longer balls, and sorted-prefix sweeps can find better cuts, e.g. $0.246$ instead of $0.326$ at $d=10$. The definition of the sweep in Lecture 6 uses threshold sets $\{f>t\}$ precisely to avoid depending on tie-breaking.)

## (f) Discussion

In the tree $\square$ path graph, two *different* bottlenecks have *nearby* eigenvalues, so the Fiedler vector can be the wrong one, and no choice of the (unique) eigenvector fixes it. On the cube, there are $d$ equally good directions with *exactly* equal eigenvalues, and the good eigenvectors (the coordinate ones) are a measure-zero set in the eigenspace. The eigenvalue $\lambda_2$ is blind to both.

A power method started from a random $x_0$ converges to the projection of $x_0$ onto the eigenspace, i.e. $\sum_i\langle x_0,\chi_i\rangle\chi_i$, a vector with essentially Gaussian coefficients. So it lands in the typical, not the best, part of the eigenspace, and the sweep loses a factor growing like $\sqrt d$. On the cube, the symmetry is what makes this unavoidable: *all* $d$ directions are equivalent, and the best cut breaks the symmetry. (In practice one can recover by running the sweep from several different Fiedler-eigenspace vectors, or by using more eigenvectors, which leads to the higher-order Cheeger inequality.)
