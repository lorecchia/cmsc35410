# Solution: Certifying Expansion with the Easy Side of Cheeger

## (a) The certificate

Write the Poincaré inequality without normalizations: for every $x$,
$$
\lambda\,\|x-\bar x\mathbf 1\|_d^2\ \le\ \mathcal E(x,x)=\sum_{ij\in E}w_{ij}(x_i-x_j)^2,\qquad\bar x=\frac{\langle x,\mathbf 1\rangle_d}{\mathrm{vol}} .
$$
For $x=\mathbf 1_S$: $\mathcal E=w(S,\bar S)$, $\bar x=\mathrm{vol}(S)/\mathrm{vol}$, and $\|x-\bar x\mathbf 1\|_d^2=\mathrm{vol}(S)\big(1-\tfrac{\mathrm{vol}(S)}{\mathrm{vol}}\big)^2+\mathrm{vol}(\bar S)\big(\tfrac{\mathrm{vol}(S)}{\mathrm{vol}}\big)^2=\dfrac{\mathrm{vol}(S)\mathrm{vol}(\bar S)}{\mathrm{vol}}$. Hence
$$
w(S,\bar S)\ \ge\ \lambda\,\frac{\mathrm{vol}(S)\mathrm{vol}(\bar S)}{\mathrm{vol}}.
$$
If $\mathrm{vol}(S)\le\mathrm{vol}/2$, then $\mathrm{vol}(\bar S)/\mathrm{vol}\ge\frac12$ and $w(S,\bar S)\ge\frac\lambda2\mathrm{vol}(S)$, i.e. $\Phi(S)\ge\lambda/2$.

For a $\delta$-regular graph: every $S$ with $|S|\le n/2$ has
$$
|E(S,\bar S)|\ \ge\ \lambda\,\delta\,|S|\Big(1-\frac{|S|}n\Big)\ \ge\ \frac{\lambda\delta}2|S| .
$$

## (b) How tight is it?

Brute force (Petersen: $n=10$; $Q_4$: $16$; $K_{4,4}$: $8$; $C_{16}$; random $3$-regular, $n=16$ and $n=20$):

| graph | $\lambda$ | $\lambda/2$ | $\Phi(G)$ | $\Phi/(\lambda/2)$ |
|---|---|---|---|---|
| Petersen | 0.6667 | 0.3333 | 0.3333 | 1.00 |
| $Q_4$ | 0.5000 | 0.2500 | 0.2500 | 1.00 |
| $K_{4,4}$ | 1.0000 | 0.5000 | 0.5000 | 1.00 |
| $C_{16}$ | 0.0761 | 0.0381 | 0.1250 | 3.28 |
| random 3-reg, $n=16$ | 0.1435 | 0.0718 | 0.1429 | 1.99 |
| random 3-reg, $n=20$ | 0.0748 | 0.0374 | 0.1111 | 2.97 |

$\Phi=\lambda/2$ exactly for Petersen, $Q_4$ and $K_{4,4}$: these have *cut-like* Fiedler vectors (see (c)): the coordinate characters in $Q_d$, the two sides in $K_{4,4}$, and the eigenspace of Petersen contains $\pm1$-valued vectors on the optimal cut. For the cycle, $\Phi(C_n)=2/n$ while $\lambda/2=\frac12(1-\cos\frac{2\pi}n)\approx\pi^2/n^2$, so $\Phi/(\lambda/2)\approx\frac{2n}{\pi^2}$ **grows linearly with $n$**: the certificate becomes arbitrarily loose, as it must, since the upper side of Cheeger is only tight up to a square root.

## (c) When is the certificate tight?

Let $\mathrm{vol}(S)=\mathrm{vol}/2$, $x=\mathbf 1_S-\mathbf 1_{\bar S}$. Then $\langle x,\mathbf 1\rangle_d=0$, $\|x\|_d^2=\mathrm{vol}$, and $\mathcal E(x,x)=4w(S,\bar S)$. So $R(x)=4w(S,\bar S)/\mathrm{vol}=2\cdot\frac{w(S,\bar S)}{\mathrm{vol}/2}=2\Phi(S)$. If $\Phi(S)=\lambda/2$ then $R(x)=\lambda$, and a vector $x\perp_d\mathbf 1$ whose Rayleigh quotient equals the minimum $\lambda$ is an eigenvector for $\lambda$. For $Q_d$: $S=\{z_i=0\}$ gives $x=\chi_i$, an eigenvector with eigenvalue $2/d=2\Phi$. ✓.

## (d) A certificate needs a lower bound on $\lambda$

For $x\perp_d\mathbf 1$ we have $R(x)\ge\lambda$ by definition of $\lambda$ as a minimum. So the power method produces **upper bounds** on $\lambda$ (and, via the sweep, *cuts*, which are upper bounds on $\Phi$). A small $R(x)$ proves that there is a sparse cut, not that there is none. To prove $\lambda\ge\ell$ one must show that *every* vector has $R\ge\ell$: a statement about the whole matrix.

For all $x$ (not necessarily centered),
$$
x^\top K_\ell x=\mathcal E(x,x)-\ell\Big(\|x\|_d^2-\frac{\langle x,\mathbf 1\rangle_d^2}{\mathrm{vol}}\Big)=\mathcal E(x,x)-\ell\|x-\bar x\mathbf 1\|_d^2,
$$
since $x^\top(D-dd^\top/\mathrm{vol})x=\sum_id_ix_i^2-(\sum_id_ix_i)^2/\mathrm{vol}=\|x-\bar x\mathbf 1\|_d^2$. Since $\mathcal E(x,x)=\mathcal E(x-\bar x\mathbf 1,\cdot)$ and $\lambda=\min\mathcal E(y,y)/\|y\|_d^2$ over $y\perp_d\mathbf 1$: $K_\ell\succeq0\iff\mathcal E(y,y)\ge\ell\|y\|_d^2$ for all $y\perp_d\mathbf 1$ $\iff\lambda\ge\ell$. Also $K_\ell\mathbf 1=L\mathbf 1-\ell(d-d)=0$, so $K_\ell$ is singular, and $K_\ell+\frac1n\mathbf 1\mathbf 1^\top$ (which moves the zero eigenvalue of $\mathbf 1$ to $1$, and does not change anything on $\mathbf 1^\perp$) is positive definite iff $K_\ell\succeq0$ on $\mathbf 1^\perp$ with no further zero eigenvalue, i.e. iff $\lambda>\ell$. A Cholesky factorization $K_\ell+\frac1n\mathbf 1\mathbf 1^\top=CC^\top$ exists iff the matrix is positive definite, and is an easily checkable *proof* that $\lambda>\ell$, hence that $\Phi(G)\ge\ell/2$.

## (e) Numerics

```python
import numpy as np, networkx as nx, math
G = nx.random_regular_graph(3, 200, seed=5); A = nx.to_numpy_array(G); n = 200
d = A.sum(1); vol = d.sum(); L = np.diag(d) - A
S = A/np.sqrt(np.outer(d, d)); lam = np.linalg.eigvalsh(np.eye(n) - S)[1]      # 0.06004

def certified(l):
    K = L - l*(np.diag(d) - np.outer(d, d)/vol) + np.ones((n, n))/n
    try: np.linalg.cholesky(K); return True
    except np.linalg.LinAlgError: return False
```

Here $\lambda=0.06004$, $\lambda/2=0.03002$, $\sqrt{2\lambda}=0.3465$.

| $f$ (with $\ell=f\lambda$) | 0.9 | 0.99 | 0.999 | 1.001 | 1.01 | 1.1 |
|---|---|---|---|---|---|---|
| Cholesky succeeds? | yes | yes | yes | no | no | no |

The Cholesky test flips exactly at $\ell=\lambda$, as the theory says. The Rayleigh quotient of a random mean-zero vector is $0.92$, nowhere near $\lambda=0.06$. The Cholesky computation gives information on the **lower side** of $\lambda$, hence a proof of expansion ($\Phi\ge\ell/2$), while Rayleigh quotients (power method, Lanczos) give information on the **upper side**, hence cuts.

## (f) The interval

With $\ell=0.999\lambda$, the certificate gives $\Phi(G)\ge0.999\cdot0.03002=0.02999$. The sweep cut of the Fiedler vector is a set $S$ with $\Phi(S)\le\sqrt{2\lambda}=0.3465$ (the sweep can be much better; compute it). So $\Phi(G)\in[0.0300,\ \Phi(S_{\text{sweep}})]$, and in the worst case the interval has width ratio $\sqrt{2\lambda}/(\lambda/2)=2\sqrt{2/\lambda}=0.3465/0.0300\approx11.6$. This is the **two-sided certificate** of the algorithm slide: the algorithm both finds a cut and proves how far it is from optimal, with ratio at most $2\sqrt{2/\lambda}$, which is large for graphs with small gap, and a constant for expanders.
