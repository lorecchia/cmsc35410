# Solution: Cheeger with an Approximate Eigenvector

## (a) The sweep needs only a small quotient

Run the proof of Cheeger with $f:=x\perp_d\mathbf 1$:

- **Lemma 3** gives $f'\ge0$ with $\pi(\{f'>0\})\le\frac12$ and $R_2(f')\le R_2(x)$. Its proof (shift by the $\pi$-median $m$, so $\sum_id_i\bar f_i^2=\sum_id_ix_i^2+m^2\mathrm{vol}(V)\ge\sum_id_ix_i^2$, then split into positive and negative parts) uses only $\sum_id_ix_i=0$.
- **Lemma 2** gives $R_1(g)\le\sqrt{2R_2(f')}$ for $g=f'^2$.
- **Lemma 1** gives a threshold set $S_t$ of $g$, which is a threshold set of $f'$, hence of $x$ or $-x$ (the positive part of $\bar f$ or of $-\bar f$ is a function of $x$ composed with a monotone map), with $w(S_t,\bar S_t)/\mathrm{vol}(S_t)\le R_1(g)$.

Since $\pi(S_t)\le\frac12$, $\Phi(S_t)=w(S_t,\bar S_t)/\mathrm{vol}(S_t)$, and so $\Phi(S_t)\le\sqrt{2R_2(x)}=\sqrt{2R(x)}$.

The eigenvector property of $f$ is **never used**: the proof uses only $f\perp_d\mathbf 1$ and the value of the quotient. The eigenvector only enters through the *value* $R_2(f)=\lambda$, which is the smallest possible.

## (b) Any nonconstant vector

Let $\tilde x=x-\bar x\mathbf 1$. Then $\tilde x\perp_d\mathbf 1$ ($\bar x$ is the $d$-weighted mean), $\mathcal E(\tilde x)=\mathcal E(x)$ (differences are unchanged), $\|\tilde x\|_d^2=\mathrm{vol}(V)\mathrm{Var}_\pi(x)$, and threshold sets of $\tilde x$ are threshold sets of $x$ with a shifted threshold. Apply (a) to $\tilde x$:
$$
\Phi(S_t)\le\sqrt{\frac{2\mathcal E(x)}{\mathrm{vol}(V)\mathrm{Var}_\pi(x)}} .
$$
This is the **Rayleigh–Poincaré quotient of $x$** appearing in the Poincaré inequality: *every* nonconstant function certifies a cut of conductance $\le\sqrt{2\cdot\text{(its quotient)}}$.

## (c) The power method

Write $y=M^tx_0=\sum_{i\ge2}\mu_i^ta_iu_i$ with $\mu_i=1-\lambda_i/2\in[0,1]$. Then
$$
\mathcal E(y)=\sum_i\lambda_i\mu_i^{2t}a_i^2,\qquad\|y\|_d^2=\sum_i\mu_i^{2t}a_i^2,\qquad R(y)=\frac{\sum_i\lambda_i\mu_i^{2t}a_i^2}{\sum_i\mu_i^{2t}a_i^2}.
$$
Let $B=\{i:\lambda_i>(1+\varepsilon)\lambda\}$, $G=\{i\ge2\}\setminus B$, and $\Sigma_B=\sum_{i\in B}\mu_i^{2t}a_i^2$, $\Sigma_G=\sum_{i\in G}\mu_i^{2t}a_i^2$. Since $\lambda_i\le(1+\varepsilon)\lambda$ on $G$ and $\lambda_i\le2$ on $B$,
$$
R(y)\le\frac{(1+\varepsilon)\lambda\,\Sigma_G+2\Sigma_B}{\Sigma_G+\Sigma_B}\le(1+\varepsilon)\lambda+2\,\frac{\Sigma_B}{\Sigma_G}.
$$
For $i\in B$: $\mu_i\le1-(1+\varepsilon)\lambda/2$, and
$$
\frac{\mu_i}{\mu_2}\le\frac{1-(1+\varepsilon)\lambda/2}{1-\lambda/2}=1-\frac{\varepsilon\lambda/2}{1-\lambda/2}\le1-\frac{\varepsilon\lambda}2,
$$
so $\mu_i^{2t}\le\mu_2^{2t}e^{-\varepsilon\lambda t}$, and $\Sigma_B\le e^{-\varepsilon\lambda t}\mu_2^{2t}\sum_ia_i^2$. Also $\Sigma_G\ge\mu_2^{2t}a_2^2\ge\mu_2^{2t}\delta\sum_ia_i^2$. Hence $\Sigma_B/\Sigma_G\le e^{-\varepsilon\lambda t}/\delta$, which is the claimed
$$
R(y)\le(1+\varepsilon)\lambda+\frac2\delta e^{-\varepsilon\lambda t}.
$$
If $t\ge\frac1{\varepsilon\lambda}\ln\frac2{\delta\varepsilon\lambda}$, then $\frac2\delta e^{-\varepsilon\lambda t}\le\varepsilon\lambda$, so $R(y)\le(1+2\varepsilon)\lambda$. By (a), some sweep set of $y$ has $\Phi(S)\le\sqrt{2(1+2\varepsilon)\lambda}$.

**Typical $\delta$.** If $x_0$ is Gaussian (after projecting out $\mathbf 1$), its coefficients $a_i$ in an orthonormal basis are i.i.d. Gaussian, and $a_2^2/\sum_ia_i^2$ is of order $1/n$ with constant probability (and $\ge1/(\mathrm{poly}(n))$ with high probability). So $t=O\!\big(\frac{\log(n/\varepsilon\lambda)}{\varepsilon\lambda}\big)$, i.e. $\tilde O(1/\lambda)$ iterations, each costing $O(m)$.

## (d) Numerics: planted bisection

```python
import numpy as np, math

def run(A, ts, seed=0):
    n = len(A); d = A.sum(1); vol = d.sum()
    S = A/np.sqrt(np.outer(d, d)); lam = np.linalg.eigvalsh(np.eye(n) - S)[1]
    M = 0.5*(np.eye(n) + A/d[:, None]); rng = np.random.default_rng(seed)
    proj = lambda x: x - (d @ x)/vol
    cur = proj(rng.standard_normal(n)); done = 0; out = []
    for t in ts:
        for _ in range(t - done): cur = proj(M @ cur)
        done = t
        R = ((d*cur - A @ cur) @ cur)/(d @ cur**2)
        order = np.argsort(cur); fs = cur[order]; Ao = A[np.ix_(order, order)]
        inner = np.cumsum(np.cumsum(Ao, 0), 1).diagonal(); cv = np.cumsum(d[order]); cut = cv - inner
        best = min(cut[k-1]/min(cv[k-1], vol - cv[k-1]) for k in range(1, n) if fs[k] - fs[k-1] > 1e-13)
        out.append((t, R/lam, best, math.sqrt(2*R)))
    return lam, out
```

For the graph generated with `rng = default_rng(3)`, blocks of $60$, $p_{in}=0.35$, $p_{out}=0.012$ (and $x_0$ from `default_rng(0)`): $\lambda=0.0836$.

| $t$ | $R(x_t)/\lambda$ | best sweep $\Phi$ | $\sqrt{2R(x_t)}$ |
|---|---|---|---|
| 0 | 11.8 | 0.499 | 1.402 |
| 1 | 11.0 | 0.476 | 1.357 |
| 2 | 10.4 | 0.455 | 1.319 |
| 5 | 7.9 | 0.398 | 1.146 |
| 10 | 1.37 | 0.0475 | 0.478 |
| 20 | 1.00 | 0.0475 | 0.409 |
| 50 | 1.00 | 0.0475 | 0.409 |

The planted cut has $\Phi=0.0475$, and $\lambda/2=0.042$, $\sqrt{2\lambda}=0.409$. Already at $t=10$ ($R=1.37\lambda$) the sweep recovers the planted bisection, far below the guarantee $0.478$. The theorem is a worst-case bound, and the sweep is much better than it on a graph with a genuine planted cut. Before the iteration count $\approx1/\lambda\approx12$, the quotient is far from $\lambda$ and the sweep is poor, which is consistent with (c).

## (e) Numerics: the cycle

$\lambda(C_{100})=1-\cos(2\pi/100)=0.001973$, so $1/\lambda\approx507$. Results:

| $t$ | $R/\lambda$ | best sweep $\Phi$ | $\sqrt{2R}$ |
|---|---|---|---|
| 0 | 455.7 | 0.479 | 1.341 |
| 10 | 23.9 | 0.091 | 0.307 |
| 100 | 2.78 | 0.0238 | 0.105 |
| 300 | 1.36 | 0.0200 | 0.073 |
| 1000 | 1.01 | 0.0200 | 0.063 |
| 3000 | 1.00 | 0.0200 | 0.063 |

The half-cycle, with $\Phi=2/n=0.02$, is found from $t\approx300$, which is $\approx0.6/\lambda$ iterations. On the planted graph, $\lambda=0.084$ and $\approx10$ iterations suffice. The ratio of the two iteration counts is $\approx30$, comparable to $\lambda_{\text{planted}}/\lambda_{C_{100}}\approx42$. This is the slide's message: **the power method needs $\approx1/\lambda$ steps**, and the graphs with the sparsest cuts are exactly the slowest. (On the cycle the sweep is lucky: any smooth enough vector gives an arc, and the half-cycle appears once the vector is smooth at scale $n$.)

## (f) Krylov

Using full-reorthogonalization Lanczos as in the Lecture 6 demo (stopping when the top Ritz value gives $R\le1.1\lambda$) and the lazy power method, averaged over 5 random starts:

| $n$ | Lanczos steps | power steps | $1/\sqrt\lambda$ | $1/\lambda$ |
|---|---|---|---|---|
| 100 | 36 | 636 | 22.5 | 507 |
| 200 | 71 | 1927 | 45 | 2027 |
| 400 | 136 | 12342 | 90 | 8106 |

Lanczos needs about $1.5/\sqrt\lambda\approx n/3$ steps, and the power method about $1$–$1.5/\lambda\approx n^2/\text{const}$ steps (the last power entry is noisy: it depends on the overlap of the random start with the Fiedler eigenspace). So the gap between the two grows linearly in $n$ on the cycle. The sweep of the Lanczos vector is the half-cycle as soon as $R\le1.1\lambda$, because (a) applies to every vector with that quotient, and on the cycle such vectors are close to the exact eigenvectors $\cos(2\pi i/n+\theta)$, whose threshold sets are arcs.
