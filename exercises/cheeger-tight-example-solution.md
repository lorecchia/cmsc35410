# Solution: A Tight Example for the Hard Side of Cheeger

Conventions as in the statement: loops count in $d_i$ but are never cut; $L=D-A$ ignores loops; for a $d$-regular graph $L_{rw}=L/d$.

## (a) Why not the cycle?

The Fiedler eigenspace of $C_n$ ($n$ even) is spanned by $\cos(2\pi i/n)$ and $\sin(2\pi i/n)$, so every vector in it is $r\cos(2\pi i/n+\theta)$. A threshold set of this vector is an **arc** of consecutive vertices. An arc with $\ell$ vertices ($1\le \ell\le n-1$) has $2$ cut edges and volume $2\min\{\ell,n-\ell\}$, so
$$
\Phi(\text{arc})=\frac1{\min\{\ell,n-\ell\}}\ \ge\ \frac1{n/2}=\frac2n=\Phi(C_n).
$$
Some threshold set is a half-cycle (an arc with $\ell = n/2$), so the sweep returns exactly $\Phi(G)$. Here $\Phi(G)=\Theta(\sqrt\lambda)$ **because the graph itself has no cheaper cut**: the cycle shows that the inequality $\Phi\le\sqrt{2\lambda}$ cannot be improved in terms of $\lambda$, but it says nothing about the *rounding loss* $\Phi(S)/\Phi(G)$, which is $1$. For the algorithm we need $\Phi(G)\ll\sqrt\lambda$ **and** a Fiedler vector that does not see the optimal cut.

## (b) Products

The edges of $G\square H$ are of two kinds: $(x,y)\sim(x',y)$ with weight $w^G_{xx'}$, and $(x,y)\sim(x,y')$ with weight $w^H_{yy'}$. So
$$
A_{G\square H}=A_G\otimes I+I\otimes A_H,\qquad D_{G\square H}=D_G\otimes I+I\otimes D_H,
$$
(the degree of $(x,y)$ is $d^G_x+d^H_y$, including loops). Subtracting gives $L_{G\square H}=L_G\otimes I+I\otimes L_H$. Using $(A\otimes B)(u\otimes v)=Au\otimes Bv$,
$$
L_{G\square H}(u\otimes v)=(L_Gu)\otimes v+u\otimes(L_Hv)=(\mu+\nu)\,u\otimes v .
$$
If $\{u_i\}$ and $\{v_j\}$ are eigenbases, the $nk$ vectors $u_i\otimes v_j$ are linearly independent (they form a basis of the tensor product space), so they account for all eigenvalues: the spectrum is $\{\mu_i+\nu_j\}$.

## (c) The two factors

**1. Upper bound on $\lambda_T$.** Let $f=+1$ on the left subtree of the root, $-1$ on the right subtree, $0$ at the root. The two subtrees have $(n-1)/2$ vertices each, so $f\perp\mathbf 1$ and $\|f\|^2=n-1$. Only the two root edges have $f_i\ne f_j$, each with $(f_i-f_j)^2=1$. Hence
$$
\lambda_T=\min_{g\perp\mathbf 1}\frac{g^\top L g}{\|g\|^2}\le\frac{f^\top Lf}{\|f\|^2}=\frac2{n-1}.
$$

**2. Lower bound on $\lambda_T$.** Let $f\perp\mathbf 1$, and route each pair $i<j$ along the unique tree path $\gamma_{ij}$, of length $\le 2h$. Then
$$
n\|f\|^2=\sum_{i<j}(f_i-f_j)^2\le\sum_{i<j}|\gamma_{ij}|\sum_{e\in\gamma_{ij}}(\Delta_ef)^2=\sum_e(\Delta_ef)^2\!\!\sum_{ij:\,e\in\gamma_{ij}}\!\!|\gamma_{ij}|\le\kappa\;f^\top Lf,
$$
where the first inequality is Cauchy–Schwarz along each path, and $\kappa=\max_e\sum_{ij:\,e\in\gamma_{ij}}|\gamma_{ij}|$. A tree edge with $s$ vertices below it lies on $s(n-s)$ paths, and $s(n-s)\le\frac{(n-1)}{2}\cdot\frac{n+1}{2}=\frac{n^2-1}4$. So $\kappa\le 2h\cdot\frac{n^2-1}{4}$ and
$$
\lambda_T\ \ge\ \frac n\kappa\ \ge\ \frac{2n}{h(n^2-1)}\ \ge\ \frac{2}{hn}.
$$

**3. The path.** For $g_y=\cos\!\big(\pi(y-\frac12)/k\big)$, the interior rows give $(Lg)_y=2g_y-g_{y-1}-g_{y+1}=(2-2\cos\frac\pi k)g_y$ by $\cos(a+b)+\cos(a-b)=2\cos a\cos b$. At the endpoints, the loop makes $(Lg)_1=g_1-g_2$, and $g_0=\cos(-\frac{\pi}{2k})=g_1$ (symmetry of cosine), so the same formula holds. Similarly at $y=k$, since $g_{k+1}=g_k$. Also $\sum_yg_y=0$ (symmetry about the midpoint), so $g\perp\mathbf 1$ and $\lambda_P\le 2-2\cos\frac\pi k$; the full spectrum is $2-2\cos\frac{j\pi}{k}$, $j=0,\dots,k-1$, so equality holds. The bound $2-2\cos x\le x^2$ gives $\lambda_P\le\pi^2/k^2$. And $g_y$ is strictly decreasing for $y=1,\dots,k$ since $\pi(y-\frac12)/k\in(0,\pi)$.

## (d) The cheap cut

Every vertex $(x,y)$ has degree $d^T_x+d^P_y=3+2=5$, so $G$ is $5$-regular and $L_{rw}=L/5$. Let $S=V(\text{left subtree of the root})\times\{1,\dots,k\}$. It has $k(n-1)/2$ vertices, so $\mathrm{vol}(S)=5k(n-1)/2$, which is less than $\mathrm{vol}(\bar S)$. The cut edges are the edge (root, left child) in each of the $k$ copies of the tree. Therefore
$$
\Phi(G)\le\Phi(S)=\frac{k}{5k(n-1)/2}=\frac2{5(n-1)} .
$$

## (e) The Fiedler vector points the wrong way

If $k>\pi\sqrt{hn/2}$, then $\lambda_P\le\pi^2/k^2<2/(hn)\le\lambda_T$ by (c). By (b), the eigenvalues of $L_G$ are $\mu_i+\nu_j$. The smallest is $0$ (both constant), and the next smallest are
$$
\lambda_P\ (\text{from } \mathbf 1\otimes g),\qquad \lambda_T\ (\text{from } f_T\otimes\mathbf 1),\qquad 2-2\cos\tfrac{2\pi}k\ (\mathbf 1\otimes g^{(2)}),
$$
and everything else is larger. Since $\lambda_P<\lambda_T$ and $2-2\cos\frac{2\pi}{k}>2-2\cos\frac\pi k=\lambda_P$, the eigenvalue $\lambda_P$ is simple. So $\lambda=\lambda_2(L)/5=\lambda_P/5$, and the Fiedler vector is $F(x,y)=g_y$.

## (f) The sweep loses a square root

Because $F(x,y)=g_y$ is strictly decreasing in $y$, every threshold set is $S=V(T_h)\times\{1,\dots,j\}$ for some $1\le j\le k-1$. Its cut consists of the $n$ edges between rows $j$ and $j+1$. Its volume is $5jn$, and $\mathrm{vol}(V)=5nk$, so
$$
\Phi(S)=\frac{n}{5n\min\{j,k-j\}}=\frac1{5\min\{j,k-j\}}\ \ge\ \frac1{5\lfloor k/2\rfloor}\ \ge\ \frac2{5k}.
$$
With $k=\lceil\pi\sqrt{hn/2}\rceil+1\le\pi\sqrt{hn/2}+2$, and (d),
$$
\frac{\Phi(S)}{\sqrt{\Phi(G)}}\ \ge\ \frac{2/(5k)}{\sqrt{2/(5(n-1))}}\ \xrightarrow[h\to\infty]{}\ \frac{2/5}{\pi\sqrt{h/2}}\sqrt{\frac52}=\frac{2\sqrt5}{5\pi}\cdot\frac1{\sqrt h}\approx\frac{0.28}{\sqrt h}.
$$
(Exact values of the left side times $\sqrt h$, computed for $h=2,\dots,14$: from $0.22$ up to $0.284$.) Meanwhile $\Phi(G)\le 2/(5(n-1))=O(1/n)$ while $\Phi(S)\ge 2/(5k)=\Omega(1/\sqrt{n\log n})$, so $\Phi(S)\ge\Omega(\sqrt{\Phi(G)/\log n})$.

## (g) Numerical check

Code (with `L_T`, `L_P` the unnormalized Laplacians, built as in (c)):

```python
import numpy as np, math

def tree_L(h):
    n = 2**(h+1) - 1; L = np.zeros((n, n))
    for i in range(1, n):
        p = (i - 1)//2
        L[i, i] += 1; L[p, p] += 1; L[i, p] -= 1; L[p, i] -= 1
    return L

def path_L(k):
    L = np.zeros((k, k))
    for i in range(k - 1):
        L[i, i] += 1; L[i+1, i+1] += 1; L[i, i+1] -= 1; L[i+1, i] -= 1
    return L

for h in [3, 4, 5, 6]:
    n = 2**(h+1) - 1; k = math.ceil(1.1*math.pi*math.sqrt(n)); N = n*k
    L = (np.kron(tree_L(h), np.eye(k)) + np.kron(np.eye(n), path_L(k))) / 5   # L_rw of G
    w, V = np.linalg.eigh(L); lam, f = w[1], V[:, 1]
    A = -5*L; np.fill_diagonal(A, 0)                    # adjacency (no loops)
    order = np.argsort(f); fs = f[order]; best = np.inf
    for t in range(1, N):
        if fs[t] - fs[t-1] < 1e-9: continue            # only threshold sets
        S = np.zeros(N, bool); S[order[:t]] = True
        best = min(best, A[np.ix_(S, ~S)].sum() / (5*min(t, N - t)))
    tree_cut = 2/(5*(n - 1))
    print(h, n, k, lam, best, tree_cut, best/tree_cut)
```

Output:

| $h$ | $n$ | $k$ | $\lambda$ | best sweep $\Phi(S)$ | cut of (d) | ratio | $\Phi(S)/\sqrt{\text{cut of (d)}}$ |
|---|---|---|---|---|---|---|---|
| 3 | 15 | 14 | 0.01003 | 0.0286 | 0.0286 | 1.0 | 0.17 |
| 4 | 31 | 20 | 0.00492 | 0.0200 | 0.0133 | 1.5 | 0.17 |
| 5 | 63 | 28 | 0.00252 | 0.0143 | 0.00645 | 2.2 | 0.18 |
| 6 | 127 | 39 | 0.00130 | 0.0105 | 0.00317 | 3.3 | 0.19 |

In each case $\lambda=\lambda_P/5$ to the printed precision. The ratio between the best sweep cut and the cheap cut grows like $\sqrt n$, and $\Phi(S)/\sqrt{\Phi(G)}$ stays bounded below by a constant. (Since the cut of (d) is only an upper bound on $\Phi(G)$, the last column is a lower bound for $\Phi(S)/\sqrt{\Phi(G)}$.)

**Where the $\log$ went.** The test function in (c) gives $\lambda_T\le 2/(n-1)$, but our proof only gives $\lambda_T\ge 2/(hn)$ because the canonical-path bound is lossy by a factor $h$. Numerically $n\lambda_T\to1$ ($1.45,1.25,1.14,1.08,\dots$ for $h=3,4,5,6,\dots$), i.e. $\lambda_T=\Theta(1/n)$. Then it suffices to take $k\approx\pi\sqrt n$ (which is what the code does, with a safety factor $1.1$), and the same computation gives $\Phi(S)\ge 2/(5k)\approx 0.13/\sqrt n$ against $\Phi(G)\le 0.4/n$: $\Phi(S)/\sqrt{\Phi(G)}\ge0.2$, with **no** $\sqrt{\log n}$.

## (h) Discussion

In $G$, a cheap cut exists because the **tree is large** (many vertices behind one edge: $\Phi\approx\lambda_T\approx1/n$). An expensive cut exists because the **path is long** (diffusion along it is slow, $\lambda_P\approx\pi^2/k^2$, but any cut costs $\approx1/k\approx\sqrt{\lambda_P}$, the cycle phenomenon). When $\lambda_P\le\lambda_T$ the smallest eigenvalue is the path's, even though the cheapest cut is the tree's. The eigenvector only encodes the *slowest mode*, and a Rayleigh quotient cannot distinguish "small because a sparse cut exists" from "small because the graph is long and thin".

On $Q_d$, $\lambda_2=2/d$ has multiplicity $d$ (coordinate functions $x_i$). The coordinate vectors give the optimal cut $\Phi=1/d$, but a vector $\sum_ig_ix_i$ with generic coefficients is close to Gaussian by the CLT, and (a standard computation, not proved here) its sweep cuts have conductance $\Theta(1/\sqrt d)$. The same loss, produced by degeneracy instead of near-degeneracy: here the tree$\,\square\,$path example perturbs the picture so the eigenvalue is *simple*, and no choice of eigenvector can rescue the sweep.
