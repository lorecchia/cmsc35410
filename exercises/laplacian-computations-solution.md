# Solutions: Computing with Graph Laplacians

Conventions are as in the exercise sheet. Two facts are used throughout.

- **Quadratic forms.** For every $f \in \mathbb R^V$,
$$
f^\top L f = \sum_{\{i,j\}\in E} w_{ij}(f_i - f_j)^2, \qquad f^\top (D + A) f = \sum_{\{i,j\}\in E} w_{ij}(f_i + f_j)^2 .
$$
  The second follows from $f^\top D f = \sum_{\{i,j\}\in E} w_{ij}(f_i^2 + f_j^2)$ and $f^\top A f = 2\sum_{\{i,j\}\in E} w_{ij} f_i f_j$.
- **Rayleigh quotients for the normalized matrices.** $L_{rw} = D^{-1/2}\mathcal L D^{1/2}$ is similar to the symmetric $\mathcal L$, so it is diagonalizable with the same spectrum. Substituting $g = D^{1/2} f$ into Courant–Fischer for $\mathcal L$ gives
$$
\lambda_k(\mathcal L) = \min_{\dim U = k}\ \max_{f \in U\setminus 0} \frac{f^\top L f}{f^\top D f}, \qquad \lambda_n(\mathcal L) = \max_{f \ne 0}\frac{f^\top L f}{f^\top D f}.
$$

---

## Exercise 1

**(a)** Each $\mathbf 1_{V_k}$ is in the kernel. For any $i$,
$$
(L\mathbf 1_{V_k})_i = \sum_j w_{ij}\big(\mathbf 1_{V_k}(i) - \mathbf 1_{V_k}(j)\big) = 0,
$$
because every neighbor $j$ of $i$ is in the same component as $i$.

Conversely, suppose $Lf = 0$. Then
$$
0 = f^\top L f = \sum_{\{i,j\}\in E} w_{ij}(f_i - f_j)^2 .
$$
All weights are positive, so $f_i = f_j$ on every edge. Any two vertices in the same component are joined by a path, so $f$ is constant on each component. Hence $f \in \operatorname{span}\{\mathbf 1_{V_k}\}$.

The indicators have disjoint supports, so they are linearly independent and $\dim\ker L = c$. Since $L$ is symmetric, the algebraic multiplicity of $0$ equals $\dim \ker L$.

**(b)** $D$ is invertible because there are no isolated vertices. So $L_{rw}f = D^{-1}Lf = 0$ if and only if $Lf = 0$, and $\ker L_{rw} = \ker L$.

$L_{rw}$ is not symmetric, but it is similar to $\mathcal L$, so it is diagonalizable. For a diagonalizable matrix, algebraic and geometric multiplicities agree, so the multiplicity of $0$ is again $c$.

For $\mathcal L$: $\mathcal L g = 0 \iff L D^{-1/2} g = 0$. So
$$
\ker \mathcal L = D^{1/2}\ker L = \operatorname{span}\{D^{1/2}\mathbf 1_{V_1}, \dots, D^{1/2}\mathbf 1_{V_c}\}.
$$

**(c)** All three matrices are PSD with $\lambda_1 = 0$. So $\lambda_2 > 0$ if and only if $0$ has multiplicity $1$, which by (a)–(b) means $c = 1$. $\blacksquare$

---

## Exercise 2

**(a)** By the quadratic-form identities,
$$
2 f^\top D f - f^\top L f = f^\top (D + A) f = \sum_{\{i,j\}\in E} w_{ij}(f_i + f_j)^2 \ \ge 0 .
$$
So $\frac{f^\top Lf}{f^\top Df} \le 2$ for all $f \ne 0$, and therefore $\lambda_n(\mathcal L) \le 2$.

**Equality.** $\lambda_n = 2$ if and only if some $f \ne 0$ satisfies $f^\top(D + A) f = 0$, that is, $f_i = -f_j$ on every edge.

- *Such an $f$ gives a bipartite component.* $|f|$ is constant on every edge, hence on every component. Pick a component $C$ where $f \ne 0$. Then $f$ is nonzero at every vertex of $C$. Every edge of $C$ joins a vertex where $f > 0$ to one where $f < 0$. So $X = \{f > 0\} \cap C$, $Y = \{f < 0\} \cap C$ is a bipartition of $C$.
- *A bipartite component gives such an $f$.* If $C$ is a component with bipartition $X \cup Y$, take $f = \mathbf 1_X - \mathbf 1_Y$. Then $f_i + f_j = 0$ on every edge of $C$, and $f$ vanishes off $C$.

Explicitly, for $i \in X$, $(L_{rw}f)_i = f_i - \frac1{d_i}\sum_j w_{ij}f_j = 1 - (-1) = 2$, and symmetrically on $Y$.

**(b)** From (a),
$$
f^\top L f = 2f^\top D f - f^\top(D+A)f \le 2 f^\top D f \le 2 d_{\max}\|f\|^2 .
$$
So $\lambda_n(L) \le 2d_{\max}$. Equality needs both inequalities to be tight for some $f \ne 0$:

1. $f^\top(D+A)f = 0$. As in (a), $f$ is nonzero on all of some component $C$, and $C$ is bipartite.
2. $f^\top D f = d_{\max}\|f\|^2$, so $f$ is supported on vertices of degree $d_{\max}$. Since $f$ is nonzero on all of $C$, every vertex of $C$ has degree $d_{\max}$.

Conversely, if $C$ is bipartite with all degrees equal to $d_{\max}$, then $f = \mathbf 1_X - \mathbf 1_Y$ makes both inequalities equalities. $\blacksquare$

> **Remark.** The signless Laplacian $D + A$ is PSD, and part (a) says $\lambda_{\min}(D+A) = 0$ if and only if some component is bipartite. This is the same statement as (a) in a different form.

---

## Exercise 3

**(a)** Let the bipartition be $X \cup Y$ and let $S = \operatorname{diag}(s)$ with $s = \mathbf 1_X - \mathbf 1_Y$. Every edge joins $X$ to $Y$, so $A_{ij} \ne 0$ implies $s_i s_j = -1$. Hence $SAS = -A$. Diagonal matrices commute, so $SDS = D$ and $S D^{-1/2} = D^{-1/2} S$. Therefore
$$
S\mathcal L S = I - D^{-1/2}(SAS)D^{-1/2} = I + D^{-1/2}AD^{-1/2} = 2I - \mathcal L .
$$
Since $S = S^{-1}$, the matrices $\mathcal L$ and $2I - \mathcal L$ are similar. They have the same characteristic polynomial, so $\lambda$ and $2 - \lambda$ are eigenvalues of $\mathcal L$ with the same multiplicity.

**Eigenvectors.** If $\mathcal L\varphi = \lambda\varphi$, then
$$
\mathcal L(S\varphi) = S(2I - \mathcal L)\varphi = (2 - \lambda)S\varphi .
$$
So flipping the sign of an eigenvector on one side maps the $\lambda$-eigenspace onto the $(2-\lambda)$-eigenspace. For example, $D^{1/2}\mathbf 1$ (eigenvalue $0$) maps to $D^{1/2}(\mathbf 1_X - \mathbf 1_Y)$ (eigenvalue $2$), which is the vector from Exercise 2.

**(b)** The eigenvalues of $P = I - L_{rw}$ are $\mu_k = 1 - \lambda_k$. If the $\lambda_k$ are symmetric about $1$, the $\mu_k$ are symmetric about $0$. Pairing $\mu$ with $-\mu$ shows that for every odd $k$,
$$
\operatorname{tr} P^k = \sum_{i} \mu_i^k = 0 .
$$
Now expand the diagonal entries:
$$
(P^k)_{ii} = \sum_{i = v_0, v_1, \dots, v_k = i}\ \prod_{t=0}^{k-1} P_{v_t v_{t+1}} \ \ge 0,
$$
where the sum is over closed walks of length $k$ at $i$. Each term is positive exactly when the walk uses only edges. So $(P^k)_{ii} > 0$ if and only if there is a closed walk of length $k$ through $i$.

A sum of non-negative terms is zero only if every term is zero. So for every odd $k$, there is no closed walk of length $k$ anywhere in $G$. An odd cycle is such a walk, so $G$ has no odd cycle, and a graph with no odd cycle is bipartite. Nothing here used connectivity. $\blacksquare$

> **Remark (a second proof of (a)).** The argument runs backwards too. If $G$ is bipartite, it has no closed walks of odd length, so $\sum_i \mu_i^k = 0 = \sum_i (-\mu_i)^k$ for odd $k$. For even $k$ the two sums agree trivially. The power sums $p_1, \dots, p_n$ determine the elementary symmetric polynomials by Newton's identities, hence the characteristic polynomial. So $\{\mu_i\}$ and $\{-\mu_i\}$ are the same multiset. This proves that **$G$ is bipartite if and only if the spectrum of $\mathcal L$ is symmetric about $1$**, for any graph.
>
> For a *connected* graph, (b) has a one-line proof: $0$ is an eigenvalue, so symmetry makes $2$ an eigenvalue, and Exercise 2 applies.

**(c)** Take $P_3 = K_{1,2}$. By Exercise 4, $L$ has spectrum $\{0, 1, 3\}$. A set of three numbers symmetric about a point $c$ must have its middle element equal to $c$, and its outer elements symmetric about $c$. Here the outer elements give $c = 3/2$, but the middle element is $1$. Normalization is essential.

---

## Exercise 4

**(a) $K_n$.** $A = J - I$ and $D = (n-1)I$, so $L = nI - J$. $J$ has eigenvalue $n$ on $\mathbf 1$ and $0$ on $\mathbf 1^\perp$. Hence:

- $L$: eigenvalue $0$ on $\mathbf 1$; eigenvalue $n$ with multiplicity $n-1$, on all of $\mathbf 1^\perp$.
- $\mathcal L = L/(n-1)$: eigenvalue $0$; eigenvalue $\frac{n}{n-1}$ with multiplicity $n-1$.

**(c) $K_{a,b}$** (we do (c) first; (b) is the case $a = 1$). Sides $X$, $|X| = a$, and $Y$, $|Y| = b$. Vertices in $X$ have degree $b$, vertices in $Y$ have degree $a$.

- *Vectors on $Y$ summing to zero.* Let $f$ be supported on $Y$ with $\sum_{j\in Y} f_j = 0$. For $j \in Y$, all neighbors are in $X$ where $f = 0$, so $(Lf)_j = a f_j$. For $i \in X$, $(Lf)_i = b \cdot 0 - \sum_{j \in Y} f_j = 0$. So $Lf = af$. This space has dimension $b - 1$.
- *Vectors on $X$ summing to zero.* Symmetrically, eigenvalue $b$ with multiplicity $a - 1$.
- $\mathbf 1$ gives eigenvalue $0$.
- The last eigenvalue comes from the trace: $\operatorname{tr} L = 2ab$, and $0 + a(b-1) + b(a-1) + \lambda = 2ab$ gives $\lambda = a + b$. Its eigenvector is orthogonal to everything above, so it is constant on each side: $f = b\,\mathbf 1_X - a\,\mathbf 1_Y$. Check: for $i \in X$, $(Lf)_i = b\cdot b - b\cdot(-a) = (a+b)\,b$.

So $\operatorname{spec} L(K_{a,b}) = \{0,\ a^{(b-1)},\ b^{(a-1)},\ a+b\}$.

For $\mathcal L$: $D$ is constant on each side, so on the two zero-sum subspaces above $\mathcal L f = \frac1a L f = f$ (on $Y$) and $\frac1b Lf = f$ (on $X$). That gives eigenvalue $1$ with multiplicity $a + b - 2$. The graph is bipartite, so $0$ and $2$ are also eigenvalues (Exercise 2). Hence
$$
\operatorname{spec}\mathcal L(K_{a,b}) = \{0,\ 1^{(a+b-2)},\ 2\}.
$$

**(b) $K_{1,m}$.** Setting $a = 1$, $b = m$:

- $L$: $\{0,\ 1^{(m-1)},\ m+1\}$. The eigenvectors for $1$ are zero-sum vectors on the leaves. The eigenvector for $m + 1 = n$ is $m$ at the center and $-1$ at each leaf.
- $\mathcal L$: $\{0,\ 1^{(m-1)},\ 2\}$.

**(d)** $K_{1,3}$ and $C_4 = K_{2,2}$ both have $\mathcal L$-spectrum $\{0, 1, 1, 2\}$. More generally, all $K_{a,b}$ with $a + b = n$ are $\mathcal L$-cospectral. Their $L$-spectra differ: $\{0,1,1,4\}$ for the star and $\{0,2,2,4\}$ for $C_4$.

> **Moral.** $\mathcal L$ forgets how mass is distributed between the two sides of $K_{a,b}$. In the random-walk picture, every complete bipartite walk is the same: from any vertex, jump to a uniform vertex on the other side. $L$ remembers the sizes.

---

## Exercise 5

**(a)** Index $\mathbb R^{V_G \times V_H} = \mathbb R^{V_G}\otimes\mathbb R^{V_H}$ by pairs $(g,h)$. By definition of the product,
$$
A_{G\square H} = A_G \otimes I + I \otimes A_H .
$$
The degree of $(g,h)$ is $d_G(g) + d_H(h)$, so $D_{G\square H} = D_G \otimes I + I \otimes D_H$. Subtracting gives $L_{G\square H} = L_G\otimes I + I\otimes L_H$.

For eigenvectors,
$$
(L_G\otimes I + I\otimes L_H)(\varphi\otimes\psi) = (L_G\varphi)\otimes\psi + \varphi\otimes(L_H\psi) = (\alpha + \beta)\,\varphi\otimes\psi .
$$
If $\{\varphi_i\}$ and $\{\psi_j\}$ are orthonormal eigenbases, then $\{\varphi_i\otimes\psi_j\}$ is orthonormal, since $\langle\varphi\otimes\psi, \varphi'\otimes\psi'\rangle = \langle\varphi,\varphi'\rangle\langle\psi,\psi'\rangle$. It has $|V_G||V_H|$ elements, so it is an eigenbasis.

**(b)** For $K_2$ with vertices $x \in \{0,1\}$, $L = \begin{pmatrix}1 & -1\\ -1 & 1\end{pmatrix}$. Its eigenvectors are $\mathbf 1$ (eigenvalue $0$) and $x \mapsto (-1)^x$ (eigenvalue $2$).

By (a) applied $d - 1$ times, the eigenvectors of $L(Q_d)$ are the products
$$
\chi_S(x) = \prod_{i \in S}(-1)^{x_i} = (-1)^{\sum_{i\in S}x_i}, \qquad S \subseteq [d],
$$
with eigenvalue $\sum_{i \in S} 2 = 2|S|$. So $L(Q_d)$ has eigenvalue $2k$ with multiplicity $\binom dk$, for $k = 0, \dots, d$.

$Q_d$ is $d$-regular, so $L_{rw} = L/d$, with eigenvalues $2k/d$. The spectral gap is $2/d$ and $t_{\rm rel} = d/2$, as in Lecture 2, §6.5.

**(c)** $D_{G\square H} = (d_G + d_H)I$, so
$$
L_{rw}(G\square H) = \frac{L_G\otimes I + I\otimes L_H}{d_G + d_H} = \frac{d_G}{d_G + d_H}\,L_{rw}(G)\otimes I + \frac{d_H}{d_G+d_H}\,I\otimes L_{rw}(H).
$$
Its eigenvalues are the weighted averages $\frac{d_G\alpha + d_H\beta}{d_G + d_H}$, with eigenvectors $\varphi\otimes\psi$.

**(d)** Label the vertices of $P_3$ by $x \in \{-1, 0, 1\}$.

1. $P_3$ is bipartite, so $0$ and $2$ are eigenvalues of $L_{rw}$. The trace is $3$, so the third eigenvalue is $1$. Its eigenvector is $\varphi = (1, 0, -1)$. Check: $(L_{rw}\varphi)_{-1} = 1 - 0 = 1$; $(L_{rw}\varphi)_0 = 0 - \tfrac12(1 - 1) = 0$; and symmetrically at $x = 1$.

2. Let $f(x,y) = \varphi_x$. In the $3\times3$ grid, corners have degree $2$, edge midpoints degree $3$, and the center degree $4$.
   - At the corner $(-1,-1)$: $f = 1$, and the neighbors $(0,-1)$, $(-1,0)$ have $f = 0$ and $f = 1$. So $(L_{rw}f) = 1 - \tfrac12(0 + 1) = \tfrac12$.
   - At the midpoint $(-1,0)$: $f = 1$, and the neighbors $(-1,\pm1)$, $(0,0)$ have $f = 1, 1, 0$. So $(L_{rw}f) = 1 - \tfrac13\cdot2 = \tfrac13$.

   The ratios $(L_{rw}f)/f$ differ, so $f$ is not an eigenvector. The proof of (c) breaks because $D_{G\square H} = D_G\otimes I + I\otimes D_H$ is not a multiple of the identity, so $D^{-1}$ does not split across the two factors. The value $\tfrac12$ predicted by (c) is not $\lambda_2$; the true value is computed next.

3. The reflections $x\mapsto -x$, $y \mapsto -y$ and the diagonal swap $x \leftrightarrow y$ are automorphisms of the grid. They preserve degrees, so they commute with $L_{rw}$, and we can split $\mathbb R^9$ by parity.

   - **Odd in $x$, even in $y$.** $f$ vanishes on the middle column. On the left column it is determined by two numbers: $c$ at both corners and $m$ at the midpoint. The right column has $-c$ and $-m$. The eigenvalue equations at $(-1,-1)$ and $(-1,0)$ are
$$
\lambda c = c - \tfrac12 m, \qquad \lambda m = m - \tfrac23 c .
$$
     The matrix $\begin{pmatrix} 1 & -1/2 \\ -2/3 & 1\end{pmatrix}$ has eigenvalues $1 \pm \frac1{\sqrt3}$.
   - **Even in $x$, odd in $y$.** The same, by the diagonal swap.
   - **Odd in both.** $f$ is supported on the corners, with alternating signs. Every neighbor of a corner is a midpoint, where $f = 0$, so $\lambda = 1$.
   - **Even in both.** The values are $c$ at the corners, $p$ at $(\pm1, 0)$, $q$ at $(0,\pm1)$, and $z$ at the center. If $p = -q$ and $c = z = 0$, this is an eigenvector with $\lambda = 1$. If $p = q$, the equations are $\lambda c = c - p$, $\lambda p = p - \frac13(2c + z)$, $\lambda z = z - p$. Subtracting the first and third gives $\lambda(c - z) = c - z$, so $\lambda = 1$ or $c = z$. When $c = z$ the system becomes $\lambda c = c - p$, $\lambda p = p - c$, with eigenvalues $0$ and $2$. So this sector contributes $\{0, 1, 1, 2\}$.

   Altogether the spectrum is $\{0,\ (1 - \tfrac1{\sqrt3})^{(2)},\ 1^{(3)},\ (1 + \tfrac1{\sqrt3})^{(2)},\ 2\}$, and
$$
\lambda_2\big(L_{rw}(P_3\square P_3)\big) = 1 - \frac1{\sqrt3} \approx 0.423, \qquad \text{not } \tfrac12 .
$$

---

## Exercise 6

**(a)** $L_{ii} = d_i$, and $\sum_i d_i = \sum_i\sum_j w_{ij} = 2\sum_{e}w_e$. Also $\mathcal L_{ii} = L_{ii}/d_i = 1$, because there are no self-loops ($A_{ii} = 0$) and no isolated vertices. So $\operatorname{tr}\mathcal L = n$.

**(b)** $\lambda_1(\mathcal L) = 0$, so $\lambda_2 + \dots + \lambda_n = n$. These are $n-1$ numbers with average $\frac{n}{n-1}$. The smallest is at most the average, and the largest is at least the average.

**(c)** *If $G$ is complete with equal weights $w$.* Then $D = (n-1)wI$, and
$$
\mathcal L = I - \frac{J - I}{n-1} = \frac{n}{n-1}I - \frac{1}{n-1}J .
$$
Its eigenvalues are $0$ and $\frac{n}{n-1}$.

*Conversely,* suppose $\lambda_2 = \frac{n}{n-1}$. By (b), $\lambda_2$ equals the average of $\lambda_2, \dots, \lambda_n$, and it is also their minimum, so they are all equal to $c := \frac{n}{n-1}$. Let $u = D^{1/2}\mathbf 1/\sqrt{\mathrm{vol}(V)}$ be the unit $0$-eigenvector. Then
$$
\mathcal L = c\,(I - uu^\top), \qquad D^{-1/2}AD^{-1/2} = I - \mathcal L = (1 - c)I + c\,uu^\top .
$$
The left side has zero diagonal. So for every $i$,
$$
0 = (1 - c) + c\,u_i^2 \quad\Longrightarrow\quad u_i^2 = \frac{c - 1}{c} = \frac1n \quad\Longrightarrow\quad d_i = \frac{\mathrm{vol}(V)}{n}.
$$
So $G$ is regular, say of degree $d$. The off-diagonal entries then give $A_{ij}/d = c\,u_iu_j = \frac{c}{n} = \frac{1}{n-1}$, so $A_{ij} = \frac{d}{n-1}$ for all $i \ne j$. This is the complete graph with equal weights. $\blacksquare$

**(d)** $f = e_v - \frac1n\mathbf 1$ is orthogonal to $\mathbf 1$. Since $L\mathbf 1 = 0$,
$$
f^\top L f = e_v^\top L e_v = d_v, \qquad \|f\|^2 = 1 - \tfrac2n + \tfrac1n = \tfrac{n-1}{n}.
$$
By Courant–Fischer, $\lambda_2(L) = \min_{f\perp\mathbf 1}\frac{f^\top Lf}{\|f\|^2} \le \frac{n}{n-1}d_v$, and $\lambda_n(L) = \max_f \frac{f^\top Lf}{\|f\|^2} \ge \frac{n}{n-1}d_v$. Choose $v$ of minimum and maximum degree respectively.

**Tightness.**
- For $K_n$, both are equalities: $d = n-1$ and every nonzero eigenvalue is $n$.
- The upper bound on $\lambda_n$ is also tight for the star: $\lambda_n = n = \frac{n}{n-1}(n-1)$.
- The bound on $\lambda_2$ is not tight for the star: $\frac{n}{n-1}$ versus the true value $1$.

---

## Exercise 7

**(a)** Let $b_e = e_u - e_v$. Then
$$
L_H = L_G + w_e\, b_e b_e^\top \succeq L_G,
$$
so $f^\top L_H f \ge f^\top L_G f$ for every $f$. By Courant–Fischer, $\lambda_k(L) = \min_{\dim U = k}\max_{f\in U\setminus0}\frac{f^\top Lf}{\|f\|^2}$. The quotient increases pointwise in $f$, and the constraint set (subspaces $U$) is the same for $G$ and $H$. So every $\lambda_k$ increases.

**(b)** Let $v$ have degree $d = d_{\max}$. $G$ contains the star $K_{1,d}$ centered at $v$ as a subgraph, plus the remaining vertices. Build $G$ from this star by adding edges, and apply (a). Then
$$
\lambda_n(L_G) \ge \lambda_n(L_{K_{1,d}}) = d + 1
$$
by Exercise 4(b). The isolated vertices only add zero eigenvalues.

This beats Exercise 6(d) for unweighted graphs: $d + 1 \ge \frac{n}{n-1}d$ is equivalent to $n - 1 \ge d$.

**(c)** $P_3$ is bipartite, with $\mathcal L$-spectrum $\{0, 1, 2\}$ (Exercise 4(b), $m = 2$). Adding the missing edge gives $K_3$, with spectrum $\{0, \tfrac32, \tfrac32\}$ (Exercise 4(a)). So $\lambda_3$ drops from $2$ to $\tfrac32$. This is forced by Exercise 2: closing an odd cycle destroys bipartiteness, so $\lambda_n$ must drop below $2$.

**(d)** Start from $K_{1,3}$, with center $c$ and leaves $a, b, e$. Its $\mathcal L$-spectrum is $\{0, 1, 1, 2\}$, so $\lambda_2 = 1$. Add the edge $\{a, b\}$. Now $d_a = d_b = 2$, $d_e = 1$, $d_c = 3$. Work with $L_{rw}$.

- **Odd under $a \leftrightarrow b$.** $f = e_a - e_b$. Then $(L_{rw}f)_a = 1 - \tfrac12(-1 + 0) = \tfrac32$, and symmetrically at $b$. At $c$ and $e$ the value is $0$. So $\lambda = \tfrac32$.
- **Even under $a \leftrightarrow b$.** Write $f = (x, x, y, z)$ on $(a, b, e, c)$. The equations are
$$
\lambda x = x - \tfrac12(x + z), \qquad \lambda y = y - z, \qquad \lambda z = z - \tfrac13(2x + y).
$$
  The matrix $M = \begin{pmatrix} 1/2 & 0 & -1/2 \\ 0 & 1 & -1 \\ -2/3 & -1/3 & 1 \end{pmatrix}$ has eigenvalue $0$ (on $\mathbf 1$) and two others, $\mu_\pm$. From the trace, $\mu_+ + \mu_- = \tfrac52$. The sum of the principal $2\times2$ minors is $\tfrac12 + \tfrac16 + \tfrac23 = \tfrac43$. Because one eigenvalue is $0$, this sum equals $\mu_+\mu_-$. So
$$
\mu^2 - \tfrac52\mu + \tfrac43 = 0, \qquad \mu_\pm = \frac{15 \pm \sqrt{33}}{12}.
$$

The new spectrum is $\{0,\ \tfrac{15-\sqrt{33}}{12},\ \tfrac32,\ \tfrac{15+\sqrt{33}}{12}\} \approx \{0,\ 0.771,\ 1.5,\ 1.729\}$, which sums to $4$ as it should. So **$\lambda_2$ drops from $1$ to $\frac{15-\sqrt{33}}{12} \approx 0.771$.**

**What breaks in (a).** The Rayleigh quotient for $\mathcal L$ is $\frac{f^\top Lf}{f^\top Df}$. Adding $\{u,v\}$ increases the numerator by $w_e(f_u - f_v)^2$, and it also increases the denominator by $w_e(f_u^2 + f_v^2)$. The ratio can go either way. The constraint changes too: $\lambda_2$ minimizes over $f$ that are $D$-orthogonal to $\mathbf 1$, and $D$ changes. In the example, $a$ and $b$ gain stationary mass, and a function separating $e$ from the rest of the graph now pays less energy per unit variance.

---

## Exercise 8

**(a)** Detailed balance $\pi_1 a = \pi_2 b$ gives
$$
\pi = \Big(\frac{b}{a+b},\ \frac{a}{a+b}\Big).
$$
Let $E = I - \mathbf 1\pi^\top$. Since $\pi^\top\mathbf 1 = 1$,
$$
E^2 = I - 2\,\mathbf 1\pi^\top + \mathbf 1(\pi^\top\mathbf 1)\pi^\top = E .
$$
Explicitly,
$$
E = \begin{pmatrix}\pi_2 & -\pi_2 \\ -\pi_1 & \pi_1\end{pmatrix} = \frac1{a+b}\begin{pmatrix} a & -a \\ -b & b\end{pmatrix}, \qquad\text{so}\qquad L_{rw} = I - P = (a+b)E = \lambda E .
$$
$E$ is a projection of rank $1$, so the eigenvalues of $L_{rw}$ are $0$ and $\lambda = a + b$. Since $E^k = E$ for $k \ge 1$,
$$
H_t = e^{-t\lambda E} = I + (e^{-\lambda t} - 1)E = \mathbf 1\pi^\top + e^{-\lambda t}E .
$$

**(b)** The first row of $H_t$ is
$$
\mu_t = (\pi_1 + s\pi_2,\ \ \pi_2 - s\pi_2), \qquad s = e^{-\lambda t}.
$$
So $\mu_t - \pi = s\pi_2\,(1, -1)$, and
$$
\|\mu_t - \pi\|_{TV} = s\,\pi_2, \qquad
\chi^2(\mu_t\Vert\pi) = (s\pi_2)^2\Big(\frac1{\pi_1} + \frac1{\pi_2}\Big) = s^2\,\frac{\pi_2}{\pi_1},
$$
$$
D(\mu_t\Vert\pi) = (\pi_1 + s\pi_2)\log\Big(1 + s\frac{\pi_2}{\pi_1}\Big) + \pi_2(1 - s)\log(1 - s).
$$
At $t = 0$ ($s = 1$) these are $\pi_2$, $\frac{\pi_2}{\pi_1} = \frac1{\pi_1} - 1$, and $\log\frac1{\pi_1}$, matching Lecture 2, §3.

**(c)** $\chi^2(\mu_t\Vert\pi) = e^{-2\lambda t}\,\chi^2(\mu_0\Vert\pi)$ exactly. The density $h_0 - 1$ lies entirely in the single non-constant eigenspace. The only inequality in Proof 1 of §4, $e^{-2\lambda_k t} \le e^{-2\lambda t}$, is an equality for every mode present.

**(d)**
$$
\frac{2\|\mu_t - \pi\|_{TV}}{\sqrt{\chi^2(\mu_t\Vert\pi)}} = \frac{2s\pi_2}{s\sqrt{\pi_2/\pi_1}} = 2\sqrt{\pi_1\pi_2} \ \le 1,
$$
for all $t$. Equality holds if and only if $\pi_1 = \pi_2$, i.e. $a = b$. Comparison (1) is Cauchy–Schwarz for $|h - 1|$ against $\mathbf 1$ in $L^2(\pi)$, and it is tight exactly when $|h - 1|$ is constant. Here $|h - 1| = s\cdot(\pi_2/\pi_1,\ 1)$, which is constant only when $\pi$ is uniform.

**(e)** Let $x = s\pi_2/\pi_1 \to 0$. Use $(1 + x)\log(1 + x) = x + \frac{x^2}{2} + O(x^3)$ and $(1 - s)\log(1 - s) = -s + \frac{s^2}{2} + O(s^3)$:
$$
D(\mu_t\Vert\pi) = \pi_1\Big(x + \frac{x^2}{2}\Big) + \pi_2\Big(-s + \frac{s^2}{2}\Big) + O(s^3) = \frac{s^2\pi_2^2}{2\pi_1} + \frac{s^2\pi_2}{2} + O(s^3) = \frac{s^2\pi_2}{2\pi_1} + O(s^3).
$$
The first-order terms cancel because $\pi_1 x = s\pi_2$. So $D \sim \frac12\chi^2$.

This is a general fact: near equilibrium, $\Phi(x) = x\log x$ has $\Phi''(1) = 1$, while $(x-1)^2$ has second derivative $2$.

For comparison (3):
- **At $t = 0$,** $\log(1 + \chi^2) = \log\frac1{\pi_1} = D$, so the first inequality is an equality. This holds for every point mass on every chain, since $D(\delta_x\Vert\pi) = \log\frac1{\pi_x} = \log(1 + \chi^2(\delta_x\Vert\pi))$. The second inequality, $\log(1 + \chi^2) \le \chi^2$, is very loose when $\pi_1$ is small: $\log\frac1{\pi_1}$ versus $\frac1{\pi_1}$.
- **As $t \to \infty$,** both $\log(1 + \chi^2)$ and $\chi^2$ are $\sim\chi^2$, while $D \sim \frac12\chi^2$. Comparison (3) loses a factor of $2$.

**(f)** Starting from $\delta_2$ gives, by the symmetric computation, $\|\cdot\|_{TV} = s\pi_1$. So
$$
d(t) = e^{-\lambda t}\max(\pi_1, \pi_2), \qquad t_{\rm mix}(\varepsilon) = \frac1\lambda\log\frac{\max(\pi_1,\pi_2)}{\varepsilon} = \frac1\lambda\Big(\log\frac1{2\varepsilon} + \log\big(2\max(\pi_1,\pi_2)\big)\Big),
$$
whenever $\max(\pi_1,\pi_2) \ge \varepsilon$. Since $2\max(\pi_1,\pi_2) \in [1, 2]$, the **lower bound** of §5.1, $\frac1\lambda\log\frac1{2\varepsilon}$, is tight up to an additive $\frac{\log 2}\lambda$.

The **upper bound** of §5.1 is $\frac1\lambda\big(\frac12\log\frac1{\pi_{\min}} + \log\frac1{2\varepsilon}\big)$. As $\pi_1 \to 0$, it tends to infinity while the true $t_{\rm mix}$ stays bounded. With only two states, there is no geometry to traverse. The entire loss comes from the initial value $\chi^2(\delta_1\Vert\pi) = \frac1{\pi_1} - 1$, which is huge even though the initial TV distance is at most $1$. This is the phenomenon of §5.1 in its simplest form.

---

## Exercise 9

**(a)** Vertices other than $u, v$ have degree $m - 1$. The vertices $u$ and $v$ have degree $m$. So
$$
\mathrm{vol}(V) = 2m(m-1) + 2 = 2(m^2 - m + 1).
$$
Let $f = +1$ on the first clique and $-1$ on the second. By the symmetry swapping the cliques, $\mathbb E_\pi f = 0$, so $\operatorname{Var}_\pi f = \mathbb E_\pi f^2 = 1$. Only the bridge sees a difference, of size $2$, so
$$
\mathcal E(f,f) = \frac{1}{\mathrm{vol}(V)}\cdot 2^2 = \frac{4}{\mathrm{vol}(V)} .
$$
Therefore $\lambda \le \frac{4}{\mathrm{vol}(V)} = \frac{2}{m^2 - m + 1}$.

**(b)** We use $\lambda \ge 1/\kappa_\pi$ with unit lengths, where
$$
\kappa_\pi = \max_e \frac{1}{Q(e)}\cdot\frac12\sum_{(a,b):\,e\in\gamma_{ab}}\pi_a\pi_b\,|\gamma_{ab}|, \qquad Q(e) = \frac{1}{\mathrm{vol}(V)}.
$$
Every path has length at most $3$. Every vertex has $\pi_a \le \frac{m}{\mathrm{vol}(V)}$. Write $\mathrm{vol} = \mathrm{vol}(V)$.

- **The bridge $\{u,v\}$.** It is used by every ordered pair with endpoints in different cliques: $2m^2$ pairs. Its load is
$$
\mathrm{vol}\cdot\frac12\cdot 2m^2\cdot\frac{m^2}{\mathrm{vol}^2}\cdot 3 = \frac{3m^4}{\mathrm{vol}} .
$$
- **A clique edge $\{a, u\}$, $a \ne u$** (and symmetrically $\{a, v\}$). It is used by the pairs $(a,u)$ and $(u,a)$, with length $1$. It is also used by $(a, b)$ and $(b, a)$ for each $b$ in the other clique, with length at most $3$: $2m$ pairs. Pairs starting at $u$ do not use it. Its load is at most
$$
\mathrm{vol}\cdot\frac12\Big(2\cdot 1 + 2m\cdot 3\Big)\frac{m^2}{\mathrm{vol}^2} = \frac{m^2 + 3m^3}{\mathrm{vol}} .
$$
- **Any other clique edge $\{a, c\}$.** It is used only by $(a,c)$ and $(c,a)$. Its load is at most $\frac{m^2}{\mathrm{vol}}$.

For $m \ge 2$, $3m^4 \ge 3m^3 + m^2$, because $m^2(3m^2 - 3m - 1) \ge 0$. So the bridge is the most congested edge, and
$$
\kappa_\pi \le \frac{3m^4}{\mathrm{vol}(V)}, \qquad \lambda \ge \frac{\mathrm{vol}(V)}{3m^4} = \frac{2(m^2 - m + 1)}{3m^4}.
$$

**(c)** Combining (a) and (b),
$$
\frac{2(m^2 - m + 1)}{3m^4} \le \lambda \le \frac{2}{m^2 - m + 1},
$$
so $\lambda = \Theta(1/m^2)$, with constants $\approx \frac23$ and $2$. Numerically, $m^2\lambda \approx 1.86$ at $m = 10$ and $1.96$ at $m = 50$, so the test function is nearly sharp.

By §5.1, with $\pi_{\min} = \frac{m-1}{\mathrm{vol}(V)}$,
$$
\frac{\log 2}{\lambda} \ \ge\ \frac{\log 2}{2}(m^2 - m + 1) \quad\text{is a lower bound on } t_{\rm mix},
$$
$$
t_{\rm mix} \le \frac{1}{\lambda}\Big(\tfrac12\log\frac{1}{\pi_{\min}} + \log 2\Big) = O(m^2\log m).
$$
The truth is $\Theta(m^2)$. To cross, the walk has to be at $u$ (probability about $1/m$ under $\pi$) and then pick the bridge (probability $1/m$).

**Comparison with $K_{2m}$.** $K_{2m}$ has $m(2m-1)$ edges, and the barbell has $m^2 - m + 1$, about half as many. But $\lambda(K_{2m}) = \frac{2m}{2m-1} \approx 1$, while $\lambda(B_m) \approx 2/m^2$. Removing the $m^2 - 1$ edges between the halves, all but one, costs a factor of order $n^2$ in the spectral gap.

> **Preview (Cheeger).** The cut between the cliques has conductance $\Phi = \frac{1}{\mathrm{vol}(V)/2} = \frac{2}{\mathrm{vol}(V)}$, and the test function in (a) is exactly the bound $\lambda \le 2\Phi$. Cheeger's lower bound $\lambda \ge \Phi^2/2$ would give only $\Omega(1/m^4)$. Here canonical paths beat it and show that the easy side of Cheeger is the tight one. The path graph is the opposite case, where $\lambda = \Theta(\Phi^2)$.
