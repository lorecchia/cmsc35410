# Solution: $\ell_1$ Contraction for Stochastic Matrices

Throughout, $P$ is row-stochastic: $P \ge 0$ and $P\mathbf 1 = \mathbf 1$. Distributions are row vectors acting on the left. Two facts are used repeatedly.

- **Products of stochastic matrices are stochastic.** If $P$ and $P'$ are stochastic, then $PP' \ge 0$ and $PP'\mathbf 1 = P\mathbf 1 = \mathbf 1$. In particular, every power $P^t$ is stochastic.
- **Stochastic matrices preserve the total sum.** $(v^\top P)\mathbf 1 = v^\top \mathbf 1$ for every $v$. In particular, $v^\top \mapsto v^\top P$ maps the subspace $\{v : v^\top \mathbf 1 = 0\}$ into itself.

## (a) Weak contraction

Let $v \in \mathbb{C}^n$. By the triangle inequality and $P \ge 0$,
$$
\|v^\top P\|_1 = \sum_j \Big| \sum_i v_i P_{ij} \Big| \le \sum_j \sum_i |v_i|\, P_{ij} = \sum_i |v_i| \sum_j P_{ij} = \sum_i |v_i| = \|v\|_1 .
$$

**The simplex is preserved.** Let $p \in \Delta$. Then $p^\top P \ge 0$, because $p \ge 0$ and $P \ge 0$. Also $p^\top P \mathbf 1 = p^\top \mathbf 1 = 1$.

**Eigenvalues.** $P$ and $P^\top$ have the same characteristic polynomial. So every eigenvalue $\mu$ of $P$ has a left eigenvector $v \ne 0$ with $v^\top P = \mu v^\top$. Then
$$
|\mu|\,\|v\|_1 = \|v^\top P\|_1 \le \|v\|_1,
$$
so $|\mu| \le 1$.

> **Remark.** For $v \ge 0$ the inequality is an equality, because there is no cancellation. So $P$ never shrinks non-negative vectors. All of the contraction happens on vectors with mixed signs, in particular on differences $p - p'$ of two distributions. This is why parts (c)–(e) restrict to $v^\top \mathbf 1 = 0$.

## (b) Existence of a stationary distribution

**$\bar p_T \in \Delta$.** Each $p^\top P^t$ lies in $\Delta$ by (a), and $\Delta$ is convex.

**The bound.** The sum telescopes:
$$
\bar p_T^\top P - \bar p_T^\top = \frac1T \sum_{t=0}^{T-1} \big(p^\top P^{t+1} - p^\top P^t\big) = \frac1T \big(p^\top P^T - p^\top\big).
$$
Both $p^\top P^T$ and $p^\top$ lie in $\Delta$, so their $\ell_1$ distance is at most $1 + 1 = 2$. Hence
$$
\|\bar p_T^\top P - \bar p_T^\top\|_1 \le \frac2T .
$$

**Taking a limit.** $\Delta$ is compact, so some subsequence $\bar p_{T_k}$ converges to a point $\pi \in \Delta$. The map $x^\top \mapsto x^\top P - x^\top$ is continuous, so
$$
\|\pi^\top P - \pi^\top\|_1 = \lim_{k \to \infty} \|\bar p_{T_k}^\top P - \bar p_{T_k}^\top\|_1 \le \lim_{k\to\infty} \frac{2}{T_k} = 0 .
$$
So $\pi^\top P = \pi^\top$. No irreducibility is needed for this part.

## (c) Strict contraction for positive matrices

**$n\delta \le 1$.** Each row of $Q$ has $n$ entries, each at least $\delta$, and the row sums to $1$.

**The case $n\delta = 1$.** Every entry equals $\delta = 1/n$, so $Q = U := \frac1n \mathbf 1 \mathbf 1^\top$. For $v^\top \mathbf 1 = 0$ we get
$$
v^\top Q = \tfrac1n (v^\top \mathbf 1)\, \mathbf 1^\top = 0,
$$
and the bound holds with both sides equal to zero.

**The case $n\delta < 1$.** Define
$$
R = \frac{Q - \delta\, \mathbf 1 \mathbf 1^\top}{1 - n\delta}, \qquad \text{so that} \qquad Q = n\delta\, U + (1 - n\delta)\, R .
$$
$R$ is stochastic:
- $R \ge 0$, because every entry of $Q$ is at least $\delta$;
- each row of $Q - \delta\mathbf 1\mathbf 1^\top$ sums to $1 - n\delta$, so each row of $R$ sums to $1$.

Now let $v^\top \mathbf 1 = 0$. Then $v^\top U = 0$, so $v^\top Q = (1 - n\delta)\, v^\top R$. By (a) applied to $R$,
$$
\|v^\top Q\|_1 = (1 - n\delta)\,\|v^\top R\|_1 \le (1 - n\delta)\,\|v\|_1 . \qquad \blacksquare
$$

> **Interpretation.** With probability $n\delta$, the chain $Q$ forgets its current state and jumps to a uniformly random state. If you start from two different distributions, that part of the mass ends up identical, which gives the factor $1 - n\delta$. This is the simplest instance of a *coupling* or *Doeblin* argument.

> **Alternative proof of strictness without the constant.** In the chain of inequalities in (a), equality forces all nonzero terms $v_i Q_{ij}$ in column $j$ to have the same sign, for every $j$. Suppose $Q > 0$, $v \ne 0$ and $v^\top \mathbf 1 = 0$. Then $v$ has both a positive and a negative entry, so every column mixes signs, and the inequality is strict.

## (d) Uniqueness for irreducible chains

**$Q$ is stochastic.** $\frac12(I + P)$ is non-negative, and each of its rows sums to $\frac12(1 + 1) = 1$. So it is stochastic, and hence so is its $(n-1)$-st power.

**$Q$ is positive.** $I$ and $P$ commute, so the binomial theorem gives
$$
Q = \frac{1}{2^{n-1}} \sum_{k=0}^{n-1} \binom{n-1}{k} P^k .
$$
All terms are non-negative. So $Q_{ij} > 0$ as soon as $(P^k)_{ij} > 0$ for a single $k \in \{0, \dots, n-1\}$.

- **If $i = j$:** take $k = 0$, since $(P^0)_{ii} = I_{ii} = 1$.
- **If $i \ne j$:** consider the directed graph with an arc $a \to b$ whenever $P_{ab} > 0$. As in the Perron–Frobenius exercise, $(P^k)_{ij} > 0$ exactly when there is a walk of length $k$ from $i$ to $j$. Irreducibility gives such a walk. A shortest such walk visits no vertex twice, so its length is at most $n - 1$.

Hence $Q > 0$ entrywise.

**Uniqueness.** A stationary distribution exists by (b). Every stationary distribution of $P$ is also stationary for $Q$: if $\pi^\top P = \pi^\top$, then $\pi^\top \cdot \frac12(I+P) = \pi^\top$, and so $\pi^\top Q = \pi^\top$.

Suppose $\pi$ and $\pi'$ are both stationary for $P$, and let $v = \pi - \pi'$. Then $v^\top \mathbf 1 = 1 - 1 = 0$ and $v^\top Q = v^\top$. Let $\delta = \min_{i,j} Q_{ij} > 0$. By (c),
$$
\|v\|_1 = \|v^\top Q\|_1 \le (1 - n\delta)\,\|v\|_1 .
$$
Since $1 - n\delta < 1$, this forces $\|v\|_1 = 0$, so $\pi = \pi'$.

**Positivity.** $\pi^\top = \pi^\top Q$ and $Q > 0$. So for every $j$,
$$
\pi_j = \sum_i \pi_i Q_{ij} \ge \delta \sum_i \pi_i = \delta > 0. \qquad \blacksquare
$$

## (e) Convergence

**The bound.** Let $p \in \Delta$ and $v = p - \pi$. Then:
- $v^\top \mathbf 1 = 0$;
- $\|v\|_1 \le 2$;
- $p^\top P^t - \pi^\top = v^\top P^t$, because $\pi^\top P^t = \pi^\top$.

Write $t = qm + r$ with $q = \lfloor t/m \rfloor$ and $0 \le r < m$. Then
$$
v^\top P^t = (v^\top P^r)\, (P^m)^q .
$$
The vector $v^\top P^r$ still sums to zero, and $\|v^\top P^r\|_1 \le \|v\|_1$ by (a). The matrix $P^m$ is stochastic with all entries at least $\delta$. Applying (c) to $P^m$ a total of $q$ times gives
$$
\|p^\top P^t - \pi^\top\|_1 \le (1 - n\delta)^q\, \|v^\top P^r\|_1 \le 2\,(1 - n\delta)^{\lfloor t/m \rfloor} .
$$

The stationary distribution $\pi$ here is unique: $P^m > 0$ implies that $P$ is irreducible, so (d) applies.

**Lazy chains.** If $P$ is irreducible, part (d) shows that $L = \frac12(I + P)$ satisfies $L^{n-1} > 0$. So the bound applies to $L$ with $m = n - 1$. $L$ has the same stationary distribution $\pi$ as $P$. Hence $p^\top L^t \to \pi^\top$ geometrically fast from every starting distribution.

**$P$ itself can fail.** Take $P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$. It is irreducible with $\pi = (\tfrac12, \tfrac12)$.
- Starting from $p = e_1$, the chain alternates between $e_1$ and $e_2$, so $\|p^\top P^t - \pi^\top\|_1 = 1$ for all $t$.
- No power of $P$ is positive.
- $P$ is an $\ell_1$ *isometry* on sum-zero vectors: $(1, -1)\,P = (-1, 1)$.

Periodicity is the obstruction. For the random walk on an undirected graph, this is the bipartite case.

## Remarks

1. **Connection to Perron–Frobenius.** For a stochastic $P$, parts (a), (b) and (d) are the Perron–Frobenius exercise applied to $M = P^\top$, with $\lambda = 1$:
   - $1$ is an eigenvalue, with right eigenvector $\mathbf 1$ and, by (b), a non-negative left eigenvector $\pi$;
   - every eigenvalue satisfies $|\mu| \le 1$;
   - for irreducible $P$, the non-negative left eigenvector is unique up to scaling, and it is positive.

   Here the arguments use only the triangle inequality and compactness.

2. **A spectral gap from contraction.** Let $Q$ be as in (c), and let $v^\top Q = \mu v^\top$ with $\mu \ne 1$. Then
$$
\mu\, v^\top \mathbf 1 = v^\top Q \mathbf 1 = v^\top \mathbf 1,
$$
so $v^\top \mathbf 1 = 0$. Part (c) then gives $|\mu| \le 1 - n\delta$.

   For complex $v = a + ib$, both $a$ and $b$ sum to zero. Applying (c) $k$ times to each gives
$$
|\mu|^k \|v\|_1 = \|v^\top Q^k\|_1 \le (1-n\delta)^k (\|a\|_1 + \|b\|_1) \le 2(1-n\delta)^k \|v\|_1 .
$$
   Taking $k$-th roots and letting $k \to \infty$ removes the factor of $2$.

   So an $\ell_1$ contraction factor bounds every non-trivial eigenvalue, which gives a spectral gap $1 - |\mu| \ge n\delta$.

3. **Dobrushin's coefficient.** The constant in (c) is not the best possible. The sharp one-step contraction factor on sum-zero vectors is
$$
\tau(P) = \tfrac12 \max_{i,k} \|e_i^\top P - e_k^\top P\|_1 = 1 - \min_{i,k} \sum_j \min(P_{ij}, P_{kj}).
$$
   This is the largest total-variation distance between two rows of $P$, that is, between the next-step distributions from two different states. The proof uses the same decomposition as (c), applied to each pair of rows.

4. **Total variation.** For distributions, $\|p - p'\|_1 = 2\, d_{\mathrm{TV}}(p, p')$. So (e) bounds the total-variation mixing time: $d_{\mathrm{TV}}(p^\top P^t, \pi^\top) \le \varepsilon$ once $\lfloor t/m \rfloor \ge \ln(1/\varepsilon)/(n\delta)$. This bound is usually far from tight. Spectral methods, later in the course, give much sharper mixing-time bounds for reversible chains.
