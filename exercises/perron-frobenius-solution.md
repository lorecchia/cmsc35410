# Solution: The Perron–Frobenius Theorem

Throughout, $M$ is an $n \times n$ non-negative irreducible matrix. For vectors, $x \ge 0$ and $x > 0$ mean entrywise non-negative and entrywise positive. A non-negative, nonzero vector $x$ is **$\rho$-good** if $Mx \ge \rho x$ entrywise.

For $x \ge 0$ with $x \ne 0$, let
$$
S_x = \{\, i \in [n] : (Mx)_i = \rho\, x_i \,\}
$$
be the set of *tight* coordinates. The value of $\rho$ will be clear from context.

## 0. The graph interpretation

Let $G_M$ be the directed graph on $[n]$ with an arc $i \to j$ whenever $M_{ij} > 0$. Then
$$
(M^k)_{ij} = \sum_{i = u_0, u_1, \dots, u_k = j} M_{u_0 u_1} M_{u_1 u_2} \cdots M_{u_{k-1} u_k}.
$$
Every term is non-negative, so nothing cancels. Hence $(M^k)_{ij} > 0$ exactly when $G_M$ has a walk of length $k$ from $i$ to $j$. So $M$ is irreducible exactly when $G_M$ is strongly connected.

We will use irreducibility only in the following form.

**Lemma 0.** Let $S \subseteq [n]$ be nonempty with $S \neq [n]$. Then there exist $i \in S$ and $j \notin S$ with $M_{ij} > 0$.

*Proof.* Suppose not, so $M_{ij} = 0$ for all $i \in S$ and $j \notin S$. We show by induction on $k$ that $(M^k)_{ij} = 0$ for all $i \in S$, $j \notin S$. The case $k = 1$ is the assumption. For the inductive step,
$$
(M^{k+1})_{ij} = \sum_{l} M_{il}\,(M^k)_{lj}.
$$
If $l \notin S$, then $M_{il} = 0$. If $l \in S$, then $(M^k)_{lj} = 0$ by induction. So every term vanishes. Now pick any $i \in S$ and $j \notin S$. Then $(M^k)_{ij} = 0$ for every $k$, which contradicts irreducibility. $\blacksquare$

In graph terms, Lemma 0 says that a strongly connected graph has an arc leaving every nonempty proper subset of vertices.

Two consequences:

- Every row of $M$ has a nonzero entry. For $n \ge 2$, apply Lemma 0 to $S = \{i\}$. For $n = 1$, irreducibility says $M_{11} \ne 0$.
- In particular, $M\mathbf{1} > 0$.

## 1. The largest $\rho$ exists and is attained

Let $\Delta = \{x \ge 0 : \sum_i x_i = 1\}$. Every $\rho$-good vector can be rescaled into $\Delta$, so we may restrict attention to $\Delta$. Define
$$
\rho^* = \sup\{\rho : \text{some } x \in \Delta \text{ is } \rho\text{-good}\}.
$$

**The set is nonempty.** Every $x \in \Delta$ is $0$-good, since $Mx \ge 0$.

**The set is bounded above.** Let $R = \max_i \sum_j M_{ij}$ be the largest row sum. Suppose $x \in \Delta$ is $\rho$-good, and pick $i$ with $x_i = \max_j x_j > 0$. Then
$$
\rho\, x_i \le (Mx)_i = \sum_j M_{ij} x_j \le R\, x_i,
$$
so $\rho \le R$. Hence $\rho^* \le R < \infty$.

**The supremum is attained.** Take $\rho_k \to \rho^*$ and $x^{(k)} \in \Delta$ with $Mx^{(k)} \ge \rho_k x^{(k)}$. The simplex $\Delta$ is compact, so a subsequence converges to some $x^* \in \Delta$. Passing to the limit in the inequality gives $Mx^* \ge \rho^* x^*$. So $x^*$ is $\rho^*$-good.

> **Remark.** The restriction to non-negative, nonzero vectors matters. Without it, $x = 0$ is $\rho$-good for every $\rho$. Likewise $x = -\mathbf 1$ is $\rho$-good for every $\rho \ge R$, so the "largest $\rho$" would not exist.

From now on, $\rho$-good means $\rho^*$-good, and $S_x$ is taken with $\rho = \rho^*$.

## 2. Some $\rho^*$-good vector is an eigenvector (part (a))

**Step 2a: no $\rho^*$-good vector has $S_x = \emptyset$.** This is the argument from Hint 1. Suppose $x$ is $\rho^*$-good and $(Mx)_i > \rho^* x_i$ for every $i$. Set
$$
\epsilon = \frac{\min_i \big((Mx)_i - \rho^* x_i\big)}{\max_i x_i} > 0.
$$
Then $(Mx)_i \ge \rho^* x_i + \epsilon \max_j x_j \ge (\rho^* + \epsilon)\, x_i$ for all $i$. So $x$ is $(\rho^*+\epsilon)$-good, which contradicts the definition of $\rho^*$.

**Step 2b: shrinking the tight set.** This is the argument from Hint 2. Among all $\rho^*$-good vectors, pick an $x$ with $|S_x|$ as small as possible. By Step 2a, $S_x \ne \emptyset$. We claim $S_x = [n]$.

Suppose instead that $S_x \ne [n]$. By Lemma 0, there are $i \in S_x$ and $j \notin S_x$ with $M_{ij} > 0$. Let $e_j$ be the $j$-th standard basis vector, and for $\delta > 0$ set
$$
y = x + \delta e_j \ \ (\ge 0,\ \ne 0).
$$
Then
$$
My - \rho^* y = (Mx - \rho^* x) + \delta\,(M e_j - \rho^* e_j).
$$
Read off the coordinates:

- **Coordinate $j$:** $(My - \rho^* y)_j = \big((Mx)_j - \rho^* x_j\big) + \delta\,(M_{jj} - \rho^*)$. The first term is strictly positive because $j \notin S_x$. So this coordinate stays strictly positive for all small enough $\delta > 0$.
- **Coordinate $k \ne j$:** $(My - \rho^* y)_k = \big((Mx)_k - \rho^* x_k\big) + \delta M_{kj}$. Since $\delta M_{kj} \ge 0$, this is at least its old value, which is $\ge 0$. Coordinates that were not tight stay not tight.
- **Coordinate $i$:** $(My - \rho^* y)_i = 0 + \delta M_{ij} > 0$. It was tight and now it is not.

So for small $\delta > 0$, the vector $y$ is $\rho^*$-good and $S_y \subseteq S_x \setminus \{i\}$. This contradicts the minimality of $|S_x|$.

Hence $S_x = [n]$, that is,
$$
Mx = \rho^* x.
$$
So $\lambda := \rho^*$ is a real eigenvalue of $M$, with a non-negative, nonzero eigenvector $x$. This proves **(a)**.

## 3. The eigenvector is positive (part (c))

We prove the stronger statement in Hint 3.

**Lemma 3.** Every non-negative, nonzero eigenvector of $M$ is entrywise positive, whatever its eigenvalue.

*Proof.* Let $u \ge 0$, $u \ne 0$ with $Mu = \mu u$, and let $Z = \{i : u_i = 0\}$. Suppose, for contradiction, that $Z \ne \emptyset$. Then $Z \ne [n]$ because $u \ne 0$. For each $i \in Z$,
$$
0 = \mu u_i = (Mu)_i = \sum_{j \notin Z} M_{ij} u_j.
$$
Every term on the right is non-negative, and $u_j > 0$ for $j \notin Z$. So $M_{ij} = 0$ for all $i \in Z$ and $j \notin Z$, which contradicts Lemma 0. Therefore $u > 0$. $\blacksquare$

Applying Lemma 3 to the eigenvector $x$ from Section 2 gives $x > 0$, which proves **(c)**.

A consequence is that $\lambda > 0$. Indeed, $Mx > 0$ because every row of $M$ has a nonzero entry and $x > 0$. So $\lambda x = Mx > 0$.

## 4. $\lambda$ dominates every eigenvalue (part (b))

Let $\mu \in \mathbb{C}$ be any eigenvalue of $M$, with eigenvector $v \in \mathbb{C}^n$, $v \ne 0$. Let $w = |v|$ entrywise. Then $w \ge 0$ and $w \ne 0$. By the triangle inequality and $M \ge 0$,
$$
(Mw)_i = \sum_j M_{ij} |v_j| \;\ge\; \Big|\sum_j M_{ij} v_j\Big| = |(Mv)_i| = |\mu|\,|v_i| = |\mu|\, w_i .
$$
So $w$ is $|\mu|$-good. By the definition of $\rho^*$, this gives $|\mu| \le \rho^* = \lambda$, which proves **(b)**.

## 5. Non-negative eigenvectors are multiples of $x$ (part (d))

Fix the positive eigenvector $x$ from Sections 2–3. Let $u \ge 0$, $u \ne 0$ with $Mu = \mu u$. By Lemma 3, $u > 0$. In particular $\mu = (Mu)_i / u_i$ is real and non-negative for any $i$.

**Step 5a: $\mu = \lambda$.** Let
$$
s = \min_i \frac{u_i}{x_i} > 0,
$$
so $u \ge s\,x$ entrywise, with equality at some index $i_0$. Since $M \ge 0$, applying $M$ preserves the inequality:
$$
\mu\, u = Mu \;\ge\; s\,Mx = s\lambda\, x.
$$
At coordinate $i_0$ this reads $\mu\, u_{i_0} \ge \lambda\, s\, x_{i_0} = \lambda\, u_{i_0}$. Since $u_{i_0} > 0$, we get $\mu \ge \lambda$. By (b), $\mu \le |\mu| \le \lambda$. Hence $\mu = \lambda$.

**Step 5b: $u$ is a multiple of $x$.** Let $w = u - s\,x$, with $s$ as above. Then:

- $w \ge 0$, by the choice of $s$;
- $w_{i_0} = 0$;
- $Mw = \lambda u - s\lambda x = \lambda w$, using Step 5a.

If $w \ne 0$, then $w$ is a non-negative, nonzero eigenvector, so Lemma 3 says $w > 0$. That contradicts $w_{i_0} = 0$. Hence $w = 0$, so $u = s\,x$. This proves **(d)**. $\blacksquare$

The argument never uses left eigenvectors or $M^\top$. Comparing $u$ with $x$ through the largest $s$ with $u \ge s x$ does all the work.

## 6. Remarks

1. **The inequality in (b) can be tight.** For the irreducible matrix $M = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, the eigenvalues are $\pm 1$, so $|\mu| = \lambda$ for $\mu = -1$. The strict inequality $|\mu| < \lambda$ for $\mu \neq \lambda$ needs *aperiodicity* (primitivity) as well. For a random walk on an undirected graph, aperiodicity means the graph is not bipartite.

2. **The characterization of $\lambda$.** The proof shows
$$
\lambda = \max_{x \ge 0,\ x \ne 0}\ \min_{i : x_i > 0} \frac{(Mx)_i}{x_i}.
$$
This is the Collatz–Wielandt formula.

3. **Application to random walks.** Let $G$ be a connected weighted graph, and let $P = D^{-1}A$ be its row-stochastic random-walk matrix. Distributions multiply $P$ from the left, so a stationary distribution is a *left* eigenvector: $\pi^\top P = \pi^\top$. We therefore apply the theorem to $M = P^\top$.
   - **Irreducibility.** $P^\top$ has the same zero pattern as $A^\top = A$. Each edge of $G$ gives an arc in each direction, so the graph of $P^\top$ is strongly connected and $P^\top$ is irreducible.
   - **The Perron value is $1$.** Let $x > 0$ satisfy $P^\top x = \lambda x$. Since $P\mathbf 1 = \mathbf 1$,
   $$
   \lambda\, x^\top \mathbf 1 = x^\top P \mathbf 1 = x^\top \mathbf 1.
   $$
   Because $x^\top \mathbf 1 > 0$, this forces $\lambda = 1$.
   - **The stationary distribution.** $\pi = x / (x^\top \mathbf 1)$ satisfies $\pi^\top P = \pi^\top$. By (d) it is unique, and in fact $\pi \propto d$.

4. **Possible extension for a harder version.** Show that the eigenspace of $\lambda$ is one-dimensional. That is, *every* eigenvector for $\lambda$, including sign-changing and complex ones, is a multiple of $x$.
   - *Hint:* for complex $u = a + ib$, the real and imaginary parts $a$ and $b$ are each real eigenvectors for $\lambda$ or zero.
   - For a real $u$ that is not a multiple of $x$, flip its sign if needed so it has a positive entry. Then take $w = x - t u$ with $t = \min_{u_i > 0} x_i / u_i$. This $w$ is a non-negative, nonzero eigenvector with a zero coordinate, which contradicts Lemma 3.
