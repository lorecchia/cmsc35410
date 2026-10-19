# Solution: Three Clusters and the Spectral Embedding

Throughout, $v_c:=\mathrm{vol}(C_i)=m(m-1)+2$ (each clique vertex has degree $m-1$, and the six bridge endpoints, two per clique, have one more), and $\mathrm{vol}(V)=3v_c$.

## (a) The 3-way conductance

Take $S_i=C_i$. Two bridge edges leave each clique, so $\Phi(C_i)=\dfrac{2}{\min\{v_c,2v_c\}}=\dfrac2{v_c}$ for all $i$, and
$$
\rho_3(G_m)\le\frac2{m(m-1)+2}:\qquad m=8:\ \frac2{58}=0.0345;\qquad m=12:\ \frac2{134}=0.0149 .
$$

## (b) The spectrum

Let $x$ be constant on cliques, $x=y_i$ on $C_i$, with $\sum_iy_i=0$ (so $x\perp_d\mathbf 1$, as all cliques have equal volume). The only edges with $x_i\ne x_j$ are the three bridges: $\mathcal E(x,x)=\sum_{i}(y_i-y_{i+1})^2=2\sum y_i^2-2\sum y_iy_{i+1}$. Since $0=(\sum y_i)^2=\sum y_i^2+2\sum_{i<j}y_iy_j$ and for three points $\sum_{i<j}y_iy_j=\sum_iy_iy_{i+1}$, we get $\sum y_iy_{i+1}=-\frac12\sum y_i^2$, and $\mathcal E=3\sum_iy_i^2$. Also $\|x\|_d^2=v_c\sum_iy_i^2$. So $R(x)=3/v_c$ for **every** such $y$, and the 2-dimensional space of such $x$ shows $\lambda_2,\lambda_3\le3/v_c$ (the min-max theorem, with a 2-dimensional subspace on which $R\le3/v_c$).

**Symmetry.** The rotation $\sigma:C_i\to C_{i+1}$ (preserving the bridge pattern) commutes with $L_{rw}$, so the eigenspaces of $\sigma$ (with eigenvalues $1,\omega,\bar\omega$, $\omega=e^{2\pi i/3}$) are invariant. $L_{rw}$ is real, so the $\omega$- and $\bar\omega$-isotypic parts are complex conjugates and have the same spectrum. The test function above lies in the nontrivial part. If the Fiedler eigenvector lies in the $\omega$-part, then it is paired with an eigenvector in the $\bar\omega$-part with the same eigenvalue, so $\lambda_2=\lambda_3$ (real eigenspace of dimension $2$, containing $\mathrm{Re}$ and $\mathrm{Im}$ of the complex one).

Numerically:

| $m$ | $\lambda_1,\dots,\lambda_5$ | $3/v_c$ |
|---|---|---|
| 8 | 0, 0.0430, 0.0430, 1.010, 1.010 | 0.0517 |
| 12 | 0, 0.0195, 0.0195, 1.004, 1.004 | 0.0224 |

So $\lambda_2=\lambda_3$ (to the printed precision), below $3/v_c$, and $\lambda_4$ jumps to $\approx1$.

## (c) One eigenvector, one cut

For random $f=a f_2+bf_3$ ($a,b$ Gaussian), the best sweep conductance is $0.0345=2/58$ for $m=8$ in all $8$ trials, and $0.0149=2/134$ for $m=12$: the sweep always **isolates one clique** (cutting its two bridge edges). Indeed $f$ restricted to the cliques is nearly the constant vector $(y_0,y_1,y_2)$ with mean zero; sorting it puts one clique at the bottom and one at the top, and the threshold set $\{f<t\}$ for $t$ between the lowest and middle clique is that one clique. So the best sweep cut has $\Phi=2/v_c$ and no sweep cut does better; in particular $\Phi(G_m)\le2/v_c$, and this is the optimal cut one expects, since any set that cuts *into* a clique pays $\Omega(m)$ edges per vertex moved.

This is a **good 2-way cut**, but it does not give a 3-way partition: it returns one clique and the union of the other two. A single real eigenvector with a threshold always produces two parts. To get three parts we need both eigenvectors, i.e. the spectral embedding.

## (d) The spectral embedding

With $F(i)=(f_2(i),f_3(i))$:

| $m$ | distances between cluster centers | max distance of a point to its center |
|---|---|---|
| 8 | 0.186, 0.186, 0.186 | 0.016 |
| 12 | 0.122, 0.122, 0.122 | 0.008 |

The three clusters sit at the vertices of an **equilateral triangle** (by the symmetry $\sigma$, which acts on the plane as a rotation by $120^\circ$), and each cluster is tight: the radius is $\approx12$ times smaller than the distance between centers for $m=8$, and $\approx15$ times smaller for $m=12$. Rounding by angle (or $k$-means with $k=3$) recovers the cliques exactly, and the resulting partition has all $\Phi(S_i)=2/v_c=\rho_3$ (optimal).

## (e) The eigengap

$\lambda_4=1.010$ ($m=8$) and $1.004$ ($m=12$). By $\rho_4\ge\lambda_4/2$: every 4-way partition has a part of conductance at least $0.505$ ($m=8$), resp. $0.502$ ($m=12$). Compare with $\rho_3\le0.0345$ ($m=8$), resp. $0.0149$ ($m=12$): a factor $\approx15$ to $\approx34$. There are three clusters and not four, and the spectrum says so: $\lambda_3$ is tiny and $\lambda_4$ is large. This is the **eigengap heuristic** for choosing $k$. (Consistently, splitting a clique into two parts costs $\approx m/2$ edges per vertex, so conductance $\approx1/2$.)

## (f) Discussion

1. In the proof of the one-cut Cheeger inequality, there is one function, and a sweep over its level sets is the natural way to round. For $k$ clusters, the relaxation is a map $V\to\mathbb R^k$ (the spectral embedding), and rounding requires clustering a $k$-dimensional point cloud. One route is to show that the embedded points are spread out in the weighted $\ell_2$ sense (small $\lambda_k$ forces $k$ well-separated masses) and then use a random-partitioning or random-projection argument to pull out $k$ disjoint sets of small conductance; these steps lose polynomial factors in $k$, which is where the $O(k^2)$ in $\rho_k\le O(k^2)\sqrt{\lambda_k}$ comes from (see the papers cited in the statement for the exact proofs). In the example above the loss is nil, because the clusters are tight and symmetric.

2. Repeated two-way sweeps: the first sweep cuts off one clique (conductance $2/v_c$), and the recursion on the remaining two cliques (joined by one bridge) cuts them apart at the same conductance $2/v_c$, a perfect result here. In general, recursion can fail: the sub-cut of the second step is performed in the subgraph, where the volumes and conductances change, and a bad first cut (e.g. one that splits a cluster) cannot be undone. The $k$-dimensional embedding avoids this by finding all $k$ parts simultaneously, using the information of all $k$ eigenvectors.
