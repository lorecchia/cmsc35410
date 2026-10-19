# Solution: Why the Median?

Let $a=900$, $b=10$; $A$ is the large clique and $B$ the small one. The bridge joins a vertex of $A$ to a vertex of $B$.

## (a) Basic quantities

Every vertex of a clique has degree $a-1$ (resp. $b-1$), except the two bridge endpoints, which have one more. So
$$
\mathrm{vol}(A)=a(a-1)+1=809{,}101,\qquad\mathrm{vol}(B)=b(b-1)+1=91 .
$$
The set $B$ has one cut edge and the smaller volume: $\Phi(B)=1/91=0.01099$. Numerically $\lambda=0.01008$, so the easy side gives $\Phi(S)\ge\lambda/2=0.00504$ for **every** cut $S$.

## (b) A vector with mean zero

With $c=\mathrm{vol}(A)/\mathrm{vol}(B)=8891.2$, $x=\mathbf 1_A-c\mathbf 1_B$ has $\langle x,\mathbf 1\rangle_d=\mathrm{vol}(A)-c\,\mathrm{vol}(B)=0$. Only the bridge contributes to $N(x)=\sum_Ew(x_i-x_j)^2=(1+c)^2$, and $\|x\|_d^2=\mathrm{vol}(A)+c^2\mathrm{vol}(B)$. Since $\mathrm{vol}(A)=c\,\mathrm{vol}(B)$, $\|x\|_d^2=c(1+c)\mathrm{vol}(B)$, so
$$
R_2(x)=\frac{(1+c)^2}{c(1+c)\mathrm{vol}(B)}=\frac{1+c}{c\,\mathrm{vol}(B)}=\frac1{\mathrm{vol}(B)}+\frac1{\mathrm{vol}(A)}=0.010990 .
$$
This is $\ge\lambda=0.010082$ and very close to it, as expected: $x$ is essentially the Fiedler vector.

## (c) The naive rounding fails

$x_+=\mathbf 1_A$. Only the bridge contributes: $N(x_+)=1$ and $\|x_+\|_d^2=\mathrm{vol}(A)$, so
$$
R_2(x_+)=\frac1{\mathrm{vol}(A)}=1.24\times10^{-6},\qquad R_1(x_+^2)=\frac{1}{\mathrm{vol}(A)}\ \text{(same computation, since }x_+^2=x_+).
$$

1. $\sqrt{2R_2(x_+)}=1.57\times10^{-3}$, while every cut has $\Phi\ge\lambda/2=5.04\times10^{-3}$. So **no** set $S_t$ can satisfy $\Phi(S_t)\le\sqrt{2R_2(x_+)}$: the claim is false for $x_+$, by a factor $\ge3$ (and the factor is unbounded as $a\to\infty$ with $b$ fixed, since $R_2(x_+)\to0$ while $\lambda\to1/\mathrm{vol}(B)$).

2. Lemma 2 holds for any $x\ge0$ (it never uses the support): $R_1(x_+^2)\le\sqrt{2R_2(x_+)}$. Lemma 1 also holds for any $x\ge0$: it gives a threshold set with $w(S,\bar S)/\mathrm{vol}(S)\le R_1$. Here the sweep returns $S=A$, with $w(S,\bar S)/\mathrm{vol}(S)=1/\mathrm{vol}(A)$, which is indeed $\le R_1$. What breaks is the identification
$$
\frac{w(S,\bar S)}{\mathrm{vol}(S)}=\Phi(S)=\frac{w(S,\bar S)}{\min\{\mathrm{vol}(S),\mathrm{vol}(\bar S)\}},
$$
which needs $\mathrm{vol}(S)\le\mathrm{vol}(V)/2$. Here $S=A$ has almost all the volume, and $\Phi(A)=1/91$, not $1/\mathrm{vol}(A)$.

The uncentered quotient $R_2$ can be arbitrarily small for a nonnegative vector with huge support (the constant vector has $R_2=0$). It only measures the *Poincaré quotient* when the vector is orthogonal to constants, or when its support is small.

## (d) The median fixes it

$\pi(A)>\frac12$ and $x=1$ on $A$, so the $\pi$-median is $m=1$. Then $\bar x=x-\mathbf 1=0$ on $A$ and $-(1+c)$ on $B$. Its positive part is $p=0$, and its negative part is $q=(1+c)\mathbf 1_B$. Only the bridge contributes to $N(q)=(1+c)^2$, and $\|q\|_d^2=(1+c)^2\mathrm{vol}(B)$, so
$$
R_2(q)=\frac1{\mathrm{vol}(B)}=0.010989\ \le\ R_2(x)=0.010990 .
$$
($R_2(p)$ is undefined, as $p=0$: the minimum is over nonzero parts.) We have $\min\{R_2(p),R_2(q)\}=R_2(q)\le R_2(x)$, as Lemma 3 asserts. The sweep of $q^2=(1+c)^2\mathbf 1_B$ returns $S=B$, whose conductance is exactly $\Phi(B)=1/91=0.01099\le\sqrt{2R_2(x)}=0.148$, the Cheeger guarantee (and it is the optimal cut).

## (e) Mean versus median

Centering at the **mean** gives $\sum_id_i\bar f_i=0$, but it says nothing about the *sign pattern*: a vector with mean zero can be positive on a set carrying almost all the measure and very negative on a small one (as $x$ above). Then $p=\bar f_+$ has large support, and the sweep sets of $p^2$ can have $\mathrm{vol}(S_t)>\frac12\mathrm{vol}(V)$. Centering at the **median** $m$ gives $\pi(\bar f>0)\le\frac12$ and $\pi(\bar f<0)\le\frac12$ by definition of the median, so both parts have small support.

**Example on $P_3$.** $d=(1,2,1)$, $\mathrm{vol}=4$. Take $y=(-a,1,b)$ with $b>0$. $\langle y,\mathbf 1\rangle_d=-a+2+b=0$ gives $a=2+b$. For $b=1$: $y=(-3,1,1)$ and $\pi(\{y>0\})=(2+1)/4=\tfrac34>\tfrac12$.

## (f) What exactly is the median used for?

For *any* shift $\bar f=f-s\mathbf 1$ with $\bar f=p-q$ ($p,q\ge0$ with disjoint supports), the edge-wise inequality $(x-y)^2\ge x^2+y^2$ (since $xy=-(p_iq_j+p_jq_i)\le0$) gives $N(p-q)\ge N(p)+N(q)$, and $D(p-q)=D(p)+D(q)$, so $R_2(\bar f)\ge\min\{R_2(p),R_2(q)\}$. None of this uses the median.

- **The median** is used for the **support condition**: it guarantees $\pi(\{p>0\})\le\frac12$ and $\pi(\{q>0\})\le\frac12$, whichever part we keep. This is what makes $w(S_t,\bar S_t)/\mathrm{vol}(S_t)=\Phi(S_t)$ in Lemma 1.
- **The identity $\sum_id_if_i=0$** is used to show that the shift does not increase the quotient: $\sum_id_i\bar f_i^2=\sum_id_if_i^2+s^2\mathrm{vol}(V)\ge\sum_id_if_i^2$ (the cross term vanishes), so $R_2(\bar f)\le R_2(f)$.

Both are needed, and the median is the only shift that gives the support condition for *both* parts simultaneously. (Any other $s$ would give the condition for at most one of the two parts, and then the "better of $p,q$" might be the one with large support.)
