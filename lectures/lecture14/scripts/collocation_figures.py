"""Lecture 14, part 2 (spectral collocation): figures and numerical checks.
Run from lectures/lecture14:  python3 scripts/collocation_figures.py   (numpy, scipy, matplotlib).
Prints the numbers quoted on the slides."""
import os, numpy as np, matplotlib
from scipy.linalg import eig, eigh, eigh_tridiagonal
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, GREY, GREEN, BG, INK = "#2563eb", "#ea580c", "#8a857b", "#15803d", "#fbfaf7", "#1c1b19"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "svg.fonttype": "path"})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"); os.makedirs(OUT, exist_ok=True)
PREV = os.environ.get("PREVIEW")

def cheb(N):
    """Chebyshev differentiation matrix (Trefethen, Spectral Methods in MATLAB, cheb.m) and points x_j = cos(j pi/N)."""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
    dX = x[:, None] - x[None, :]
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(1))                      # negative-sum trick: D @ 1 = 0
    return D, x

def interp(xn, fn, xx):
    """Barycentric Lagrange interpolation (numerically stable)."""
    w = np.array([1 / np.prod(xn[j] - np.delete(xn, j)) for j in range(len(xn))])
    d = xx[:, None] - xn[None, :]; exact = np.isclose(d, 0)
    d[exact] = 1; t = w / d
    p = (t @ fn) / t.sum(1)
    rows = exact.any(1); p[rows] = fn[exact[rows].argmax(1)]
    return p

def lebesgue(xn, xx):
    L = np.zeros_like(xx)
    for j in range(len(xn)):
        others = np.delete(xn, j); L += np.abs(np.prod((xx[:, None] - others) / (xn[j] - others), axis=1))
    return L.max()

def fig_runge():
    f = lambda x: 1 / (1 + 25 * x ** 2); xx = np.linspace(-1, 1, 4001)
    Ns = np.arange(4, 61, 4); err_e, err_c = [], []
    for N in Ns:
        xe = np.linspace(-1, 1, N + 1); xc = np.cos(np.pi * np.arange(N + 1) / N)
        err_e.append(np.abs(interp(xe, f(xe), xx) - f(xx)).max()); err_c.append(np.abs(interp(xc, f(xc), xx) - f(xx)).max())
    for N in (10, 20, 40):
        xe = np.linspace(-1, 1, N + 1); xc = np.cos(np.pi * np.arange(N + 1) / N)
        print(f"Lebesgue constants N={N}: equispaced {lebesgue(xe, xx):.3g}, Chebyshev {lebesgue(xc, xx):.3f} (2/pi log N + 1 = {2/np.pi*np.log(N)+1:.3f})")
    rho = 0.2 + np.sqrt(1.04)                     # Bernstein ellipse through the poles +-i/5
    print(f"Runge: Chebyshev error at N=20, 40, 60: {err_c[4]:.2e}, {err_c[9]:.2e}, {err_c[14]:.2e}; predicted rate rho^-N with rho = {rho:.3f}")
    print(f"Runge: equispaced error at N=20, 40: {err_e[4]:.2e}, {err_e[9]:.2e}")
    N = 16; xe = np.linspace(-1, 1, N + 1); xc = np.cos(np.pi * np.arange(N + 1) / N)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3)); ax = axes[0]
    ax.plot(xx, f(xx), color=INK, lw=2, label=r"$f=1/(1+25x^2)$")
    ax.plot(xx, interp(xe, f(xe), xx), color=ORANGE, lw=1.6, label="equispaced, $N=16$"); ax.plot(xe, f(xe), "o", color=ORANGE, ms=4)
    ax.plot(xx, interp(xc, f(xc), xx), color=BLUE, lw=1.6, label="Chebyshev, $N=16$"); ax.plot(xc, f(xc), "o", color=BLUE, ms=4)
    ax.set_ylim(-0.6, 1.6); ax.set_xlabel("$x$"); ax.legend(frameon=False, fontsize=10, loc="upper center"); ax.set_title("polynomial interpolation", fontsize=12)
    ax = axes[1]
    ax.semilogy(Ns, err_e, "o-", color=ORANGE, lw=2, ms=4, label="equispaced")
    ax.semilogy(Ns, err_c, "o-", color=BLUE, lw=2, ms=4, label="Chebyshev")
    ax.semilogy(Ns, 2 * rho ** (-Ns.astype(float)), "--", color=GREY, lw=1.4, label=r"$\rho^{-N}$, $\rho=\frac{1}{5}+\sqrt{1.04}$")
    ax.set_xlabel("$N$"); ax.set_ylabel(r"$\max|f-p_N|$"); ax.legend(frameon=False, fontsize=10); ax.set_title("interpolation error", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "runge.svg"), bbox_inches="tight")
    if PREV: fig.savefig(PREV + "-runge.png", dpi=70, bbox_inches="tight")

def cheb_dirichlet_eigs(N):
    D, x = cheb(N); A = -(D @ D)[1:-1, 1:-1]                     # -u'' on [-1,1], u(+-1)=0
    lam = eig(A, right=False); return np.sort(lam.real), np.abs(lam.imag).max()

def fd_dirichlet_eigs(N):
    h = 2 / N; m = N - 1; j = np.arange(1, m + 1)
    return np.sort((2 - 2 * np.cos(np.pi * j / N)) / h ** 2)        # uniform grid, N-1 interior points

def fig_eigs():
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for N, ls in ((32, "-"), (64, "--")):
        lc, im = cheb_dirichlet_eigs(N); lf = fd_dirichlet_eigs(N); k = np.arange(1, N)
        ex = (k * np.pi / 2) ** 2
        ec = np.abs(lc - ex) / ex; ef = np.abs(lf - ex) / ex
        good = np.sum(ec < 1e-6); print(f"Chebyshev N={N}: {good}/{N-1} eigenvalues to rel. err 1e-6 (fraction {good/(N-1):.2f}; 2/pi = {2/np.pi:.2f}); max imag part {im:.1e}; largest eigenvalue / N^4 = {lc.max()/N**4:.4f}")
        print(f"   finite differences N={N}: {np.sum(ef < 1e-6)} eigenvalues to 1e-6; rel. error of lambda_1 = {ef[0]:.1e}, lambda_10 = {ef[9]:.1e}")
        ax.semilogy(k / N, np.maximum(ec, 1e-16), ls, color=BLUE, lw=2, label=f"Chebyshev, $N={N}$")
        ax.semilogy(k / N, np.maximum(ef, 1e-16), ls, color=ORANGE, lw=2, label=f"finite differences, $N={N}$")
    ax.axvline(2 / np.pi, color=GREY, ls=":", lw=1.2); ax.text(2 / np.pi - 0.17, 1e1, r"$k/N=2/\pi$", color=GREY)
    ax.set_xlabel("$k/N$"); ax.set_ylabel(r"relative error of $\lambda_k$"); ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(0.0, 0.62))
    ax.set_title(r"$-u''=\lambda u$ on $[-1,1]$, Dirichlet: $\lambda_k=(k\pi/2)^2$", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cheb-eigs.svg"), bbox_inches="tight")
    if PREV: fig.savefig(PREV + "-eigs.png", dpi=70, bbox_inches="tight")

def sl_cheb(N, p, r):
    """-(p u')' = lam r u on [0,1], Dirichlet, by Chebyshev collocation (map x = (t+1)/2, d/dx = 2 d/dt)."""
    D, t = cheb(N); x = (t + 1) / 2; D = 2 * D
    A = -(D @ np.diag(p(x)) @ D)[1:-1, 1:-1]; B = np.diag(r(x))[1:-1, 1:-1]
    lam = eig(A, B, right=False); return np.sort(lam.real)

def sl_fd(N, p, r, k=8):
    """Weighted path (finite differences), Dirichlet; symmetrized tridiagonal, lowest k eigenvalues."""
    h = 1 / N; x = np.arange(1, N) * h; xm = (np.arange(N) + 0.5) * h; c = p(xm) / h ** 2; s = 1 / np.sqrt(r(x))
    d = (c[:-1] + c[1:]) * s ** 2; e = -c[1:-1] * s[:-1] * s[1:]
    return eigh_tridiagonal(d, e, select="i", select_range=(0, min(k, N - 2) - 1), eigvals_only=True)

def fig_sl():
    r = lambda x: 1 + 0.5 * x
    smooth = lambda x: 1 + 0.8 * np.sin(2 * np.pi * x)
    jump = lambda x: np.where(x < 0.5, 1.0, 0.25)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for ax, p, name in ((axes[0], smooth, r"smooth $p=1+0.8\sin2\pi x$"), (axes[1], jump, r"discontinuous $p$: $1$ on $[0,\frac{1}{2})$, $\frac{1}{4}$ on $[\frac{1}{2},1]$")):
        ref = sl_cheb(160, p, r)[:5] if p is smooth else sl_fd(64000, p, r)[:5]
        Nc = np.arange(8, 81, 4); Nf = np.unique(np.round(np.logspace(np.log10(8), np.log10(4000), 14)).astype(int))
        for k, ls in ((0, "-"), (4, "--")):
            ec = [abs(sl_cheb(N, p, r)[k] - ref[k]) / ref[k] for N in Nc]
            ef = [abs(sl_fd(N, p, r)[k] - ref[k]) / ref[k] for N in Nf]
            ax.loglog(Nc, np.maximum(ec, 1e-16), ls, marker="o", ms=3, color=BLUE, lw=1.8, label=f"Chebyshev, $\\lambda_{k+1}$")
            ax.loglog(Nf, np.maximum(ef, 1e-16), ls, marker="s", ms=3, color=ORANGE, lw=1.8, label=f"finite differences, $\\lambda_{k+1}$")
            if k == 0: print(f"{name[:12]}: lambda_1 rel. error Chebyshev N=16,32,64: {ec[2]:.1e}, {ec[6]:.1e}, {ec[14]:.1e}; FD N~{Nf[3]},{Nf[7]}: {ef[3]:.1e}, {ef[7]:.1e}")
        ax.set_xlabel("$N$"); ax.set_ylabel("relative error"); ax.set_title(name, fontsize=12); ax.set_ylim(1e-16, 1)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sl-convergence.svg"), bbox_inches="tight")
    if PREV: fig.savefig(PREV + "-sl.png", dpi=70, bbox_inches="tight")

def fraction_check():
    for N in (32, 64, 128, 256):
        lc, _ = cheb_dirichlet_eigs(N); k = np.arange(1, N); ex = (k * np.pi / 2) ** 2; ec = np.abs(lc - ex) / ex
        print(f"fraction of Chebyshev eigenvalues accurate to 1e-6 / 1e-2: N={N}: {np.sum(ec < 1e-6)/(N-1):.2f} / {np.sum(ec < 1e-2)/(N-1):.2f}")

def nonnormal():
    for N in (16, 32, 64):
        D, x = cheb(N); A = -(D @ D)[1:-1, 1:-1]
        lam, V = eig(A); cond_V = np.linalg.cond(V)
        print(f"nonnormality N={N}: ||A A^T - A^T A|| / ||A||^2 = {np.linalg.norm(A @ A.T - A.T @ A) / np.linalg.norm(A) ** 2:.3f}, cond(eigenvectors) = {cond_V:.2e}, cond(A) = {np.linalg.cond(A):.2e}")

if __name__ == "__main__":
    fig_runge(); fig_eigs(); fig_sl(); fraction_check(); nonnormal()
    print("wrote figures/runge.svg, figures/cheb-eigs.svg, figures/sl-convergence.svg")
