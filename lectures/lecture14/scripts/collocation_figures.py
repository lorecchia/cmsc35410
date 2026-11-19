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

def bernstein_rho(z0):
    """Parameter of the Bernstein ellipse through z0: |z0 + sqrt(z0^2 - 1)|, branch with modulus >= 1."""
    w = z0 + np.sqrt(complex(z0) ** 2 - 1); return max(abs(w), 1 / abs(w))

def cheb_coeffs(f, n=400):
    """Chebyshev coefficients a_k from samples at x_j = cos(j pi/n) (DCT-I), k = 0..n."""
    th = np.pi * np.arange(n + 1) / n; v = f(np.cos(th))
    a = np.real(np.fft.fft(np.r_[v, v[-2:0:-1]]))[: n + 1] / n; a[0] /= 2; a[-1] /= 2
    return a

def fig_bernstein():
    fns = [(r"$1/(1+25x^2)$, poles $\pm\frac{i}{5}$", lambda x: 1 / (1 + 25 * x ** 2), 0.2j, BLUE),
           (r"$1/(1+400x^2)$, poles $\pm\frac{i}{20}$", lambda x: 1 / (1 + 400 * x ** 2), 0.05j, ORANGE),
           (r"$1/(1.05-x)$, pole $1.05$", lambda x: 1 / (1.05 - x), 1.05, GREEN)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4)); ax = axes[0]; t = np.linspace(0, 2 * np.pi, 400)
    N = 12; th = np.pi * np.arange(N + 1) / N
    ax.plot(np.cos(t[:200]), np.sin(t[:200]), color=GREY, lw=0.8, ls=":")
    for xj, yj in zip(np.cos(th), np.sin(th)): ax.plot([xj, xj], [0, yj], color=GREY, lw=0.6, ls=":")
    ax.plot(np.cos(th), np.sin(th), "o", color=GREY, ms=3); ax.plot(np.cos(th), 0 * th, "o", color=INK, ms=4)
    ax.plot([-1, 1], [0, 0], color=INK, lw=2.5)
    for lab, f, z0, c in fns:
        rho = bernstein_rho(z0); a, b = (rho + 1 / rho) / 2, (rho - 1 / rho) / 2
        ax.plot(a * np.cos(t), b * np.sin(t), color=c, lw=1.8, label=fr"$\rho={rho:.3f}$")
        ax.plot([np.real(z0)] * (2 if np.imag(z0) else 1), [np.imag(z0), -np.imag(z0)][: 2 if np.imag(z0) else 1], "x", color=c, ms=9, mew=2.2)
    ax.set_aspect("equal"); ax.set_xlim(-1.45, 1.45); ax.set_ylim(-0.5, 1.08); ax.spines["left"].set_visible(False); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=10, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.32))
    ax.set_title(r"Bernstein ellipses through the nearest singularity ($\times$)", fontsize=12)
    ax = axes[1]; k = np.arange(0, 161)
    for lab, f, z0, c in fns:
        a = np.abs(cheb_coeffs(f))[: len(k)]; rho = bernstein_rho(z0); m = a > 1e-15
        ax.semilogy(k[m], a[m], ".", color=c, ms=4, label=lab)
        ax.semilogy(k, a[2] * rho ** (2.0 - k), "--", color=c, lw=1, alpha=0.8)
        kk = np.arange(20, 60, 2); kk = kk[a[kk] > 1e-13]
        print(f"Bernstein: {lab}: rho = {rho:.4f}, fitted decay rate of |a_k| = {np.exp(-np.polyfit(kk, np.log(a[kk]), 1)[0]):.4f}, digits per node = {np.log10(rho):.3f}")
    ax.set_ylim(1e-16, 10); ax.set_xlabel("$k$"); ax.set_ylabel(r"$|a_k|$"); ax.legend(frameon=False, fontsize=10)
    ax.set_title(r"Chebyshev coefficients decay like $\rho^{-k}$ (dashed)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "bernstein.svg"), bbox_inches="tight")
    if PREV: fig.savefig(PREV + "-bernstein.png", dpi=70, bbox_inches="tight")
    # endpoint vs middle asymptotics and the strip <-> ellipse correspondence
    for d in (1e-2, 1e-3): print(f"delta={d}: rho(1+delta) = {bernstein_rho(1 + d):.4f} vs 1+sqrt(2 delta) = {1 + np.sqrt(2 * d):.4f};  rho(i delta) = {bernstein_rho(1j * d):.5f} vs 1+delta")
    al = 0.3; print("cos(theta + i alpha) lies on E_rho with rho = e^alpha:", np.allclose([bernstein_rho(np.cos(th0 + 1j * al)) for th0 in (0.3, 1.0, 2.5)], np.exp(al)))

def fig_aliasing():
    """A high mode T_l (l > N) sampled at N+1 nodes, and the degree-N polynomial the samples are mistaken for."""
    T = lambda l, x: np.cos(l * np.arccos(np.clip(x, -1, 1))); xx = np.linspace(-1, 1, 4001)
    N, l = 12, 16
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, (name, xn, c) in zip(axes[:2], (("equispaced", np.linspace(-1, 1, N + 1), ORANGE),
                                            ("Chebyshev", np.cos(np.pi * np.arange(N + 1) / N), BLUE))):
        q = interp(xn, T(l, xn), xx)
        ax.plot(xx, T(l, xx), color=GREY, lw=1.2, alpha=0.8, label=fr"high mode $T_{{{l}}}$")
        ax.plot(xx, q, color=c, lw=2.2, label=fr"its alias: interpolant of degree $\leq{N}$")
        ax.plot(xn, T(l, xn), "o", color=INK, ms=5, zorder=5, label=f"samples at the {N+1} nodes")
        ax.set_ylim(-2.6, 2.6); ax.set_xlabel("$x$"); ax.axhline(0, color=GREY, lw=0.5)
        ax.set_title(f"{name} nodes, $N={N}$:  max of alias $= {abs(q).max():.3g}$", fontsize=12)
        print(f"aliasing figure: {name}, N={N}, l={l}: sup of interpolant = {abs(q).max():.3g}")
    axes[0].annotate("off the chart:\nreaches $\\pm70$", xy=(-0.93, 2.5), xytext=(-0.55, 1.9), fontsize=10, color=ORANGE,
                     arrowprops=dict(arrowstyle="->", color=ORANGE))
    axes[1].text(0.0, -2.35, r"$T_{16}=T_8$ at these nodes  ($16=2\cdot12-8$)", ha="center", fontsize=10, color=BLUE)
    from matplotlib.lines import Line2D; from matplotlib.legend_handler import HandlerTuple
    hs = [Line2D([], [], color=GREY, lw=1.2), (Line2D([], [], color=ORANGE, lw=2.2), Line2D([], [], color=BLUE, lw=2.2)),
          Line2D([], [], color=INK, marker="o", ls="", ms=5)]
    fig.legend(hs, [fr"high mode $T_{{{l}}}$", fr"its alias: the interpolant of degree $\leq{N}$", f"samples at the {N+1} nodes"],
               handler_map={tuple: HandlerTuple(ndivide=None)}, frameon=False, fontsize=11, ncol=3, loc="lower center", bbox_to_anchor=(0.36, -0.07))
    ax = axes[2]; N2 = 20; ls = np.arange(0, 61)
    for name, xn, c in (("equispaced", np.linspace(-1, 1, N2 + 1), ORANGE), ("Chebyshev", np.cos(np.pi * np.arange(N2 + 1) / N2), BLUE)):
        ax.semilogy(ls, [abs(interp(xn, T(m, xn), xx)).max() for m in ls], "o-", color=c, ms=3, lw=1.4, label=name)
    ax.axvline(N2, color=GREY, ls=":", lw=1); ax.text(N2 + 1, 1.6, "$l=N$", color=GREY, fontsize=10)
    ax.set_xlabel("mode $l$"); ax.set_ylabel(r"$\max_{[-1,1]}|I_NT_l|$"); ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title(f"size of the alias, $N={N2}$", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "aliasing.svg"), bbox_inches="tight")
    if PREV: fig.savefig(PREV + "-aliasing.png", dpi=70, bbox_inches="tight")

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
    fig_runge(); fig_bernstein(); fig_aliasing(); fig_eigs(); fig_sl(); fraction_check(); nonnormal()
    print("wrote figures/runge.svg, figures/bernstein.svg, figures/aliasing.svg, figures/cheb-eigs.svg, figures/sl-convergence.svg")
