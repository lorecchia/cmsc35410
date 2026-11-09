"""Lecture 13 figures and numerical checks. Run from lectures/lecture13:  python3 scripts/make_figures.py
Needs numpy, scipy, matplotlib. Prints the numbers quoted on the slides.

Conventions (Lectures 4-5): continuous-time walk H_t = exp(-t L_rw); on the hypercube {0,1}^d a uniform
coordinate is flipped at rate 1, so lambda = 2/d; Dirichlet form E(f,g) = <f, L_rw g>_pi."""
import os, numpy as np, matplotlib
from scipy.stats import binom
from scipy.optimize import minimize, brentq
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, GREY, GREEN, BG, INK = "#2563eb", "#ea580c", "#8a857b", "#15803d", "#fbfaf7", "#1c1b19"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "svg.fonttype": "path"})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"); os.makedirs(OUT, exist_ok=True)
xlogx = lambda x: np.where(x > 0, x * np.log(np.where(x > 0, x, 1)), 0.0)

# ---------- two-point space: Ent(f^2) <= (a-b)^2/2, i.e. phi(s) <= 2 s^2 ----------
def phi(s):  # Ent(f^2) for f = (1+s, 1-s) under the uniform measure
    a2, b2 = (1 + s) ** 2, (1 - s) ** 2
    return 0.5 * (xlogx(a2) + xlogx(b2)) - xlogx(0.5 * (a2 + b2))
def dphi(s):
    return 2 * (1 + s) * np.log1p(s) - 2 * xlogx(1 - s) / np.where(s < 1, 1, 1) * 1 - 2 * s * np.log1p(s ** 2)
def checks_two_point():
    s = np.linspace(1e-4, 1 - 1e-9, 20001)
    ratio = phi(s) / (2 * s ** 2)
    num = (phi(s[1:]) - phi(s[:-1])) / (s[1:] - s[:-1]); mid = (s[1:] + s[:-1]) / 2
    formula = 2 * (1 + mid) * np.log1p(mid) - 2 * (1 - mid) * np.log(1 - mid) - 2 * mid * np.log1p(mid ** 2)
    print(f"two-point: max phi(s)/(2s^2) = {ratio.max():.6f} (at s->0), value at s=1: {phi(np.array(1.0)):.4f} (= 2 log 2 <= 2)")
    print(f"two-point: phi' formula check, max abs error {np.abs(num - formula).max():.2e}; max phi'(s)/(4s) = {(formula/(4*mid)).max():.6f}")
    # MLSI constant of the two-point chain: min_h E(h, log h)/Ent(h), E(f,g) = (f0-f1)(g0-g1)/2
    r = np.logspace(-6, 6, 4001)
    ent = 0.5 * (xlogx(r) + xlogx(np.ones_like(r))) - xlogx(0.5 * (r + 1))
    E = 0.5 * (r - 1) * np.log(r)
    ok = ent > 1e-14
    print(f"two-point: MLSI rho_0 = min E(h,log h)/Ent(h) = {np.min(E[ok]/ent[ok]):.5f} (claim 4); gap lambda = 2")
    return s, ratio

def hypercube_constants(d):
    """Brute-force alpha and rho_0 of the hypercube {0,1}^d (small d) by minimizing over positive h."""
    n = 2 ** d; X = np.array([[(v >> i) & 1 for i in range(d)] for v in range(n)])
    nbrs = [[v ^ (1 << i) for i in range(d)] for v in range(n)]
    def E(f, g):  # (1/n) * sum over ordered pairs x~y of (1/d)(f_x-f_y)(g_x-g_y) / 2
        tot = 0.0
        for v in range(n):
            for u in nbrs[v]: tot += (f[v] - f[u]) * (g[v] - g[u])
        return tot / (2 * n * d)
    def ent(h): m = h.mean(); return np.mean(xlogx(h)) - xlogx(m)
    rng = np.random.default_rng(0); best_a, best_r = np.inf, np.inf
    for trial in range(30):
        z0 = rng.normal(size=n) * (0.05 if trial < 15 else 2.0)
        fa = lambda z: E(np.exp(z / 2), np.exp(z / 2)) / max(ent(np.exp(z)), 1e-300)
        fr = lambda z: E(np.exp(z), z) / max(ent(np.exp(z)), 1e-300)
        best_a = min(best_a, minimize(fa, z0, method="BFGS").fun); best_r = min(best_r, minimize(fr, z0, method="BFGS").fun)
    return best_a, best_r

# ---------- hypercube mixing from a point mass ----------
def hypercube_curves(d, t):
    p = 0.5 * (1 - np.exp(-2 * t / d))                       # flip probability per coordinate
    D = d * (np.log(2) + xlogx(p) + xlogx(1 - p))            # d * KL(Bern(p) || Bern(1/2))
    chi2 = (1 + np.exp(-4 * t / d)) ** d - 1
    k = np.arange(d + 1)
    tv = np.array([0.5 * np.abs(binom.pmf(k, d, pp) - binom.pmf(k, d, 0.5)).sum() for pp in p])
    return D, chi2, tv


# ---------- the mismatch: experiments on the hypercube walk ----------
def simulate_tv(d, times, n_walks=20000, seed=0):
    """Monte Carlo: run n_walks continuous-time walks from a vertex; at each time, estimate TV to uniform
    from the empirical law of the Hamming distance K_t to the start (a sufficient statistic)."""
    rng = np.random.default_rng(seed); K = np.zeros(n_walks, dtype=int); t_prev = 0.0; out = []
    k = np.arange(d + 1); target = binom.pmf(k, d, 0.5)
    for t in times:
        jumps = rng.poisson(t - t_prev, size=n_walks); t_prev = t
        for j in range(jumps.max()):
            active = jumps > j
            up = rng.random(n_walks) < (d - K) / d          # flip a coordinate that still agrees with the start
            K = np.where(active, K + np.where(up, 1, -1), K)
        emp = np.bincount(K, minlength=d + 1) / n_walks
        out.append(0.5 * np.abs(emp - target).sum())
    return np.array(out)

def exact_tmix(d, eps=0.25):
    tv_at = lambda tt: hypercube_curves(d, np.array([tt]))[2][0]
    return brentq(lambda tt: tv_at(tt) - eps, 1e-3, 10 * d * np.log(d + 1))

def mismatch_figure():
    ds = [16, 32, 64, 128, 256]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]; cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(ds)))
    for d, c in zip(ds, cols):
        cut = d / 4 * np.log(d); x = np.linspace(0.02, 2.2, 300)
        ax.plot(x, hypercube_curves(d, x * cut)[2], color=c, lw=2, label=f"$d={d}$")
        xs = np.linspace(0.3, 2.0, 14); sim = simulate_tv(d, xs * cut, seed=d)
        ax.plot(xs, sim, "o", color=c, ms=4.5, mec="white", mew=0.6)
    ax.axvline(1, color=GREY, lw=1, ls=":"); ax.set_xlabel(r"$t\,/\,(\frac{d}{4}\log d)$")
    ax.set_ylabel("TV from a vertex"); ax.set_title("cutoff: lines exact, dots simulated (20,000 walks)", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax = axes[1]; dd = np.unique(np.round(np.logspace(np.log10(8), np.log10(1024), 15)).astype(int))
    tm = np.array([exact_tmix(d) for d in dd])
    ax.loglog(dd, dd / 2 * (dd * np.log(2) / 2 + np.log(2)), "--", color=ORANGE, lw=2, label=r"upper: $\chi^2$ + gap, $\Theta(d^2)$")
    ax.loglog(dd, tm, "-", color=INK, lw=2.4, label=r"truth: $t_{\rm mix}(\frac{1}{4})$")
    ax.loglog(dd, dd / 4 * np.log(dd), ":", color=BLUE, lw=2, label=r"$\frac{d}{4}\log d$")
    ax.loglog(dd, dd / 2 * np.log(2), "--", color=GREEN, lw=2, label=r"lower: $t_{\rm rel}\log 2=\frac{d}{2}\log 2$")
    sim_d = [16, 64, 256]; sim_t = []
    for d in sim_d:
        cut = d / 4 * np.log(d); xs = np.linspace(0.6, 1.8, 25); sim = simulate_tv(d, xs * cut, seed=7 * d)
        i = np.argmax(sim <= 0.25); sim_t.append(cut * (xs[i - 1] + (xs[i] - xs[i - 1]) * (sim[i - 1] - 0.25) / (sim[i - 1] - sim[i])))
    ax.loglog(sim_d, sim_t, "o", color=INK, ms=7, mec="white", label="simulated")
    ax.set_xlabel("dimension $d$"); ax.set_ylabel("time"); ax.set_title("mixing time vs. the bounds we have", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "hypercube-mismatch.svg"), bbox_inches="tight")
    if os.environ.get("PREVIEW"): fig.savefig(os.environ["PREVIEW"] + "-mismatch.png", dpi=70, bbox_inches="tight")
    for d, t in zip(sim_d, sim_t): print(f"mismatch: d={d}: simulated t_mix(1/4) = {t:.1f}, exact = {exact_tmix(d):.1f}, (d/4)log d = {d/4*np.log(d):.1f}, chi2 bound = {d/2*(d*np.log(2)/2+np.log(2)):.0f}, t_rel log2 = {d/2*np.log(2):.1f}")
    print("wrote figures/hypercube-mismatch.svg")

def main():
    s, ratio = checks_two_point()
    for d in (1, 2, 3):
        a, r = hypercube_constants(d)
        print(f"hypercube d={d}: alpha ~ {a:.4f} (claim 1/d = {1/d:.4f}), rho_0 ~ {r:.4f} (claim 4/d = {4/d:.4f})")
    d = 100; t = np.logspace(0, np.log10(3000), 800)
    D, chi2, tv = hypercube_curves(d, t)
    cut = d / 4 * np.log(d)
    D_mlsi = np.exp(-4 * t / d) * d * np.log(2)
    chi2_gap = np.exp(-4 * t / d) * (2.0 ** d - 1)
    tv_mlsi = np.minimum(1, np.sqrt(D_mlsi / 2)); tv_chi = np.minimum(1, 0.5 * np.sqrt(chi2_gap))
    tv_at = lambda tt: hypercube_curves(d, np.array([tt]))[2][0]
    t_exact = brentq(lambda tt: tv_at(tt) - 0.25, 1, 1000)
    t_mlsi = brentq(lambda tt: np.sqrt(np.exp(-4 * tt / d) * d * np.log(2) / 2) - 0.25, 1, 1000)
    t_chi = brentq(lambda tt: 0.5 * np.sqrt(np.exp(-4 * tt / d) * (2.0 ** d - 1)) - 0.25, 1, 5000)
    print(f"d={d}: cutoff (d/4)log d = {cut:.1f}; t_mix(1/4): exact {t_exact:.1f}, MLSI+Pinsker bound {t_mlsi:.1f}, chi2+gap bound {t_chi:.1f}")
    print(f"d={d}: L4 formula (d/4)(loglog(1/pi_min)+log(1/(2 eps^2))) at eps=1/4: {d/4*(np.log(d*np.log(2))+np.log(8)):.1f}; chi2 formula (d/2)(d log2/2 + log 2): {d/2*(d*np.log(2)/2+np.log(2)):.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    ax.loglog(t, D, color=BLUE, lw=2.2, label=r"$D(\delta_xH_t\Vert\pi)$ exact")
    ax.loglog(t, D_mlsi, "--", color=BLUE, lw=1.5, label=r"MLSI: $e^{-4t/d}\,d\log 2$")
    ax.loglog(t, chi2, color=ORANGE, lw=2.2, label=r"$\chi^2$ exact")
    ax.loglog(t, chi2_gap, "--", color=ORANGE, lw=1.5, label=r"gap: $e^{-4t/d}(2^d-1)$")
    ax.axvline(cut, color=GREY, lw=1, ls=":"); ax.text(cut * 1.02, 1e12, r"$\frac{d}{4}\log d$", color=GREY)
    ax.set_ylim(1e-3, 1e31); ax.set_xlabel("$t$ (log scale)"); ax.set_title(f"entropy vs. $\\chi^2$, $d={d}$", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax = axes[1]
    ax.semilogx(t, tv, color=INK, lw=2.2, label="TV exact")
    ax.semilogx(t, tv_mlsi, "--", color=BLUE, lw=1.8, label="MLSI + Pinsker")
    ax.semilogx(t, tv_chi, "--", color=ORANGE, lw=1.8, label=r"$\chi^2$ + gap")
    ax.axvline(cut, color=GREY, lw=1, ls=":"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("$t$ (log scale)"); ax.set_title("total variation from a vertex", fontsize=13); ax.legend(frameon=False, fontsize=10, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "hypercube-mixing.svg"), bbox_inches="tight")
    if os.environ.get("PREVIEW"): fig.savefig(os.environ["PREVIEW"] + "-mix.png", dpi=70, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(s, phi(s), color=BLUE, lw=2.2, label=r"$\mathrm{Ent}(f^2)$")
    ax.plot(s, 2 * s ** 2, "--", color=ORANGE, lw=2, label=r"$\frac{1}{2}(a-b)^2=2s^2$")
    ax.set_xlabel(r"$s$  ($f=(1+s,\,1-s)$)"); ax.legend(frameon=False); ax.set_title("two-point space", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "two-point.svg"), bbox_inches="tight")
    if os.environ.get("PREVIEW"): fig.savefig(os.environ["PREVIEW"] + "-two.png", dpi=70, bbox_inches="tight")
    print("wrote figures/hypercube-mixing.svg, figures/two-point.svg")
    mismatch_figure()

if __name__ == "__main__":
    main()
