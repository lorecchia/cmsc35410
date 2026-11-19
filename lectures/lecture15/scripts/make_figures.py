"""Lecture 15 figures. Run from lectures/lecture15:  python3 scripts/make_figures.py  (numpy, matplotlib).
Normalization (as in Lectures 13, 16, 17): generator Delta, i.e. dX = sqrt(2) dW, heat kernel (4 pi t)^{-1/2} e^{-(x-y)^2/(4t)}."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, GREY, GREEN, BG, INK = "#2563eb", "#ea580c", "#8a857b", "#15803d", "#fbfaf7", "#1c1b19"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "svg.fonttype": "path"})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"); os.makedirs(OUT, exist_ok=True)
PREV = os.environ.get("PREVIEW"); rng = np.random.default_rng(0)

def walk_path(h, T=1.0):
    """Continuous-time walk on hZ, jumps +-h each at rate 1/h^2 (generator -> f''), sampled on its jump times."""
    n = rng.poisson(2 * T / h ** 2); times = np.sort(rng.uniform(0, T, n))
    steps = rng.choice([-h, h], size=n)
    return np.concatenate([[0], times, [T]]), np.concatenate([[0], np.cumsum(steps), [np.sum(steps)]])

def fig_scaling():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3)); ax = axes[0]
    for h, c in ((0.3, ORANGE), (0.1, GREEN), (0.02, BLUE)):
        t, x = walk_path(h); ax.step(t, x, where="post", color=c, lw=1.3, label=f"$h={h}$")
    ax.set_xlabel("$t$"); ax.set_ylabel("$X_t$"); ax.legend(frameon=False, fontsize=10)
    ax.set_title(r"walk on $h\mathbb{Z}$, jumps $\pm h$ at rate $h^{-2}$ each", fontsize=12)
    ax = axes[1]; xs = np.linspace(-5, 5, 400)
    h = 0.05; n_paths = 20000
    n = rng.poisson(2 / h ** 2, size=n_paths); X1 = h * (2 * rng.binomial(n, 0.5) - n)
    ax.hist(X1, bins=np.arange(-5 - h / 2, 5 + h, 4 * h), density=True, color=BLUE, alpha=0.45, label=f"walk, $h={h}$, $t=1$")
    ax.plot(xs, np.exp(-xs ** 2 / 4) / np.sqrt(4 * np.pi), color=INK, lw=2, label=r"$p_1(0,x)=\frac{e^{-x^2/4}}{\sqrt{4\pi}}$")
    print(f"scaling: sample variance of X_1 = {X1.var():.3f} (Brownian motion with generator Delta: 2t = 2)")
    ax.set_xlabel("$x$"); ax.legend(frameon=False, fontsize=10); ax.set_title(r"$X_1$ vs. the heat kernel ($\mathrm{Var}=2t$)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "scaling.svg"), bbox_inches="tight")
    if PREV: fig.savefig(PREV + "-scaling.png", dpi=70, bbox_inches="tight")

def killed_walk(x0, h, n_paths, t_max):
    """Discrete-time walk on hZ killed at 0 and 1; each step +-h takes time dt = h^2/2 (variance 2 dt). Returns exit times."""
    pos = np.full(n_paths, int(round(x0 / h))); M = int(round(1 / h)); tau = np.full(n_paths, np.inf); alive = np.ones(n_paths, bool)
    dt = h ** 2 / 2; steps = int(t_max / dt)
    for s in range(1, steps + 1):
        idx = np.where(alive)[0]
        if idx.size == 0: break
        pos[idx] += rng.choice([-1, 1], size=idx.size)
        dead = idx[(pos[idx] <= 0) | (pos[idx] >= M)]; tau[dead] = s * dt; alive[dead] = False
    return tau

def fig_exit():
    h = 1 / 100
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3)); ax = axes[0]
    tt = np.linspace(0, 0.5, 200)
    for x0, c in ((0.5, BLUE), (0.2, ORANGE)):
        tau = killed_walk(x0, h, 20000, 0.6)
        emp = [(tau > t).mean() for t in tt]
        series = sum(4 / (k * np.pi) * np.sin(k * np.pi * x0) * np.exp(-(k * np.pi) ** 2 * tt) for k in range(1, 400, 2))
        ax.plot(tt, emp, color=c, lw=2.2, label=f"simulated, $x={x0}$")
        ax.plot(tt, series, "--", color=INK, lw=1.2)
        ax.plot(tt, 4 / np.pi * np.sin(np.pi * x0) * np.exp(-np.pi ** 2 * tt), ":", color=c, lw=1.4)
        print(f"exit: x={x0}: P(tau>0.2) simulated {(tau > 0.2).mean():.4f}, series {sum(4/(k*np.pi)*np.sin(k*np.pi*x0)*np.exp(-(k*np.pi)**2*0.2) for k in range(1,400,2)):.4f}")
    ax.set_yscale("log"); ax.set_ylim(5e-3, 1.1); ax.set_xlabel("$t$"); ax.set_ylabel(r"$\mathbb{P}_x(\tau>t)$")
    ax.set_title(r"survival: dashed = series, dotted = $\frac{4}{\pi}\sin(\pi x)\,e^{-\pi^2t}$", fontsize=12); ax.legend(frameon=False, fontsize=10)
    ax = axes[1]; xs = np.linspace(0.05, 0.95, 10); m = []
    for x0 in xs:
        tau = killed_walk(x0, h, 4000, 3.0); m.append(tau.mean())
    xx = np.linspace(0, 1, 200)
    ax.plot(xx, xx * (1 - xx) / 2, color=INK, lw=2, label=r"$\frac{x(1-x)}{2}=\int_0^1G(x,y)\,dy$")
    ax.plot(xs, m, "o", color=BLUE, ms=6, label="simulated $\\mathbb{E}_x\\tau$")
    print("exit: E_x tau simulated vs x(1-x)/2:", [(round(a, 2), round(b, 4), round(a * (1 - a) / 2, 4)) for a, b in zip(xs[::3], m[::3])])
    ax.set_xlabel("$x$"); ax.legend(frameon=False, fontsize=10); ax.set_title(r"mean exit time from $[0,1]$", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "exit.svg"), bbox_inches="tight")
    if PREV: fig.savefig(PREV + "-exit.png", dpi=70, bbox_inches="tight")

if __name__ == "__main__":
    fig_scaling(); fig_exit(); print("wrote figures/scaling.svg, figures/exit.svg")
