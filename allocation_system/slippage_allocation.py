"""
slippage_allocation.py
======================
Simulation engine for comparing capital allocation strategies between two
trading algorithms, based on their historical slippage distributions.

Four strategies:
  - Equal weight   : fixed 50/50
  - Inverse variance: weight ∝ 1/σ²  (rewards consistency)
  - Sharpe-based   : weight ∝ μ/σ    (rewards risk-adjusted edge)
  - Softmax        : smooth probabilistic allocation via temperature

Usage:
    python slippage_allocation.py                  # default params, show plots
    python slippage_allocation.py --help           # full option list

Examples:
    python slippage_allocation.py --a-mean 2 --a-std 4 --b-mean 5 --b-std 2.5
    python slippage_allocation.py --periods 120 --trades 200 --temp 0.5
    python slippage_allocation.py --no-plot --output results.csv
"""

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "equal":    "#888780",
    "inv_var":  "#378ADD",
    "sharpe":   "#1D9E75",
    "softmax":  "#D85A30",
    "algo_a":   "#378ADD",
    "algo_b":   "#D85A30",
}

STRATEGY_LABELS = {
    "equal":   "Equal weight",
    "inv_var": "Inv variance",
    "sharpe":  "Sharpe-based",
    "softmax": "Softmax",
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class AlgoParams:
    name: str
    mean: float        # mean slippage in bps (higher = worse)
    std: float         # std dev in bps
    skew: float = 0.0  # distribution skewness


@dataclass
class SimConfig:
    algo_a: AlgoParams
    algo_b: AlgoParams
    n_periods: int = 60
    n_trades: int = 100
    window: int = 10        # rolling periods for parameter estimation
    temperature: float = 1.0  # softmax temperature
    seed: Optional[int] = None


@dataclass
class PeriodResult:
    period: int
    mean_a: float
    mean_b: float
    var_a: float
    var_b: float
    weights: dict  # strategy -> (w_a, w_b)
    edges: dict    # strategy -> period edge saved (bps)


@dataclass
class SimResult:
    config: SimConfig
    periods: list[PeriodResult] = field(default_factory=list)
    samples_a: np.ndarray = field(default_factory=lambda: np.array([]))
    samples_b: np.ndarray = field(default_factory=lambda: np.array([]))

    def cum_edge(self, strategy: str) -> np.ndarray:
        """Cumulative edge saved (bps) for a strategy, starting from 0."""
        edges = [0.0] + [p.edges[strategy] for p in self.periods]
        return np.cumsum(edges)

    def weights_over_time(self, strategy: str) -> tuple[np.ndarray, np.ndarray]:
        """Returns (w_a, w_b) arrays, one value per period."""
        wa = np.array([p.weights[strategy][0] for p in self.periods])
        wb = np.array([p.weights[strategy][1] for p in self.periods])
        return wa, wb

    def summary(self) -> pd.DataFrame:
        rows = []
        for strat in ["equal", "inv_var", "sharpe", "softmax"]:
            cum = self.cum_edge(strat)
            diffs = np.diff(cum)
            mn = diffs.mean()
            sd = diffs.std(ddof=1) if len(diffs) > 1 else np.nan
            sharpe = mn / sd if sd > 0 else np.nan
            rows.append({
                "strategy":        STRATEGY_LABELS[strat],
                "total_edge_bps":  round(cum[-1], 2),
                "avg_period_bps":  round(mn, 3),
                "volatility_bps":  round(sd, 3),
                "edge_sharpe":     round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
                "final_w_a":       round(self.periods[-1].weights[strat][0], 3),
            })
        return pd.DataFrame(rows)


# ── Slippage sampling ─────────────────────────────────────────────────────────

def sample_slippage(params: AlgoParams, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample n slippage observations from a skew-normal distribution.
    Mean and std are in bps. Positive skew = right tail (occasional large slippage).
    """
    if params.skew == 0.0:
        return rng.normal(params.mean, params.std, n)

    # Fit a skew-normal with the requested moments
    alpha = params.skew  # shape parameter (approx skewness direction)
    delta = alpha / np.sqrt(1 + alpha**2)
    mu_z = delta * np.sqrt(2 / np.pi)
    sigma_z = np.sqrt(1 - mu_z**2)

    raw = stats.skewnorm.rvs(a=alpha, size=n, random_state=rng)
    # Standardise to zero mean, unit var, then rescale
    raw = (raw - mu_z) / sigma_z
    return params.mean + params.std * raw


# ── Allocation strategies ─────────────────────────────────────────────────────

def weights_equal() -> tuple[float, float]:
    return 0.5, 0.5


def weights_inv_variance(var_a: float, var_b: float, eps: float = 1e-9) -> tuple[float, float]:
    """    Allocate inversely proportional to variance.
    Lower variance → higher weight.
    """
    inv_a = 1.0 / (var_a + eps)
    inv_b = 1.0 / (var_b + eps)
    total = inv_a + inv_b
    return inv_a / total, inv_b / total


def weights_sharpe(mean_a: float, mean_b: float,
                   var_a: float, var_b: float,
                   eps: float = 1e-9) -> tuple[float, float]:
    """
    Allocate based on negative Sharpe of slippage:
    lower mean + lower variance → higher weight.
    We negate means because lower slippage is better.
    """
    sh_a = -mean_a / (np.sqrt(var_a) + eps)  # negative slippage = edge
    sh_b = -mean_b / (np.sqrt(var_b) + eps)

    # Shift to positive, then normalise
    floor = min(sh_a, sh_b) - eps
    pos_a = sh_a - floor
    pos_b = sh_b - floor
    total = pos_a + pos_b
    return pos_a / total, pos_b / total


def weights_softmax(mean_a: float, mean_b: float,
                    temperature: float = 1.0,
                    eps: float = 1e-9) -> tuple[float, float]:
    """
    Softmax over negative means (lower slippage preferred).
    temperature → 0 : winner-take-all
    temperature → ∞ : equal weight
    """
    T = max(temperature, eps)
    score_a = -mean_a / T
    score_b = -mean_b / T
    # Subtract max for numerical stability
    m = max(score_a, score_b)
    ea = np.exp(score_a - m)
    eb = np.exp(score_b - m)
    total = ea + eb
    return ea / total, eb / total


# ── Simulation engine ─────────────────────────────────────────────────────────

STRATEGIES = ["equal", "inv_var", "sharpe", "softmax"]


def run_simulation(cfg: SimConfig) -> SimResult:
    rng = np.random.default_rng(cfg.seed)
    result = SimResult(config=cfg)

    all_a, all_b = [], []
    rolling: list[dict] = []  # list of {mean_a, mean_b, var_a, var_b}

    for t in range(cfg.n_periods):
        # Sample this period's trades
        samp_a = sample_slippage(cfg.algo_a, cfg.n_trades, rng)
        samp_b = sample_slippage(cfg.algo_b, cfg.n_trades, rng)
        all_a.append(samp_a)
        all_b.append(samp_b)

        mean_a = samp_a.mean()
        mean_b = samp_b.mean()
        var_a  = samp_a.var(ddof=1)
        var_b  = samp_b.var(ddof=1)

        rolling.append({"mean_a": mean_a, "mean_b": mean_b,
                        "var_a": var_a, "var_b": var_b})
        if len(rolling) > cfg.window:
            rolling.pop(0)

        # Rolling estimates (use all available periods up to window)
        r_mean_a = np.mean([r["mean_a"] for r in rolling])
        r_mean_b = np.mean([r["mean_b"] for r in rolling])
        r_var_a  = np.mean([r["var_a"]  for r in rolling])
        r_var_b  = np.mean([r["var_b"]  for r in rolling])

        # Compute weights for each strategy
        # (for the first few periods before window fills, still estimate)
        wts = {
            "equal":   weights_equal(),
            "inv_var": weights_inv_variance(r_var_a, r_var_b),
            "sharpe":  weights_sharpe(r_mean_a, r_mean_b, r_var_a, r_var_b),
            "softmax": weights_softmax(r_mean_a, r_mean_b, cfg.temperature),
        }

        # Edge saved vs doing nothing (0 bps baseline) = negative weighted mean
        edges = {
            s: -(wts[s][0] * mean_a + wts[s][1] * mean_b)
            for s in STRATEGIES
        }

        result.periods.append(PeriodResult(
            period=t + 1,
            mean_a=mean_a, mean_b=mean_b,
            var_a=var_a,   var_b=var_b,
            weights=wts, edges=edges,
        ))

    result.samples_a = np.concatenate(all_a)
    result.samples_b = np.concatenate(all_b)
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(result: SimResult, output_path: Optional[Path] = None) -> None:
    cfg = result.config
    period_idx = np.arange(len(result.periods) + 1)

    fig = plt.figure(figsize=(16, 12), facecolor="#fafafa")
    fig.suptitle(
        f"Slippage Allocation Simulation  ·  "
        f"Algo A: μ={cfg.algo_a.mean} σ={cfg.algo_a.std} sk={cfg.algo_a.skew}  |  "
        f"Algo B: μ={cfg.algo_b.mean} σ={cfg.algo_b.std} sk={cfg.algo_b.skew}  |  "
        f"{cfg.n_periods} periods × {cfg.n_trades} trades",
        fontsize=11, color="#444", y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)
    ax1 = fig.add_subplot(gs[0, 0])  # cumulative edge
    ax2 = fig.add_subplot(gs[0, 1])  # allocation weights
    ax3 = fig.add_subplot(gs[1, 0])  # slippage distributions
    ax4 = fig.add_subplot(gs[1, 1])  # risk vs reward scatter

    _style_ax(ax1, "Cumulative edge saved (bps)", "Period",
              "Cumulative edge (bps)")
    _style_ax(ax2, "Allocation weight — Algo A", "Period", "Weight on Algo A")
    _style_ax(ax3, "Slippage distributions", "Slippage (bps)", "Density")
    _style_ax(ax4, "Risk vs reward", "Volatility of period edge (bps)",
              "Avg period edge (bps)")

    # ── 1. Cumulative edge ────────────────────────────────────────────────────
    styles = {"equal": (1.5, (4, 3)), "inv_var": (2.2, None),
              "sharpe": (2.2, (5, 2)), "softmax": (2.2, (2, 2))}
    for s in STRATEGIES:
        lw, dash = styles[s]
        cum = result.cum_edge(s)
        ax1.plot(period_idx, cum, color=COLORS[s], linewidth=lw,
                 dashes=dash or (None, None) if dash else [],
                 label=STRATEGY_LABELS[s])
    ax1.axhline(0, color="#ccc", linewidth=0.8, zorder=0)
    ax1.legend(fontsize=9, framealpha=0.6)

    # ── 2. Allocation weights ─────────────────────────────────────────────────
    ax2.axhline(0.5, color=COLORS["equal"], linewidth=1.2,
                linestyle=(0, (4, 3)), label="Equal")
    for s in ["inv_var", "sharpe", "softmax"]:
        lw, dash = styles[s]
        wa, _ = result.weights_over_time(s)
        x = np.arange(1, len(wa) + 1)
        ax2.plot(x, wa, color=COLORS[s], linewidth=lw,
                 dashes=dash or [],
                 label=STRATEGY_LABELS[s])
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.legend(fontsize=9, framealpha=0.6)
    ax2.set_title(ax2.get_title() + "\n(50% line = equal weight)", fontsize=9,
                  color="#888", pad=2)

    # ── 3. Slippage distributions ─────────────────────────────────────────────
    for samp, color, label in [
        (result.samples_a, COLORS["algo_a"], cfg.algo_a.name),
        (result.samples_b, COLORS["algo_b"], cfg.algo_b.name),
    ]:
        ax3.hist(samp, bins=60, density=True, alpha=0.35, color=color, edgecolor="none")
        kde_x = np.linspace(samp.min(), samp.max(), 300)
        kde = stats.gaussian_kde(samp)
        ax3.plot(kde_x, kde(kde_x), color=color, linewidth=2, label=label)
        ax3.axvline(samp.mean(), color=color, linewidth=1.2, linestyle="--", alpha=0.7)
    ax3.legend(fontsize=9, framealpha=0.6)

    # ── 4. Risk vs reward scatter ─────────────────────────────────────────────
    for s in STRATEGIES:
        diffs = np.diff(result.cum_edge(s))
        mn, sd = diffs.mean(), diffs.std(ddof=1)
        ax4.scatter(sd, mn, color=COLORS[s], s=120, zorder=5,
                    label=STRATEGY_LABELS[s])
        ax4.annotate(STRATEGY_LABELS[s], (sd, mn),
                     textcoords="offset points", xytext=(6, 4),
                     fontsize=8, color=COLORS[s])
    ax4.axhline(0, color="#ccc", linewidth=0.8, zorder=0)
    ax4.legend(fontsize=9, framealpha=0.6)

    # ── Summary table in figure ───────────────────────────────────────────────
    summary = result.summary()
    winner = summary.loc[summary["total_edge_bps"].idxmax(), "strategy"]
    print("\n" + "─" * 64)
    print(" Simulation Summary")
    print("─" * 64)
    print(summary.to_string(index=False))
    print("─" * 64)
    print(f" Winner: {winner}")
    print("─" * 64 + "\n")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {output_path}")
    else:
        plt.show()
    plt.close(fig)


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor("#ffffff")
    ax.set_title(title, fontsize=11, fontweight="500", color="#333", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color="#888")
    ax.set_ylabel(ylabel, fontsize=9, color="#888")
    ax.tick_params(colors="#aaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e0e0e0")
    ax.grid(True, color="#f0f0f0", linewidth=0.8, zorder=0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Slippage allocation simulation engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Algo A
    p.add_argument("--a-mean", type=float, default=2.0,
                   help="Algo A mean slippage (bps)")
    p.add_argument("--a-std",  type=float, default=4.0,
                   help="Algo A std dev (bps)")
    p.add_argument("--a-skew", type=float, default=0.3,
                   help="Algo A skewness")
    p.add_argument("--a-name", type=str,   default="Algo A")

    # Algo B
    p.add_argument("--b-mean", type=float, default=5.0,
                   help="Algo B mean slippage (bps)")
    p.add_argument("--b-std",  type=float, default=2.5,
                   help="Algo B std dev (bps)")
    p.add_argument("--b-skew", type=float, default=-0.2,
                   help="Algo B skewness")
    p.add_argument("--b-name", type=str,   default="Algo B")

    # Simulation
    p.add_argument("--periods",  type=int,   default=60,
                   help="Number of simulation periods")
    p.add_argument("--trades",   type=int,   default=100,
                   help="Trades sampled per period")
    p.add_argument("--window",   type=int,   default=10,
                   help="Rolling estimation window (periods)")
    p.add_argument("--temp",     type=float, default=1.0,
                   help="Softmax temperature (lower = more aggressive)")
    p.add_argument("--seed",     type=int,   default=None,
                   help="Random seed for reproducibility")

    # Output
    p.add_argument("--no-plot",  action="store_true",
                   help="Skip plot, print summary only")
    p.add_argument("--output",   type=str,   default=None,
                   help="Save plot to this path (e.g. results.png)")
    p.add_argument("--csv",      type=str,   default=None,
                   help="Export per-period results to CSV")

    return p.parse_args()


def export_csv(result: SimResult, path: str) -> None:
    rows = []
    for pr in result.periods:
        row = {
            "period": pr.period,
            "mean_a": round(pr.mean_a, 4),
            "mean_b": round(pr.mean_b, 4),
            "var_a":  round(pr.var_a,  4),
            "var_b":  round(pr.var_b,  4),
        }
        for s in STRATEGIES:
            row[f"w_a_{s}"]   = round(pr.weights[s][0], 4)
            row[f"edge_{s}"]  = round(pr.edges[s], 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Add cumulative edge columns
    for s in STRATEGIES:
        df[f"cum_edge_{s}"] = df[f"edge_{s}"].cumsum().round(4)

    df.to_csv(path, index=False)
    print(f"Per-period data saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    cfg = SimConfig(
        algo_a=AlgoParams(name=args.a_name, mean=args.a_mean,
                          std=args.a_std,  skew=args.a_skew),
        algo_b=AlgoParams(name=args.b_name, mean=args.b_mean,
                          std=args.b_std,  skew=args.b_skew),
        n_periods=args.periods,
        n_trades=args.trades,
        window=args.window,
        temperature=args.temp,
        seed=args.seed,
    )

    print(f"\nRunning simulation: {cfg.n_periods} periods × {cfg.n_trades} trades/period")
    print(f"  {cfg.algo_a.name}: μ={cfg.algo_a.mean} σ={cfg.algo_a.std} sk={cfg.algo_a.skew}")
    print(f"  {cfg.algo_b.name}: μ={cfg.algo_b.mean} σ={cfg.algo_b.std} sk={cfg.algo_b.skew}")
    print(f"  Softmax temperature: {cfg.temperature}  |  Rolling window: {cfg.window}")

    result = run_simulation(cfg)

    if args.csv:
        export_csv(result, args.csv)

    if not args.no_plot:
        out = Path(args.output) if args.output else None
        plot_results(result, output_path=out)
    else:
        print(result.summary().to_string(index=False))


if __name__ == "__main__":
    main()
