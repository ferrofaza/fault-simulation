"""Avalanche-size distribution analysis (Gutenberg-Richter).

Characterization, not the Paper 1 headline: confirms the model produces
critical (power-law) avalanche statistics, the foundation the percolation
analysis builds on.

Provides:
  - log_binned_pdf:  logarithmically-binned P(S), the conventional plot
  - ccdf:            binning-free complementary CDF, cleaner for the tail
  - mle_exponent:    quick approximate power-law exponent (Clauset et al.)
  - fit_powerlaw:    rigorous fit via the `powerlaw` package, if installed
                     (KS-based x_min selection + goodness-of-fit)
  - plot_distribution: PDF + CCDF figure for one or more size arrays

Avalanche "size" here is the number of topplings (site-failure events,
counting repeats), as recorded by dynamics.AvalancheRecord.size. The shed
stress (AvalancheRecord.total_shed) is an alternative magnitude proxy; pass
either array to these functions.
"""
from __future__ import annotations

import numpy as np


def log_binned_pdf(sizes: np.ndarray, n_bins: int = 25):
    """Logarithmically-binned probability density of avalanche sizes.

    Returns (centers, density) with empty bins dropped. Log binning is
    essential for power-law data: linear bins leave the tail with one-or-zero
    counts per bin and the slope becomes unreadable.
    """
    s = np.asarray(sizes)
    s = s[s > 0]
    bins = np.logspace(0, np.log10(s.max() + 1), n_bins)
    counts, edges = np.histogram(s, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])      # geometric bin centers
    density = counts / np.diff(edges) / counts.sum()
    nz = counts > 0
    return centers[nz], density[nz]


def ccdf(sizes: np.ndarray):
    """Complementary cumulative distribution P(S' >= S). Binning-free."""
    s = np.sort(np.asarray(sizes)[np.asarray(sizes) > 0])
    c = 1.0 - np.arange(len(s)) / len(s)
    return s, c


def mle_exponent(sizes: np.ndarray, s_min: int = 3):
    """Approximate discrete power-law exponent tau for P(S) ~ S^-tau, S >= s_min.

    Continuous MLE with a continuity correction (Clauset, Shalizi & Newman
    2009, eq. 3.7). Quick and good for exploration; for publication use
    fit_powerlaw() which selects s_min by KS minimization and reports a
    goodness-of-fit. Returns (tau, standard_error, n_tail).
    """
    s = np.asarray(sizes, dtype=float)
    s = s[s >= s_min]
    n = len(s)
    if n < 2:
        raise ValueError(f"too few avalanches with size >= {s_min} (got {n})")
    tau = 1.0 + n / np.sum(np.log(s / (s_min - 0.5)))
    err = (tau - 1.0) / np.sqrt(n)
    return tau, err, n


def fit_powerlaw(sizes: np.ndarray):
    """Rigorous power-law fit via the `powerlaw` package (Clauset et al.).

    Selects x_min by KS minimization and returns the fit object, from which
    you can read fit.alpha (= tau), fit.xmin, and run
    fit.distribution_compare('power_law', 'lognormal') for model selection.

    Requires: pip install powerlaw
    """
    try:
        import powerlaw
    except ImportError as e:
        raise ImportError(
            "fit_powerlaw needs the `powerlaw` package: pip install powerlaw"
        ) from e
    s = np.asarray(sizes)
    s = s[s > 0]
    return powerlaw.Fit(s, discrete=True)


def plot_distribution(size_arrays: dict, n_bins: int = 25, s_min: int = 3, ax=None):
    """Plot log-binned PDF + CCDF for one or more labelled size arrays.

    size_arrays: {label: sizes_array}. Returns (fig, axes). Prints the quick
    MLE exponent for each. Pass a single {'run': sizes} for one dataset.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    else:
        axes = ax
        fig = axes[0].figure

    for label, sizes in size_arrays.items():
        centers, density = log_binned_pdf(sizes, n_bins)
        (line,) = axes[0].loglog(centers, density, "o", ms=5, alpha=0.8, label=label)
        color = line.get_color()

        s, c = ccdf(sizes)
        axes[1].loglog(s, c, color=color, alpha=0.8, label=label)

        tau, err, n = mle_exponent(sizes, s_min)
        print(f"{label}: tau = {tau:.3f} +/- {err:.3f}  (s>={s_min}, n={n}, max={int(np.max(sizes))})")

        xs = np.logspace(np.log10(s_min), np.log10(np.max(sizes)), 50)
        ref = np.argmin(np.abs(centers - s_min))
        A = density[ref] * s_min ** tau
        axes[0].loglog(xs, A * xs ** (-tau), "--", color=color, alpha=0.5,
                       label=f"{label} τ={tau:.2f}")

    axes[0].set_xlabel("avalanche size S (topplings)")
    axes[0].set_ylabel("P(S)  [log-binned]")
    axes[0].set_title("avalanche-size distribution")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("avalanche size S")
    axes[1].set_ylabel("P(S' ≥ S)  [CCDF]")
    axes[1].set_title("complementary cumulative distribution")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig, axes


def collect_sizes(cfg, n_avalanches: int, burn_in: int) -> np.ndarray:
    """Convenience: run a simulation and return the avalanche-size array.

    Imports the simulator lazily so this module can also be used on
    already-saved size arrays without constructing a lattice.
    """
    from src.lattice import FaultLattice
    from src.redistributionv2 import Redistributor
    from src.dynamics import run

    lattice = FaultLattice(cfg)
    redistributor = Redistributor(cfg)
    return np.array([rec.size for rec in run(lattice, redistributor, n_avalanches, burn_in=burn_in)])


if __name__ == "__main__":
    # Example: run one simulation and show its distribution.
    from dataclasses import replace
    from src.config import load_config
    import matplotlib.pyplot as plt

    cfg = load_config("experiments/params.yaml")
    cfg = replace(cfg, L=64, boundary_condition="all_open", seed=0)

    sizes = collect_sizes(cfg, n_avalanches=12000, burn_in=2000)
    plot_distribution({"all_open": sizes})
    plt.savefig("data/figures/gutenberg_richter.png", dpi=110, bbox_inches="tight")
    print("saved data/figures/gutenberg_richter.png")