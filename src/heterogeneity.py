"""Quenched, spatially-correlated heterogeneity fields.

Generates the failure-threshold field sigma_th (spec 5.1):

  1. Build a Gaussian random field with power spectrum P(k) ~ k^-beta_str by
     spectral filtering of white noise. beta_str = 0 is white noise (uncorrelated);
     larger beta_str is smoother (asperity-scale correlations).
  2. Rank-order map that field onto the target threshold distribution,
     truncated to [sigma_th_min, sigma_th_max], fixing the marginal
     independently of the spatial structure.

Quenched: generated once from the run's seed and held fixed. Independent of
lattice state, so it is tested on its own.
"""
from __future__ import annotations

import numpy as np

from .config import Config


def gaussian_random_field(L: int, beta_str: float, rng: np.random.Generator) -> np.ndarray:
    """Real Gaussian field on an L x L grid with power spectrum ~ k^-beta_str.

    Zero mean (DC mode removed); the marginal is fixed later by the rank map,
    so only the spatial structure matters here.
    """
    noise_k = np.fft.fft2(rng.standard_normal((L, L)))

    kx = np.fft.fftfreq(L)
    ky = np.fft.fftfreq(L)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k = np.sqrt(KX ** 2 + KY ** 2)
    k[0, 0] = 1.0  # placeholder; avoids divide-by-zero at the DC mode

    amplitude = k ** (-beta_str / 2.0)
    amplitude[0, 0] = 0.0  # remove the mean (DC mode)

    return np.fft.ifft2(noise_k * amplitude).real


def _rank_order_map(field: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Map field values onto target values preserving rank.

    Smallest field value -> smallest target value, and so on. Decouples the
    spatial structure (field) from the value distribution (target).
    """
    ranks = np.argsort(np.argsort(field, axis=None))
    mapped = np.sort(target, axis=None)[ranks]
    return mapped.reshape(field.shape)


def threshold_field(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Quenched threshold field sigma_th(i, j) per spec 5.1."""

    n = cfg.L * cfg.L
    q = (np.arange(n) + 0.5) / n
    target = cfg.sigma_th_min + q * (cfg.sigma_th_max - cfg.sigma_th_min)

    if cfg.threshold_distribution == "uniform":
        # iid draw, no spatial structure — fastest baseline
        return rng.uniform(cfg.sigma_th_min, cfg.sigma_th_max, (cfg.L, cfg.L))

    if cfg.threshold_distribution == "correlated":
        field = gaussian_random_field(cfg.L, cfg.beta_str, rng)
        return _rank_order_map(field, target)

    if cfg.threshold_distribution == "gradient":
        # linear ramp along dip direction (rows = depth), rank-mapped to marginal
        ramp = np.linspace(0, 1, cfg.L)
        field = np.tile(ramp[:, np.newaxis], (1, cfg.L))
        return _rank_order_map(field, target)

    if cfg.threshold_distribution == "checkerboard":
        # alternating strong/weak sites; direct assignment, no rank-map
        i, j = np.indices((cfg.L, cfg.L))
        pattern = (i + j) % 2
        return np.where(pattern == 0, cfg.sigma_th_min, cfg.sigma_th_max)

    if cfg.threshold_distribution == "file":
        arr = np.load(cfg.threshold_file).astype(np.float64)
        if arr.shape != (cfg.L, cfg.L):
            raise ValueError(
                f"threshold file shape {arr.shape} does not match "
                f"lattice ({cfg.L}, {cfg.L})"
            )
        if arr.min() <= 0:
            raise ValueError("threshold file contains non-positive values")
        return arr
    raise ValueError(f"unknown threshold_distribution {cfg.threshold_distribution!r}")

def residual_field(cfg: Config) -> np.ndarray:
    """Residual-stress field sigma_res(i, j). Uniform by default (spec 5.3)."""
    if cfg.heterogeneous_residual:
        raise NotImplementedError(
            "heterogeneous residual is a deferred sensitivity option; "
            "set heterogeneous_residual: false."
        )
    return np.full((cfg.L, cfg.L), cfg.sigma_res, dtype=np.float64)