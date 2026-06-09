"""Stress-redistribution kernel.

Pure geometry: builds the elastic transfer kernel K on a lattice grid and
normalizes it. The FFT machinery that convolves this kernel with the source
field lives in redistribution.py, so the kernel can be tested on its own.

Isotropic kernel (regularized 1/r^3, the Okada asymptotic envelope):

    K(r) = A / (r^2 + r0^2)^(3/2),   K(0) = 0

A is fixed so that sum_{r != 0} K(r) = 1 - epsilon_dissipation, making
epsilon_dissipation the fraction of shed stress lost per redistribution,
beyond whatever later leaves through the domain boundary.
"""
from __future__ import annotations

import numpy as np


def _displacement_grids(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Integer displacement grids (di, dj) on a (2L-1)x(2L-1) centered array.

    Spans every displacement a source can deliver within an L x L domain,
    from -(L-1) to +(L-1) on each axis. The center element is zero
    displacement (the self site).
    """
    d = np.arange(-(L - 1), L)
    di, dj = np.meshgrid(d, d, indexing="ij")
    return di, dj


def isotropic_kernel(L: int, r0: float, epsilon_dissipation: float) -> np.ndarray:
    """Normalized regularized 1/r^3 kernel, centered, with K(0) = 0."""
    di, dj = _displacement_grids(L)
    r2 = di.astype(np.float64) ** 2 + dj.astype(np.float64) ** 2
    K = 1.0 / (r2 + r0 ** 2) ** 1.5
    center = (L - 1, L - 1)
    K[center] = 0.0  # self-transfer handled by the sigma_fail -> sigma_res reset
    K *= (1.0 - epsilon_dissipation) / K.sum()
    return K


def build_kernel(
    L: int,
    r0: float,
    epsilon_dissipation: float,
    kind: str = "isotropic",
    theta0: float = 0.0,
) -> np.ndarray:
    """Return the normalized kernel for the requested kind.

    Centered layout: element [L-1, L-1] is zero displacement. redistribution.py
    arranges this into FFT (wraparound) order and pads it.
    """
    if kind == "isotropic":
        return isotropic_kernel(L, r0, epsilon_dissipation)
    if kind == "dipole":
        raise NotImplementedError(
            "dipole kernel is a Paper 2 robustness option; its angular lobes "
            "sum to ~0, so it needs a different normalization, added later."
        )
    raise ValueError(f"unknown kernel kind {kind!r}")