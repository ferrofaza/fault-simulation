"""Fault-plane state: the stress field and its quenched properties.

FaultLattice holds the current state of the seismogenic patch — the live
stress field sigma and the quenched threshold/residual fields — and the
operations that act only on that state: quasistatic loading, the
sigma_fail -> sigma_res failure reset, and applying redistributed stress.

It is deliberately ignorant of the redistribution kernel, boundary conditions,
segmentation, and the cascade loop. Those live in redistribution.py and
dynamics.py. dynamics.py mutates the field only through this class's methods
(load_to_failure, fail, apply_delta); it never writes to sigma directly.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .heterogeneity import residual_field, threshold_field


class FaultLattice:
    """The L x L stress field and its quenched threshold/residual fields."""

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None) -> None:
        self.cfg = cfg
        self.L = cfg.L

        # Reproducible from cfg.seed; a test may inject its own rng.
        if rng is None:
            rng = np.random.default_rng(cfg.seed)
        # Independent streams, so the initial stress does not depend on how many
        # random numbers the threshold field happened to consume.
        rng_th, rng_init = rng.spawn(2)

        self.sigma_th = threshold_field(cfg, rng_th)
        self.sigma_res = residual_field(cfg)

        # Initial stress: every site below its own threshold (risk ratio < 1),
        # so nothing is unstable before the first load. The run still needs a
        # burn-in transient before its statistics are meaningful.
        r0 = rng_init.uniform(0.0, 1.0, (self.L, self.L))
        self.sigma = r0 * self.sigma_th

    def load_to_failure(self) -> float:
        """Quasistatic loading: raise the whole field until one site fails.

        Adds the smallest stress deficit (min over sites of sigma_th - sigma)
        uniformly to every site, bringing the closest site exactly to its
        threshold. Returns that increment — the loading interval, used as the
        inter-event clock. Driven by the absolute deficit, never the risk ratio.
        """
        delta_sigma = float((self.sigma_th - self.sigma).min())
        self.sigma += delta_sigma
        return delta_sigma

    def unstable_sites(self) -> np.ndarray:
        """Boolean mask of sites at or above threshold (sigma >= sigma_th)."""
        return self.sigma >= self.sigma_th

    def fail(self, mask: np.ndarray) -> np.ndarray:
        """Apply the sigma_fail -> sigma_res reset at the masked sites.

        Returns the per-site shed stress S = sigma_fail - sigma_res as a full
        L x L source field (zero away from the failed sites), computed before
        the reset so overshoot acquired during the cascade is included. That
        field is both the source for redistribution and the per-avalanche
        shed-stress data product.
        """
        source = np.zeros_like(self.sigma)
        source[mask] = self.sigma[mask] - self.sigma_res[mask]
        self.sigma[mask] = self.sigma_res[mask]
        return source

    def apply_delta(self, delta: np.ndarray) -> None:
        """Add redistributed stress to the field."""
        self.sigma += delta

    def risk_ratio(self) -> np.ndarray:
        """Risk-ratio field r = sigma / sigma_th (analysis observable, 13.1).

        Never used to select the failing site; loading uses the deficit.
        """
        return self.sigma / self.sigma_th

    def __repr__(self) -> str:
        return (
            f"FaultLattice(L={self.L}, "
            f"mean_sigma={self.sigma.mean():.4f}, "
            f"max_risk={self.risk_ratio().max():.4f})"
        )