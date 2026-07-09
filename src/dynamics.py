"""Cascade dynamics: the slow-loading / fast-rupture alternation (spec 3).

run_avalanche() performs one full load-cascade cycle:
  1. load_to_failure(): raise the field by the minimum deficit, bringing
     exactly one site to threshold (spec 3.1).
  2. Cascade: while any site is at or above threshold, fail those sites
     (sigma_fail -> sigma_res reset, returning the shed-stress source field
     S = sigma_fail - sigma_res), redistribute S through the kernel, and add
     the result back to sigma (spec 3.2). Repeat until stable.

Returns an AvalancheRecord per avalanche. run() yields one record per
avalanche and discards the first `burn_in` avalanches (spec 10.4) before
yielding, so the system reaches its stationary state before any record is
returned to the caller.

This module composes FaultLattice and Redistributor; it never accesses
lattice.sigma directly, only through FaultLattice's methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .lattice import FaultLattice
from .redistribution import Redistributor


@dataclass
class AvalancheRecord:
    size: int                 # total (site, step) failures across the cascade
    n_steps: int               # number of redistribution events (cascade steps)
    total_shed: float          # sum of S = sigma_fail - sigma_res over all failures
    delta_sigma_load: float    # loading increment that triggered this avalanche
    sigma_total: float          # sum(sigma) over the lattice after the cascade settles


def run_avalanche(lattice: FaultLattice, redistributor: Redistributor) -> AvalancheRecord:
    """One load-then-cascade cycle. Mutates lattice in place."""
    delta_sigma_load = lattice.load_to_failure()

    size = 0
    n_steps = 0
    total_shed = 0.0

    while True:
        mask = lattice.unstable_sites()
        if not mask.any():
            break
        source = lattice.fail(mask)
        total_shed += float(source.sum())
        size += int(mask.sum())
        n_steps += 1
        delta = redistributor.redistribute(source)
        lattice.apply_delta(delta)

    return AvalancheRecord(
        size=size,
        n_steps=n_steps,
        total_shed=total_shed,
        delta_sigma_load=delta_sigma_load,
        sigma_total=float(lattice.sigma.sum()),
    )


def run(
    lattice: FaultLattice,
    redistributor: Redistributor,
    n_avalanches: int,
    burn_in: int = 0,
) -> Iterator[AvalancheRecord]:
    """Run n_avalanches load-cascade cycles, yielding one AvalancheRecord each.

    The first `burn_in` avalanches are run (advancing lattice's state) but
    not yielded, per spec 10.4: statistics should only be computed once the
    system has reached its stationary state.

    n_avalanches is the total number of cycles run, including burn-in, so
    the number of yielded records is n_avalanches - burn_in.
    """
    if burn_in >= n_avalanches:
        raise ValueError(
            f"burn_in ({burn_in}) must be < n_avalanches ({n_avalanches})"
        )

    for i in range(n_avalanches):
        record = run_avalanche(lattice, redistributor)
        if i >= burn_in:
            yield record