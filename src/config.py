"""Configuration loading and validation for the fault simulator.

Reads params.yaml into a frozen, validated Config object. Every other module
receives parameters from this object; nothing else reads the YAML or hardcodes
a value. Validation runs at load time, so an invalid configuration fails before
a run starts rather than partway through a long sweep.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

BOUNDARY_CONDITIONS = {
    "all_open",
    "top_reflective",
    "top_bottom_reflective",
    "all_closed",
    "periodic",  # validation-only, not part of the main four-case comparison
}
KERNEL_KINDS = {"isotropic", "dipole"}
THRESHOLD_DISTRIBUTIONS = {"uniform", "correlated", "gradient", "checkerboard", "file"}


@dataclass(frozen=True)
class Config:
    # lattice
    L: int
    seed: int
    # primary phase-diagram parameters (the three swept axes)
    beta: float
    epsilon_boundary: float
    epsilon_dissipation: float
    # threshold field
    sigma_th_min: float
    sigma_th_max: float
    threshold_distribution: str
    threshold_file: str | None
    # residual stress
    sigma_res: float
    heterogeneous_residual: bool
    # kernel
    r0: float
    kernel_kind: str
    theta0: float
    # segmentation
    n_segments: int
    # boundary condition
    boundary_condition: str
    # run control
    n_avalanches: int
    burn_in: int
    # output
    output_dir: str
    snapshot_every: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.L < 2:
            raise ValueError(f"L must be >= 2, got {self.L}")
        if self.sigma_th_min <= 0:
            raise ValueError(
                f"sigma_th_min must be > 0 to keep the risk ratio finite, "
                f"got {self.sigma_th_min}"
            )
        if self.sigma_th_max <= self.sigma_th_min:
            raise ValueError(
                f"sigma_th_max ({self.sigma_th_max}) must exceed "
                f"sigma_th_min ({self.sigma_th_min})"
            )
        if not self.heterogeneous_residual and self.sigma_res >= self.sigma_th_min:
            raise ValueError(
                f"uniform sigma_res ({self.sigma_res}) must be below sigma_th_min "
                f"({self.sigma_th_min}) so every site has a positive stress drop"
            )
        if not 0.0 <= self.epsilon_boundary <= 1.0:
            raise ValueError(
                f"epsilon_boundary must be in [0, 1], got {self.epsilon_boundary}"
            )
        if not 0.0 <= self.epsilon_dissipation < 1.0:
            raise ValueError(
                f"epsilon_dissipation must be in [0, 1), got {self.epsilon_dissipation}"
            )
        if self.beta < 0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")
        if self.kernel_kind not in KERNEL_KINDS:
            raise ValueError(
                f"kernel_kind must be one of {sorted(KERNEL_KINDS)}, "
                f"got {self.kernel_kind!r}"
            )
        if self.threshold_distribution not in THRESHOLD_DISTRIBUTIONS:
            raise ValueError(
                f"threshold_distribution must be one of "
                f"{sorted(THRESHOLD_DISTRIBUTIONS)}, got {self.threshold_distribution!r}"
            )
        if self.threshold_distribution == "file":
            if not self.threshold_file:
                raise ValueError(
                    "threshold_file must be set when threshold_distribution is 'file'"
                )
            if not Path(self.threshold_file).exists():
                raise ValueError(
                    f"threshold_file not found: {self.threshold_file!r}"
                )
        else: 
            pass
        if self.boundary_condition not in BOUNDARY_CONDITIONS:
            raise ValueError(
                f"boundary_condition must be one of {sorted(BOUNDARY_CONDITIONS)}, "
                f"got {self.boundary_condition!r}"
            )
        if self.n_segments < 1:
            raise ValueError(f"n_segments must be >= 1, got {self.n_segments}")
        if self.burn_in < 0:
            raise ValueError(f"burn_in must be >= 0, got {self.burn_in}")
        if self.n_avalanches <= self.burn_in:
            raise ValueError(
                f"n_avalanches ({self.n_avalanches}) must exceed burn_in ({self.burn_in})"
            )


def load_config(path: str | Path) -> Config:
    """Read a YAML file and return a validated Config."""
    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f)
    try:
        return Config(**data)
    except TypeError as e:
        raise ValueError(
            f"params file {path} is missing a field or has an unexpected one: {e}"
        ) from e