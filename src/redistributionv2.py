"""FFT-based stress redistribution.

Connects kernel.py's kernel array to FaultLattice. At startup, the kernel is
embedded into a padded domain sized for the chosen boundary condition and
FFT'd once (stored). Per cascade step, the source field s (from
FaultLattice.fail()) is embedded the same way, FFT'd, multiplied by the
stored kernel transform, inverse-FFT'd, and the physical L x L block is
extracted and returned for FaultLattice.apply_delta().

Paper 1 scope: n_segments == 1, so there is no segment-block decomposition;
each boundary condition is handled entirely by how the source/kernel are
embedded and padded before the FFT.

Padding rule: FFT-based circular convolution requires every axis of the
padded domain to be >= 2L-1, so that every kernel displacement
(-(L-1)..L-1) maps to a distinct index via _wrap_kernel without collision.
Reflective boundary conditions additionally need room for the field's
mirrored copy without it overlapping the original, so their pad shapes are
larger still.

One _embed_* function per boundary condition (spec 7.2):
  - all_open:              zero-pad, linear convolution (no wraparound)
  - top_reflective:        mirror-extend top edge, zero-pad the rest
  - top_bottom_reflective: mirror-extend top and bottom, zero-pad left/right
  - all_closed:            mirror-extend all four edges
  - periodic:              no padding; circular convolution (validation only, 10.3)

A direct O(L^4) summation is included for the FFT-vs-direct validation (10.1)
on small lattices; not for production use.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .kernel import build_kernel


def _wrap_kernel(K: np.ndarray, pad_shape: tuple[int, int]) -> np.ndarray:
    """Place a centered (2L-1)x(2L-1) kernel into wraparound (FFT) order
    inside a padded array of shape pad_shape.

    K is centered: K[c, c] (c = L-1) is zero displacement. Displacement
    (di, dj) for di, dj in [-(L-1), L-1] contributes to index
    (di mod pad_shape[0], dj mod pad_shape[1]) in the output, accumulated
    (+=) rather than overwritten.

    For pad_shape >= 2L-1 in both axes (all_open, reflective, closed cases),
    every displacement maps to a distinct index, so accumulation is
    equivalent to assignment and the full kernel weight is preserved.

    For pad_shape == (L, L) (periodic), multiple displacements that differ
    by a multiple of L are physically the same wrap-around displacement on
    the ring and must add together — this is the periodic kernel, correctly
    summed by accumulation.
    """
    L = (K.shape[0] + 1) // 2
    c = L - 1
    out = np.zeros(pad_shape, dtype=np.float64)
    for di in range(-(L - 1), L):
        for dj in range(-(L - 1), L):
            out[di % pad_shape[0], dj % pad_shape[1]] += K[c + di, c + dj]
    return out


def _embed_all_open(field: np.ndarray, pad_shape: tuple[int, int]) -> np.ndarray:
    """Zero-pad field into the top-left corner of a padded array."""
    L = field.shape[0]
    out = np.zeros(pad_shape, dtype=field.dtype)
    out[:L, :L] = field
    return out


def _embed_top_reflective(field: np.ndarray, pad_shape: tuple[int, int]) -> np.ndarray:
    """Mirror-extend the top edge (row 0) upward; zero-pad elsewhere.

    The mirror image of the field is placed in the last L rows of the
    padded array, which under circular convolution sit immediately
    "above" row 0 (index -1 wraps to pad_shape[0]-1).
    """
    L = field.shape[0]
    out = np.zeros(pad_shape, dtype=field.dtype)
    out[:L, :L] = field
    out[-L:, :L] = field[::-1, :]
    return out


def _embed_top_bottom_reflective(field: np.ndarray, pad_shape: tuple[int, int]) -> np.ndarray:
    """Mirror-extend both vertical edges (top and bottom); zero-pad horizontally.

    The mirror of the top edge goes in the last L rows (wraps to just above
    row 0); the mirror of the bottom edge goes in rows [L, 2L), immediately
    below row L-1.
    """
    L = field.shape[0]
    out = np.zeros(pad_shape, dtype=field.dtype)
    out[:L, :L] = field
    out[-L:, :L] = field[::-1, :]    # mirror of top edge, wraps to above row 0
    out[L:2 * L, :L] = field[::-1, :]  # mirror of bottom edge, just below row L-1
    return out


def _embed_all_closed(field: np.ndarray, pad_shape: tuple[int, int]) -> np.ndarray:
    """Mirror-extend all four edges (top, bottom, left, right) and corners."""
    L = field.shape[0]
    out = np.zeros(pad_shape, dtype=field.dtype)
    out[:L, :L] = field
    v = field[::-1, :]
    h = field[:, ::-1]
    vh = field[::-1, ::-1]
    out[-L:, :L] = v          # mirror above
    out[L:2 * L, :L] = v       # mirror below
    out[:L, -L:] = h           # mirror left
    out[:L, L:2 * L] = h       # mirror right
    out[-L:, -L:] = vh         # corner: above-left
    out[-L:, L:2 * L] = vh     # corner: above-right
    out[L:2 * L, -L:] = vh     # corner: below-left
    out[L:2 * L, L:2 * L] = vh  # corner: below-right
    return out


_EMBED_FUNCS = {
    "all_open": _embed_all_open,
    "top_reflective": _embed_top_reflective,
    "top_bottom_reflective": _embed_top_bottom_reflective,
    "all_closed": _embed_all_closed,
}

# Padded domain size for each non-periodic BC, as a function of L.
# Every axis must be >= 2L-1 for _wrap_kernel. Reflective/closed cases need
# extra room so the mirrored copies don't overlap the original or each other.
_PAD_SHAPE = {
    "all_open": lambda L: (2 * L - 1, 2 * L - 1),
    "top_reflective": lambda L: (3 * L, 2 * L - 1),
    "top_bottom_reflective": lambda L: (4 * L, 2 * L - 1),
    "all_closed": lambda L: (4 * L, 4 * L),
}


class Redistributor:
    """Precomputes the kernel transform and applies FFT-based redistribution."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.L = cfg.L
        self.bc = cfg.boundary_condition

        K = build_kernel(cfg.L, cfg.r0, cfg.epsilon_dissipation, cfg.kernel_kind, cfg.theta0)
        # K is centered: K[L-1, L-1] is zero displacement (di=dj=0).

        if self.bc == "periodic":
            self._pad_shape = (self.L, self.L)
            self._embed = lambda field, shape: field  # no padding needed
            K_padded = _wrap_kernel(K, self._pad_shape)
            self._K_hat = np.fft.fft2(K_padded)
            self._extract = lambda padded: padded
            return

        self._pad_shape = _PAD_SHAPE[self.bc](self.L)
        self._embed = _EMBED_FUNCS[self.bc]
        K_padded = _wrap_kernel(K, self._pad_shape)
        self._K_hat = np.fft.fft2(K_padded)
        self._extract = lambda padded: padded[: self.L, : self.L]

    def redistribute(self, source: np.ndarray) -> np.ndarray:
        """Convolve the source field with the stored kernel transform.

        Returns the L x L stress increment to add via FaultLattice.apply_delta().
        """
        s_padded = self._embed(source, self._pad_shape)
        s_hat = np.fft.fft2(s_padded)
        g_hat = self._K_hat * s_hat
        g = np.fft.ifft2(g_hat).real
        return self._extract(g)


def redistribute_direct(source: np.ndarray, K: np.ndarray, bc: str) -> np.ndarray:
    """Direct O(L^4) summation, for FFT-vs-direct validation (10.1).

    K is the centered (2L-1)x(2L-1) kernel from build_kernel. Only all_open
    (zero outside the domain) is implemented; small-L validation case.
    """
    if bc != "all_open":
        raise NotImplementedError("direct-sum reference is implemented for all_open only")

    L = source.shape[0]
    c = L - 1
    out = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            total = 0.0
            for ii in range(L):
                for jj in range(L):
                    if source[ii, jj] == 0.0:
                        continue
                    di = i - ii
                    dj = j - jj
                    total += K[c + di, c + dj] * source[ii, jj]
            out[i, j] = total
    return out