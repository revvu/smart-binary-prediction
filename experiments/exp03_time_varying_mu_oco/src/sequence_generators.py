from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

MuGenerator = Callable[[int, np.random.Generator], NDArray[np.float64]]


def mu_stable_benign() -> MuGenerator:
    """
    FTL-dominant baseline: mild AR(1) variation around a stable center.
    Interpretable as a mostly stationary environment with low process noise.
    """

    def _gen(n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        out = np.zeros(n, dtype=float)
        x = 0.60 + 0.01 * float(rng.standard_normal())
        for t in range(n):
            x = 0.98 * x + 0.02 * 0.60 + 0.006 * float(rng.standard_normal())
            out[t] = x
        return np.clip(out, -0.9, 0.9)

    return _gen


def mu_corruption_burst() -> MuGenerator:
    """
    Adversarial-realistic sequence: baseline positive center with two
    transient corruption-style windows that push mu_t to the opposite side.
    """

    def _gen(n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        out = np.full(n, 0.55, dtype=float)
        out += 0.015 * rng.standard_normal(n)

        bursts = [
            (int(0.30 * n), int(0.42 * n), -0.85),
            (int(0.60 * n), int(0.72 * n), -0.80),
        ]
        for lo, hi, val in bursts:
            out[lo:hi] = val + 0.03 * rng.standard_normal(hi - lo)

        return np.clip(out, -0.95, 0.95)

    return _gen


def mu_drift_plus_shift() -> MuGenerator:
    """
    Representative mixed regime: gradual drift followed by one structural shift.
    """

    def _gen(n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        out = np.zeros(n, dtype=float)
        benign_end = int(0.55 * n)
        shift_start = int(0.72 * n)

        # Long benign phase near a stable center.
        out[:benign_end] = 0.55 + 0.012 * rng.standard_normal(benign_end)

        # Gradual drift pre-shift.
        for t in range(benign_end, shift_start):
            alpha = (t - benign_end) / max(shift_start - benign_end - 1, 1)
            out[t] = 0.55 - 0.30 * alpha + 0.01 * float(rng.standard_normal())

        # Structural shift to a sustained negative regime.
        out[shift_start:] = -0.70 + 0.02 * rng.standard_normal(n - shift_start)

        return np.clip(out, -0.95, 0.95)

    return _gen


def available_generators() -> dict[str, MuGenerator]:
    return {
        "stable_benign": mu_stable_benign(),
        "corruption_burst": mu_corruption_burst(),
        "drift_plus_shift": mu_drift_plus_shift(),
    }
