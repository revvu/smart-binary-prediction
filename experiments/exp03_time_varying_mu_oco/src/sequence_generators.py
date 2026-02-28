from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

MuGenerator = Callable[[int, np.random.Generator], NDArray[np.float64]]


def mu_constant(value: float = 0.25) -> MuGenerator:
    def _gen(n: int, _rng: np.random.Generator) -> NDArray[np.float64]:
        return np.full(n, float(value), dtype=float)

    return _gen


def mu_step(first: float = 0.75, second: float = 0.25) -> MuGenerator:
    def _gen(n: int, _rng: np.random.Generator) -> NDArray[np.float64]:
        out = np.full(n, float(second), dtype=float)
        out[: n // 2] = float(first)
        return out

    return _gen


def mu_sine(amplitude: float = 0.35, offset: float = 0.0, cycles: float = 4.0) -> MuGenerator:
    def _gen(n: int, _rng: np.random.Generator) -> NDArray[np.float64]:
        t = np.arange(1, n + 1, dtype=float)
        return offset + amplitude * np.sin(2.0 * np.pi * cycles * t / n)

    return _gen


def mu_random_uniform(low: float = -0.25, high: float = 0.25) -> MuGenerator:
    def _gen(n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.uniform(low, high, size=n).astype(float)

    return _gen


def available_generators() -> dict[str, MuGenerator]:
    return {
        "constant_0.25": mu_constant(0.25),
        "step_0.75_to_0.25": mu_step(0.75, 0.25),
        "sine": mu_sine(),
        "uniform_random": mu_random_uniform(),
    }
