from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class Sequence:
    z: Array
    y: Array
    name: str
    description: str


GeneratorFn = Callable[[int, int, float, np.random.Generator], Sequence]


def _unit(v: Array, eps: float = 1e-12) -> Array:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _basis(d: int, idx: int = 0) -> Array:
    e = np.zeros(d, dtype=np.float64)
    e[min(max(idx, 0), d - 1)] = 1.0
    return e


def _orthogonal_noise(rng: np.random.Generator, u: Array, scale: float) -> Array:
    v = rng.normal(size=u.shape[0])
    v = v - float(np.dot(v, u)) * u
    return scale * v


def _sample_margin_sequence(
    T: int,
    d: int,
    rho: float,
    rng: np.random.Generator,
    *,
    separator: Array | None = None,
    label_noise: float = 0.0,
    margin: float = 0.55,
    noise_scale: float = 0.85,
) -> tuple[Array, Array, Array]:
    u = _unit(rng.normal(size=d)) if separator is None else _unit(separator)
    z = np.zeros((T, d), dtype=np.float64)
    y = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=T)

    for t in range(T):
        signed_feature = margin * u + _orthogonal_noise(rng, u, noise_scale)
        z[t] = rho * y[t] * _unit(signed_feature)

    if label_noise > 0.0:
        flips = rng.random(T) < label_noise
        y[flips] *= -1.0

    return z, y.astype(np.float64), u


def iid_separable_margin(T: int, d: int, rho: float, rng: np.random.Generator) -> Sequence:
    z, y, _ = _sample_margin_sequence(T, d, rho, rng, label_noise=0.0)
    return Sequence(
        z=z,
        y=y,
        name="iid_separable_margin",
        description="i.i.d. bounded-margin separable stream",
    )


def massart_10(T: int, d: int, rho: float, rng: np.random.Generator) -> Sequence:
    z, y, _ = _sample_margin_sequence(T, d, rho, rng, label_noise=0.10)
    return Sequence(
        z=z,
        y=y,
        name="massart_10",
        description="same margin model with 10% bounded label flips",
    )


def alternating_antileader(T: int, d: int, rho: float, rng: np.random.Generator) -> Sequence:
    del rng
    z = np.tile(rho * _basis(d, 0), (T, 1))
    y = np.empty(T, dtype=np.float64)
    y[0::2] = 1.0
    y[1::2] = -1.0
    return Sequence(
        z=z,
        y=y,
        name="alternating_antileader",
        description="fixed feature direction with alternating labels",
    )


def switching_leaders(T: int, d: int, rho: float, rng: np.random.Generator) -> Sequence:
    del rng  # deterministic
    z = np.tile(rho * _basis(d, 0), (T, 1))
    y = np.empty(T, dtype=np.float64)
    sign = 1.0
    idx = 0
    while idx < T:
        run = min(20, T - idx)
        y[idx : idx + run] = sign
        idx += run
        sign = -sign
    return Sequence(
        z=z,
        y=y,
        name="switching_leaders",
        description="fixed feature with switching label blocks (exp02 negative-regret sequence)",
    )


def benign_to_hard_suffix(T: int, d: int, rho: float, rng: np.random.Generator) -> Sequence:
    split = int(round(0.45 * T))
    u = _unit(rng.normal(size=d))
    z = np.zeros((T, d), dtype=np.float64)
    y = np.zeros(T, dtype=np.float64)
    moment = np.zeros(d, dtype=np.float64)

    z_prefix, y_prefix, _ = _sample_margin_sequence(
        split,
        d,
        rho,
        rng,
        separator=u,
        label_noise=0.0,
        margin=0.65,
        noise_scale=0.60,
    )
    z[:split] = z_prefix
    y[:split] = y_prefix
    for t in range(split):
        moment += y[t] * z[t]

    # The suffix is an anti-leader phase: each example points along the
    # current FTL direction and receives the opposite label. This is
    # intentionally illustrative and creates a clear, sustained deterioration.
    fallback = u
    for t in range(split, T):
        leader = _unit(moment)
        if float(np.linalg.norm(leader)) < 1e-12:
            leader = fallback
        z[t] = rho * leader
        y[t] = -1.0
        moment += y[t] * z[t]
        fallback = leader

    return Sequence(
        z=z,
        y=y,
        name="benign_to_hard_suffix",
        description="separable prefix followed by adaptive anti-leader suffix",
    )


def separator_drift(T: int, d: int, rho: float, rng: np.random.Generator) -> Sequence:
    u0 = _unit(rng.normal(size=d))
    raw = rng.normal(size=d)
    u1 = _unit(raw - float(np.dot(raw, u0)) * u0)
    if np.linalg.norm(u1) < 1e-12:
        u1 = _basis(d, 1)

    z = np.zeros((T, d), dtype=np.float64)
    y = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=T)
    start = int(0.30 * T)
    end = int(0.82 * T)

    for t in range(T):
        alpha = 0.0 if t < start else (t - start) / max(end - start, 1)
        alpha = min(1.0, max(0.0, alpha))
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        u_t = _unit((1.0 - smooth) * u0 + smooth * u1)
        signed_feature = 0.55 * u_t + _orthogonal_noise(rng, u_t, 0.75)
        z[t] = rho * y[t] * _unit(signed_feature)

    return Sequence(
        z=z,
        y=y,
        name="separator_drift",
        description="gradual rotation of the latent separator",
    )


def random_labels_isotropic(T: int, d: int, rho: float, rng: np.random.Generator) -> Sequence:
    z = rng.normal(size=(T, d))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = rho * z / np.maximum(norms, 1e-12)
    y = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=T)
    return Sequence(
        z=z,
        y=y,
        name="random_labels_isotropic",
        description="isotropic features with independent random labels",
    )


def available_generators() -> dict[str, GeneratorFn]:
    return {
        "iid_separable_margin": iid_separable_margin,
        "massart_10": massart_10,
        "alternating_antileader": alternating_antileader,
        "switching_leaders": switching_leaders,
        "benign_to_hard_suffix": benign_to_hard_suffix,
        "separator_drift": separator_drift,
        "random_labels_isotropic": random_labels_isotropic,
    }


def primary_scenarios() -> list[str]:
    return [
        "iid_separable_margin",
        "massart_10",
        "alternating_antileader",
        "benign_to_hard_suffix",
        "separator_drift",
    ]


def hard_calibration_scenarios() -> list[str]:
    return [
        "alternating_antileader",
        "random_labels_isotropic",
        "benign_to_hard_suffix",
    ]
