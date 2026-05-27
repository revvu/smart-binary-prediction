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


GeneratorFn = Callable[[int, int, np.random.Generator], Sequence]


def _unit(v: Array, eps: float = 1e-12) -> Array:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _basis(d: int, idx: int = 0) -> Array:
    e = np.zeros(d, dtype=np.float64)
    e[min(max(idx, 0), d - 1)] = 1.0
    return e


def _orthogonal_unit(rng: np.random.Generator, u: Array) -> Array:
    if u.shape[0] <= 1:
        return np.zeros_like(u)
    v = rng.normal(size=u.shape[0])
    v = v - float(np.dot(v, u)) * u
    n = float(np.linalg.norm(v))
    if n > 1e-12:
        return v / n

    fallback_idx = 1 if abs(float(u[0])) > 0.9 and u.shape[0] > 1 else 0
    v = _basis(u.shape[0], fallback_idx)
    v = v - float(np.dot(v, u)) * u
    return _unit(v)


def _orthogonal_noise(rng: np.random.Generator, u: Array, scale: float) -> Array:
    v = rng.normal(size=u.shape[0])
    v = v - float(np.dot(v, u)) * u
    return scale * v


def _sample_margin_sequence(
    T: int,
    d: int,
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
        z[t] = y[t] * _unit(signed_feature)

    if label_noise > 0.0:
        flips = rng.random(T) < label_noise
        y[flips] *= -1.0

    return z, y.astype(np.float64), u


def _sample_covariate_margin_sequence(
    T: int,
    d: int,
    rng: np.random.Generator,
    *,
    separator: Array | None = None,
    label_noise: float = 0.0,
    margin: float = 0.70,
    noise_scale: float = 0.55,
) -> tuple[Array, Array, Array]:
    """Exogenous class-conditional margin stream with diverse bounded covariates."""
    u = _unit(rng.normal(size=d)) if separator is None else _unit(separator)
    z = np.zeros((T, d), dtype=np.float64)
    y = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=T)

    for t in range(T):
        signed_feature = margin * u + noise_scale * _orthogonal_unit(rng, u)
        z[t] = y[t] * _unit(signed_feature)

    if label_noise > 0.0:
        flips = rng.random(T) < label_noise
        y[flips] *= -1.0

    return z, y.astype(np.float64), u


def covariate_diverse_stationary(T: int, d: int, rng: np.random.Generator) -> Sequence:
    z, y, _ = _sample_covariate_margin_sequence(
        T,
        d,
        rng,
        label_noise=0.0,
        margin=0.72,
        noise_scale=0.58,
    )
    return Sequence(
        z=z,
        y=y,
        name="covariate_diverse_stationary",
        description="stationary class-conditional margin stream with diverse bounded covariates",
    )


def mild_label_noise(T: int, d: int, rng: np.random.Generator) -> Sequence:
    z, y, _ = _sample_covariate_margin_sequence(
        T,
        d,
        rng,
        label_noise=0.10,
        margin=0.72,
        noise_scale=0.58,
    )
    return Sequence(
        z=z,
        y=y,
        name="mild_label_noise",
        description="covariate-diverse margin stream with 10% independent label flips",
    )


def delayed_signal_emergence(T: int, d: int, rng: np.random.Generator) -> Sequence:
    split = int(round(0.45 * T))
    u = _unit(rng.normal(size=d))
    z = np.zeros((T, d), dtype=np.float64)
    y = np.zeros(T, dtype=np.float64)

    for t in range(split):
        z[t] = _orthogonal_unit(rng, u)
        y[t] = rng.choice(np.array([-1.0, 1.0], dtype=np.float64))

    z_suffix, y_suffix, _ = _sample_covariate_margin_sequence(
        T - split,
        d,
        rng,
        separator=u,
        label_noise=0.0,
        margin=0.72,
        noise_scale=0.58,
    )
    z[split:] = z_suffix
    y[split:] = y_suffix

    return Sequence(
        z=z,
        y=y,
        name="delayed_signal_emergence",
        description="uninformative cold-start prefix followed by stable covariate-diverse signal",
    )


def market_shift_change_point(T: int, d: int, rng: np.random.Generator) -> Sequence:
    split = int(round(0.45 * T))
    u0 = _unit(rng.normal(size=d))
    turn = _orthogonal_unit(rng, u0)
    if float(np.linalg.norm(turn)) <= 1e-12:
        turn = -u0
    u1 = _unit(0.25 * u0 + 0.9682458365518543 * turn)

    z0, y0, _ = _sample_covariate_margin_sequence(
        split,
        d,
        rng,
        separator=u0,
        label_noise=0.02,
        margin=0.74,
        noise_scale=0.50,
    )
    z1, y1, _ = _sample_covariate_margin_sequence(
        T - split,
        d,
        rng,
        separator=u1,
        label_noise=0.08,
        margin=0.70,
        noise_scale=0.62,
    )

    return Sequence(
        z=np.vstack([z0, z1]),
        y=np.concatenate([y0, y1]),
        name="market_shift_change_point",
        description="exogenous separator change point with a rotated customer-response regime",
    )


def strategic_corruption_suffix(T: int, d: int, rng: np.random.Generator) -> Sequence:
    prefix_end = int(round(0.20 * T))
    erosion_end = int(round(0.40 * T))
    u = _unit(rng.normal(size=d))
    z = np.zeros((T, d), dtype=np.float64)
    y = np.zeros(T, dtype=np.float64)

    for t in range(prefix_end):
        z[t] = _unit(0.96 * u + 0.04 * _orthogonal_unit(rng, u))
        y[t] = 1.0

    for t in range(prefix_end, erosion_end):
        z[t] = _unit(0.96 * u + 0.04 * _orthogonal_unit(rng, u))
        y[t] = -1.0

    for t in range(erosion_end, T):
        z[t] = _unit(0.96 * u + 0.04 * _orthogonal_unit(rng, u))
        y[t] = 1.0 if (t - erosion_end) % 2 == 0 else -1.0

    return Sequence(
        z=z,
        y=y,
        name="strategic_corruption_suffix",
        description="reliable margin prefix followed by sustained corrupted feedback",
    )


def olc_fmg_leader_gap(T: int, d: int, rng: np.random.Generator) -> Sequence:
    del rng
    z = np.tile(_basis(d, 0), (T, 1))
    y = np.empty(T, dtype=np.float64)
    alternating_len = int(round(0.40 * T))
    alternating_len -= alternating_len % 2
    y[:alternating_len:2] = 1.0
    y[1:alternating_len:2] = -1.0
    y[alternating_len:] = 1.0
    return Sequence(
        z=z,
        y=y,
        name="olc_fmg_leader_gap",
        description="one-dimensional OLC analogue of alternating-then-stable individual sequences",
    )


def iid_separable_margin(T: int, d: int, rng: np.random.Generator) -> Sequence:
    z, y, _ = _sample_margin_sequence(T, d, rng, label_noise=0.0)
    return Sequence(
        z=z,
        y=y,
        name="iid_separable_margin",
        description="i.i.d. bounded-margin separable stream",
    )


def massart_10(T: int, d: int, rng: np.random.Generator) -> Sequence:
    z, y, _ = _sample_margin_sequence(T, d, rng, label_noise=0.10)
    return Sequence(
        z=z,
        y=y,
        name="massart_10",
        description="same margin model with 10% bounded label flips",
    )


def alternating_antileader(T: int, d: int, rng: np.random.Generator) -> Sequence:
    del rng
    z = np.tile(_basis(d, 0), (T, 1))
    y = np.empty(T, dtype=np.float64)
    y[0::2] = 1.0
    y[1::2] = -1.0
    return Sequence(
        z=z,
        y=y,
        name="alternating_antileader",
        description="fixed feature direction with alternating labels",
    )


def switching_leaders(T: int, d: int, rng: np.random.Generator) -> Sequence:
    del rng  # deterministic
    z = np.tile(_basis(d, 0), (T, 1))
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


def benign_to_hard_suffix(T: int, d: int, rng: np.random.Generator) -> Sequence:
    split = int(round(0.45 * T))
    u = _unit(rng.normal(size=d))
    z = np.zeros((T, d), dtype=np.float64)
    y = np.zeros(T, dtype=np.float64)
    moment = np.zeros(d, dtype=np.float64)

    z_prefix, y_prefix, _ = _sample_margin_sequence(
        split,
        d,
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
        z[t] = leader
        y[t] = -1.0
        moment += y[t] * z[t]
        fallback = leader

    return Sequence(
        z=z,
        y=y,
        name="benign_to_hard_suffix",
        description="separable prefix followed by adaptive anti-leader suffix",
    )


def separator_drift(T: int, d: int, rng: np.random.Generator) -> Sequence:
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
        z[t] = y[t] * _unit(signed_feature)

    return Sequence(
        z=z,
        y=y,
        name="separator_drift",
        description="gradual rotation of the latent separator",
    )


def random_labels_isotropic(T: int, d: int, rng: np.random.Generator) -> Sequence:
    z = rng.normal(size=(T, d))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.maximum(norms, 1e-12)
    y = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=T)
    return Sequence(
        z=z,
        y=y,
        name="random_labels_isotropic",
        description="isotropic features with independent random labels",
    )


def available_generators() -> dict[str, GeneratorFn]:
    return {
        "covariate_diverse_stationary": covariate_diverse_stationary,
        "mild_label_noise": mild_label_noise,
        "delayed_signal_emergence": delayed_signal_emergence,
        "market_shift_change_point": market_shift_change_point,
        "strategic_corruption_suffix": strategic_corruption_suffix,
        "olc_fmg_leader_gap": olc_fmg_leader_gap,
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
        "covariate_diverse_stationary",
        "delayed_signal_emergence",
        "strategic_corruption_suffix",
        "olc_fmg_leader_gap",
    ]


def hard_calibration_scenarios() -> list[str]:
    return [
        "alternating_antileader",
        "strategic_corruption_suffix",
        "olc_fmg_leader_gap",
        "random_labels_isotropic",
    ]
