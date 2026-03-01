from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class SynthesisConfig:
    d: int = 5
    max_delta_norm: float = 0.45
    label_mismatch_prob: float = 0.02
    hard_window_mismatch_boost: float = 0.10
    seed: int = 0


@dataclass
class SynthesizedSequence:
    z: Array
    y: Array
    target_leaders: Array
    realized_leaders: Array
    w_star_path: Array


def _normalize(v: Array, eps: float = 1e-12) -> Array:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def action_ftl(theta: Array) -> Array:
    return _normalize(-theta)


def _orthonormal_basis(rng: np.random.Generator, d: int) -> Array:
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return q.astype(float)


def _smooth_rotation(u_from: Array, u_to: Array, alpha: float) -> Array:
    # Interpolate and renormalize to obtain a smooth path over the unit sphere.
    return _normalize((1.0 - alpha) * u_from + alpha * u_to)


def _target_direction_path(name: str, T: int, d: int, rng: np.random.Generator) -> tuple[Array, Array]:
    v1 = _normalize(rng.standard_normal(d))
    v2 = _normalize(rng.standard_normal(d))
    v3 = _normalize(rng.standard_normal(d))

    target = np.zeros((T, d), dtype=float)
    w_star = np.zeros((T, d), dtype=float)

    if name == "stable_benign":
        for t in range(T):
            target[t] = _normalize(v1 + 0.01 * rng.standard_normal(d))
            w_star[t] = _normalize(v1 + 0.03 * rng.standard_normal(d))
    elif name == "persistent_shift":
        shift_center = int(0.62 * T)
        for t in range(T):
            if t < shift_center:
                target[t] = _normalize(v1 + 0.012 * rng.standard_normal(d))
                w_star[t] = _normalize(v1 + 0.03 * rng.standard_normal(d))
            else:
                target[t] = _normalize(v2 + 0.014 * rng.standard_normal(d))
                w_star[t] = _normalize(v2 + 0.035 * rng.standard_normal(d))
    elif name == "delayed_hardening":
        hard_start = int(0.55 * T)
        shift_center = int(0.78 * T)
        for t in range(T):
            if t < hard_start:
                target[t] = _normalize(v1 + 0.010 * rng.standard_normal(d))
                w_star[t] = _normalize(v1 + 0.028 * rng.standard_normal(d))
            elif t < shift_center:
                alpha = (t - hard_start) / max(shift_center - hard_start - 1, 1)
                drift_target = _smooth_rotation(v1, v2, 0.40 * alpha)
                drift_star = _smooth_rotation(v1, v2, 0.45 * alpha)
                target[t] = _normalize(drift_target + 0.012 * rng.standard_normal(d))
                w_star[t] = _normalize(drift_star + 0.032 * rng.standard_normal(d))
            else:
                jump_target = _smooth_rotation(v1, v3, 0.70)
                jump_star = _smooth_rotation(v1, v3, 0.78)
                target[t] = _normalize(jump_target + 0.015 * rng.standard_normal(d))
                w_star[t] = _normalize(jump_star + 0.04 * rng.standard_normal(d))
    else:
        raise ValueError(f"Unknown regime: {name}")

    return target, w_star


def _magnitude_schedule(name: str, T: int, rng: np.random.Generator) -> Array:
    # Radius of theta_t. Large radius stabilizes leader direction; low radius increases volatility.
    r = np.zeros(T, dtype=float)

    if name == "stable_benign":
        base = 0.10
        slope = 0.020
        for t in range(T):
            r[t] = base + slope * np.sqrt(t + 1) + 0.003 * rng.standard_normal()
    elif name == "persistent_shift":
        base = 0.10
        slope = 0.022
        for t in range(T):
            r[t] = base + slope * np.sqrt(t + 1) + 0.003 * rng.standard_normal()
        lo = int(0.62 * T)
        r[lo:] *= 0.72
    elif name == "delayed_hardening":
        base = 0.11
        slope = 0.019
        for t in range(T):
            r[t] = base + slope * np.sqrt(t + 1) + 0.003 * rng.standard_normal()
        lo = int(0.55 * T)
        hi = int(0.78 * T)
        r[lo:hi] *= 0.85
        r[hi:] *= 0.74
    else:
        raise ValueError(f"Unknown regime: {name}")

    return np.clip(r, 0.02, 1.60)


def _realizable_theta_path(target: Array, radii: Array, max_delta_norm: float) -> tuple[Array, Array]:
    T, d = target.shape
    theta = np.zeros((T, d), dtype=float)
    deltas = np.zeros((T, d), dtype=float)

    prev = np.zeros(d, dtype=float)
    for t in range(T):
        desired = -radii[t] * target[t]
        delta = desired - prev
        norm = float(np.linalg.norm(delta))
        if norm > max_delta_norm:
            delta = (max_delta_norm / norm) * delta
        curr = prev + delta
        theta[t] = curr
        deltas[t] = delta
        prev = curr

    return theta, deltas


def _delta_to_example(
    delta: Array,
    w_star: Array,
    rng: np.random.Generator,
    mismatch_prob: float,
    in_corruption_window: bool,
    hard_window_boost: float,
) -> tuple[Array, float]:
    # The update is g_t = 0.5 * sign(q_t - y_t) * z_t.
    # Choosing z_t = ±2 delta and y_t in {±1} with the matching sign realizes g_t = delta.
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm < 1e-12:
        z = np.zeros_like(delta)
        y = 1.0
        return z, y

    raw = 2.0 * delta
    z_plus = raw
    z_minus = -raw

    # Choose label orientation using latent separator to keep examples plausible.
    score_plus = float(np.dot(w_star, z_plus))
    score_minus = float(np.dot(w_star, z_minus))
    y_base_plus = 1.0 if score_plus >= 0.0 else -1.0
    y_base_minus = 1.0 if score_minus >= 0.0 else -1.0

    # Option A: y=-1, z=+2delta. Option B: y=+1, z=-2delta.
    # Prefer the option whose implied label better matches latent base label.
    cost_a = 0.0 if y_base_plus == -1.0 else 1.0
    cost_b = 0.0 if y_base_minus == 1.0 else 1.0
    if cost_a <= cost_b:
        z = z_plus
        y = -1.0
    else:
        z = z_minus
        y = 1.0

    # Hard-window mismatch increase to model difficult post-change data.
    p = mismatch_prob
    if in_corruption_window:
        p = max(p, hard_window_boost)
    if rng.random() < p:
        y = -y

    return z, y


def _is_corruption_window(name: str, t: int, T: int) -> bool:
    if name != "persistent_shift":
        return False
    return int(0.62 * T) <= t


def synthesize_sequence(
    T: int,
    regime: str,
    cfg: SynthesisConfig,
) -> SynthesizedSequence:
    rng = np.random.default_rng(cfg.seed)

    target, w_star_path = _target_direction_path(regime, T, cfg.d, rng)
    radii = _magnitude_schedule(regime, T, rng)
    theta_path, deltas = _realizable_theta_path(target, radii, cfg.max_delta_norm)

    z = np.zeros((T, cfg.d), dtype=float)
    y = np.zeros(T, dtype=float)
    realized = np.zeros((T, cfg.d), dtype=float)

    for t in range(T):
        z_t, y_t = _delta_to_example(
            deltas[t],
            w_star_path[t],
            rng,
            mismatch_prob=cfg.label_mismatch_prob,
            in_corruption_window=_is_corruption_window(regime, t, T),
            hard_window_boost=cfg.hard_window_mismatch_boost,
        )
        z[t] = z_t
        y[t] = y_t
        realized[t] = action_ftl(theta_path[t])

    return SynthesizedSequence(
        z=z,
        y=y,
        target_leaders=target,
        realized_leaders=realized,
        w_star_path=w_star_path,
    )


def regime_names() -> list[str]:
    return [
        "stable_benign",
        "persistent_shift",
        "delayed_hardening",
    ]
