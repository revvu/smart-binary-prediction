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
    direction_noise_scale: float = 1.0
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


def _smoothstep(x: float) -> float:
    z = min(1.0, max(0.0, x))
    return z * z * (3.0 - 2.0 * z)


def _phase_points(T: int, points: tuple[int, ...]) -> tuple[int, ...]:
    # Use fixed global phase boundaries (for horizon comparability), clamped to [0, T].
    out: list[int] = []
    prev = 0
    for p in points:
        v = min(T, max(prev, int(p)))
        out.append(v)
        prev = v
    return tuple(out)


def _target_direction_path(name: str, T: int, d: int, rng: np.random.Generator, noise_scale: float) -> tuple[Array, Array]:
    # Fix regime anchors to reduce cross-horizon variance while keeping fresh per-run noise.
    anchor_rng = np.random.default_rng(1000 + 97 * d + 13 * sum(ord(c) for c in name))
    v1 = _normalize(anchor_rng.standard_normal(d))
    v2 = _normalize(anchor_rng.standard_normal(d))
    v3 = _normalize(anchor_rng.standard_normal(d))

    target = np.zeros((T, d), dtype=float)
    w_star = np.zeros((T, d), dtype=float)

    if name == "stable_benign":
        for t in range(T):
            target[t] = _normalize(v1 + (0.01 * noise_scale) * rng.standard_normal(d))
            w_star[t] = _normalize(v1 + (0.03 * noise_scale) * rng.standard_normal(d))
    elif name == "persistent_shift":
        shift_start, shift_end = _phase_points(T, (140, 900))
        for t in range(T):
            if t < shift_start:
                target[t] = _normalize(v1 + (0.012 * noise_scale) * rng.standard_normal(d))
                w_star[t] = _normalize(v1 + (0.03 * noise_scale) * rng.standard_normal(d))
            else:
                alpha = (t - shift_start) / max(shift_end - shift_start - 1, 1)
                blend = _smoothstep(alpha)
                drift_target = _smooth_rotation(v1, v2, blend)
                drift_star = _smooth_rotation(v1, v2, blend)
                target[t] = _normalize(drift_target + (0.013 * noise_scale) * rng.standard_normal(d))
                w_star[t] = _normalize(drift_star + (0.033 * noise_scale) * rng.standard_normal(d))
    elif name == "delayed_hardening":
        hard_start, shift_start, shift_end = _phase_points(T, (320, 560, 860))
        for t in range(T):
            if t < hard_start:
                target[t] = _normalize(v1 + (0.010 * noise_scale) * rng.standard_normal(d))
                w_star[t] = _normalize(v1 + (0.028 * noise_scale) * rng.standard_normal(d))
            elif t < shift_start:
                alpha = (t - hard_start) / max(shift_start - hard_start - 1, 1)
                drift_target = _smooth_rotation(v1, v2, 0.40 * alpha)
                drift_star = _smooth_rotation(v1, v2, 0.45 * alpha)
                target[t] = _normalize(drift_target + (0.012 * noise_scale) * rng.standard_normal(d))
                w_star[t] = _normalize(drift_star + (0.032 * noise_scale) * rng.standard_normal(d))
            else:
                alpha = (t - shift_start) / max(shift_end - shift_start - 1, 1)
                blend = _smoothstep(alpha)
                jump_target = _smooth_rotation(v1, v3, 0.70)
                jump_star = _smooth_rotation(v1, v3, 0.78)
                target_base = _normalize((1.0 - blend) * _smooth_rotation(v1, v2, 0.40) + blend * jump_target)
                star_base = _normalize((1.0 - blend) * _smooth_rotation(v1, v2, 0.45) + blend * jump_star)
                target[t] = _normalize(target_base + (0.015 * noise_scale) * rng.standard_normal(d))
                w_star[t] = _normalize(star_base + (0.04 * noise_scale) * rng.standard_normal(d))
    else:
        raise ValueError(f"Unknown regime: {name}")

    return target, w_star


def _magnitude_schedule(name: str, T: int, rng: np.random.Generator, noise_scale: float) -> Array:
    # Radius of theta_t. Large radius stabilizes leader direction; low radius increases volatility.
    r = np.zeros(T, dtype=float)

    if name == "stable_benign":
        base = 0.10
        slope = 0.020
        for t in range(T):
            r[t] = base + slope * np.sqrt(t + 1) + (0.003 * noise_scale) * rng.standard_normal()
    elif name == "persistent_shift":
        base = 0.10
        slope = 0.022
        start, end = _phase_points(T, (220, 920))
        for t in range(T):
            base_r = base + slope * np.sqrt(t + 1) + (0.003 * noise_scale) * rng.standard_normal()
            if t < start:
                scale = 1.0
            else:
                alpha = (t - start) / max(end - start - 1, 1)
                scale = 1.0 - 0.40 * _smoothstep(alpha)
            r[t] = base_r * scale
    elif name == "delayed_hardening":
        base = 0.11
        slope = 0.019
        for t in range(T):
            r[t] = base + slope * np.sqrt(t + 1) + (0.003 * noise_scale) * rng.standard_normal()
        lo, mid, hi, tail = _phase_points(T, (320, 560, 860, 980))
        if mid > lo:
            scale1 = np.linspace(1.0, 0.85, mid - lo, endpoint=False)
            r[lo:mid] *= scale1
        if hi > mid:
            scale2 = np.linspace(0.85, 0.74, hi - mid, endpoint=False)
            r[mid:hi] *= scale2
        if tail > hi:
            scale3 = np.linspace(0.74, 0.70, tail - hi, endpoint=False)
            r[hi:tail] *= scale3
        r[tail:] *= 0.70
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
    force_flip: bool,
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

    if force_flip:
        y = -y

    return z, y


def _mismatch_rate_schedule(name: str, T: int, base_p: float, boost_p: float) -> Array:
    rates = np.full(T, base_p, dtype=float)
    if name == "persistent_shift":
        start, end = _phase_points(T, (240, 920))
        for t in range(start, T):
            alpha = (t - start) / max(end - start - 1, 1)
            rates[t] = base_p + (boost_p - base_p) * _smoothstep(alpha)
    elif name == "delayed_hardening":
        start, mid, end = _phase_points(T, (340, 640, 920))
        for t in range(start, T):
            if t < mid:
                alpha = (t - start) / max(mid - start - 1, 1)
                rates[t] = base_p + 0.6 * (boost_p - base_p) * _smoothstep(alpha)
            else:
                alpha = (t - mid) / max(end - mid - 1, 1)
                rates[t] = base_p + (boost_p - base_p) * (0.6 + 0.4 * _smoothstep(alpha))
    return np.clip(rates, 0.0, 0.49)


def _deterministic_flip_mask(rates: Array, phase: float) -> NDArray[np.bool_]:
    mask = np.zeros(rates.shape[0], dtype=bool)
    acc = float(phase)
    for t, p in enumerate(rates):
        acc += float(p)
        if acc >= 1.0:
            mask[t] = True
            acc -= 1.0
    return mask


def synthesize_sequence(
    T: int,
    regime: str,
    cfg: SynthesisConfig,
) -> SynthesizedSequence:
    rng = np.random.default_rng(cfg.seed)

    # Direction/magnitude noise scaling controls variance of horizon curves.
    noise_scale = max(0.0, float(cfg.direction_noise_scale))
    target, w_star_path = _target_direction_path(regime, T, cfg.d, rng, noise_scale=noise_scale)
    radii = _magnitude_schedule(regime, T, rng, noise_scale=noise_scale)
    theta_path, deltas = _realizable_theta_path(target, radii, cfg.max_delta_norm)
    boost_p = max(cfg.label_mismatch_prob, cfg.hard_window_mismatch_boost)
    mismatch_rates = _mismatch_rate_schedule(regime, T, cfg.label_mismatch_prob, boost_p)
    flip_mask = _deterministic_flip_mask(mismatch_rates, phase=float(rng.random()))

    z = np.zeros((T, cfg.d), dtype=float)
    y = np.zeros(T, dtype=float)
    realized = np.zeros((T, cfg.d), dtype=float)

    for t in range(T):
        z_t, y_t = _delta_to_example(
            deltas[t],
            w_star_path[t],
            force_flip=bool(flip_mask[t]),
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
