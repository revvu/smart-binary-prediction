from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class SynthesisConfig:
    d: int = 5
    n_candidates: int = 64
    label_noise_std: float = 0.15
    leader_weight: float = 1.0
    realism_weight: float = 0.03
    difficulty_weight: float = 0.5
    flip_penalty: float = 0.15
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


def _grad_sign(q: float, y: float) -> float:
    diff = q - y
    if diff > 0.0:
        return 0.5
    if diff < 0.0:
        return -0.5
    return 0.0


def _sample_covariate(
    rng: np.random.Generator,
    sqrt_diag: Array,
    rotation: Array,
) -> Array:
    base = rng.standard_normal(sqrt_diag.shape[0]) * sqrt_diag
    z = rotation @ base
    n = np.linalg.norm(z)
    if n > 1.0:
        z = z / n
    return z.astype(float)


def _orthonormal_basis(rng: np.random.Generator, d: int) -> Array:
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return q.astype(float)


def generate_target_regime(name: str, T: int, d: int, seed: int) -> tuple[Array, Array, Array, Array, Array, Array]:
    """
    Returns (target_leaders, w_star_path, sqrt_diag_path, difficulty_path, leader_weight_path, rotation).

    target_leaders: desired FTL-leader path to emulate
    w_star_path: latent separator path used to generate realistic base labels
    sqrt_diag_path: per-round feature scale (covariate shift)
    rotation: fixed orthonormal basis for correlated feature sampling
    """
    rng = np.random.default_rng(seed)
    rot = _orthonormal_basis(rng, d)

    v1 = _normalize(rng.standard_normal(d))
    v2 = _normalize(rng.standard_normal(d))
    v3 = _normalize(rng.standard_normal(d))

    target = np.zeros((T, d), dtype=float)
    difficulty = np.zeros(T, dtype=float)
    leader_weight_path = np.ones(T, dtype=float)

    if name == "stable_benign":
        for t in range(T):
            target[t] = _normalize(v1 + 0.03 * rng.standard_normal(d))
        difficulty[:] = 0.25
    elif name == "gradual_drift":
        for t in range(T):
            alpha = t / max(T - 1, 1)
            vec = (1.0 - alpha) * v1 + alpha * v2 + 0.02 * rng.standard_normal(d)
            target[t] = _normalize(vec)
        difficulty[:] = 0.35
    elif name == "regime_shift":
        center = 0.55 * T
        sharpness = 0.04 * T
        for t in range(T):
            weight = 1.0 / (1.0 + np.exp(-(t - center) / max(sharpness, 1.0)))
            vec = (1.0 - weight) * v1 + weight * v2 + 0.02 * rng.standard_normal(d)
            target[t] = _normalize(vec)
        difficulty[:] = 0.30
        lo = int(0.45 * T)
        hi = int(0.7 * T)
        difficulty[lo:hi] = 0.80
        leader_weight_path[lo:hi] = 0.30
    elif name == "bursty_corruption":
        target[:] = v1
        bursts = [(int(0.20 * T), int(0.45 * T), v2), (int(0.60 * T), int(0.85 * T), v3)]
        for lo, hi, v in bursts:
            target[lo:hi] = v
        target += 0.03 * rng.standard_normal((T, d))
        target = np.array([_normalize(v) for v in target], dtype=float)
        difficulty[:] = 0.35
        for lo, hi, _ in bursts:
            difficulty[lo:hi] = 0.99
            leader_weight_path[lo:hi] = 0.10
    else:
        raise ValueError(f"Unknown regime: {name}")

    # Latent true separator roughly follows target but with independent noise
    w_star_path = np.array([_normalize(u + 0.07 * rng.standard_normal(d)) for u in target], dtype=float)

    # Covariate shift: AR(1) log-scales per dimension
    log_scales = np.zeros((T, d), dtype=float)
    for t in range(1, T):
        log_scales[t] = 0.92 * log_scales[t - 1] + 0.08 * rng.standard_normal(d)
    sqrt_diag_path = np.exp(0.5 * np.clip(log_scales, -0.8, 0.8))

    return target, w_star_path, sqrt_diag_path, difficulty, leader_weight_path, rot


def synthesize_sequence(
    T: int,
    regime: str,
    cfg: SynthesisConfig,
) -> SynthesizedSequence:
    rng = np.random.default_rng(cfg.seed)

    target, w_star_path, sqrt_diag_path, difficulty_path, leader_weight_path, rotation = generate_target_regime(
        regime, T, cfg.d, cfg.seed + 101
    )

    z = np.zeros((T, cfg.d), dtype=float)
    y = np.zeros(T, dtype=float)
    realized = np.zeros((T, cfg.d), dtype=float)

    theta = np.zeros(cfg.d, dtype=float)

    for t in range(T):
        x_t = action_ftl(theta)
        w_star_t = w_star_path[t]
        sqrt_diag_t = sqrt_diag_path[t]
        diff_t = float(difficulty_path[t])
        leader_w_t = float(leader_weight_path[t])

        best_score = np.inf
        best_z = None
        best_y = None
        best_theta_next = None

        for _ in range(cfg.n_candidates):
            if regime == "bursty_corruption" and diff_t >= 0.9:
                # Corruption burst: concentrated feature direction and unstable labels.
                anchor = np.zeros(cfg.d, dtype=float)
                anchor[0] = 1.0
                z_cand = _normalize(anchor + 0.12 * rng.standard_normal(cfg.d))
                base_margin = float(np.dot(w_star_t, z_cand))
                if rng.random() < 0.25:
                    noisy_margin = base_margin + cfg.label_noise_std * float(rng.standard_normal())
                    y_base = 1.0 if noisy_margin >= 0.0 else -1.0
                else:
                    y_base = 1.0 if (t % 2 == 0) else -1.0
            else:
                z_cand = _sample_covariate(rng, sqrt_diag_t, rotation)
                base_margin = float(np.dot(w_star_t, z_cand))
                noisy_margin = base_margin + cfg.label_noise_std * float(rng.standard_normal())
                y_base = 1.0 if noisy_margin >= 0.0 else -1.0

            for y_cand in (y_base, -y_base):
                q = float(np.dot(x_t, z_cand))
                grad = _grad_sign(q, y_cand)
                theta_next = theta + grad * z_cand
                u_next = action_ftl(theta_next)

                leader_err = float(np.sum((u_next - target[t]) ** 2))
                ftl_loss = 0.5 * abs(q - y_cand)
                # Prefer non-extreme margins and avoid too many forced flips.
                realism_pen = abs(base_margin)
                difficulty_pen = abs(ftl_loss - diff_t)
                flip_pen = cfg.flip_penalty if y_cand != y_base else 0.0

                score = (
                    cfg.leader_weight * leader_w_t * leader_err
                    + cfg.realism_weight * realism_pen
                    + cfg.difficulty_weight * difficulty_pen
                    + flip_pen
                )
                if score < best_score:
                    best_score = score
                    best_z = z_cand
                    best_y = float(y_cand)
                    best_theta_next = theta_next

        assert best_z is not None and best_y is not None and best_theta_next is not None

        z[t] = best_z
        y[t] = best_y
        theta = best_theta_next
        realized[t] = action_ftl(theta)

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
        "gradual_drift",
        "regime_shift",
        "bursty_corruption",
    ]
