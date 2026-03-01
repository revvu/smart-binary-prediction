from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class OCOConfig:
    n: int = 1000
    domain_low: float = -1.0
    domain_high: float = 1.0
    threshold_scale: float = 1.0
    seed: int = 0

    @property
    def domain(self) -> tuple[float, float]:
        return (self.domain_low, self.domain_high)

    @property
    def eta_fixed(self) -> float:
        return 2.0 / np.sqrt(self.n)


def _project(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _argmin_aggregate(A: float, B: float, lo: float, hi: float) -> float:
    if A > 0.0:
        return _project(-B / A, lo, hi)
    if B > 0.0:
        return lo
    if B < 0.0:
        return hi
    return 0.0


def _hindsight_min_value(A: float, B: float, lo: float, hi: float) -> float:
    if A > 0.0:
        x_uncon = -B / A
        if lo <= x_uncon <= hi:
            return -0.5 * (B * B) / A
    val_lo = 0.5 * A * lo * lo + B * lo
    val_hi = 0.5 * A * hi * hi + B * hi
    return float(min(val_lo, val_hi))


def _aggregate_update(A: float, B: float, mu_t: float, q_t: float = 1.0) -> tuple[float, float]:
    # Loss_t(a) = 0.5 q_t a^2 - q_t mu_t a (+ constants ignored)
    return A + q_t, B - q_t * mu_t


def _loss_and_grad(x_t: float, mu_t: float, q_t: float = 1.0) -> tuple[float, float]:
    loss = 0.5 * q_t * (x_t - mu_t) ** 2
    grad = q_t * (x_t - mu_t)
    return float(loss), float(grad)


def _regret_from_actions(actions: Array, mu: Array, cfg: OCOConfig, q: Array | None = None) -> Array:
    n = cfg.n
    lo, hi = cfg.domain
    if q is None:
        q = np.ones(n, dtype=float)

    cum_loss = np.zeros(n + 1, dtype=float)
    cum_best = np.zeros(n + 1, dtype=float)
    A = 0.0
    B = 0.0

    for t in range(1, n + 1):
        loss_t, _ = _loss_and_grad(actions[t], float(mu[t - 1]), float(q[t - 1]))
        cum_loss[t] = cum_loss[t - 1] + loss_t
        A, B = _aggregate_update(A, B, float(mu[t - 1]), float(q[t - 1]))
        cum_best[t] = _hindsight_min_value(A, B, lo, hi)

    return cum_loss - cum_best


def run_ftl(mu: Array, cfg: OCOConfig, q: Array | None = None) -> dict[str, Array]:
    n = cfg.n
    lo, hi = cfg.domain
    if q is None:
        q = np.ones(n, dtype=float)

    A = 0.0
    B = 0.0
    actions = np.zeros(n + 1, dtype=float)

    for t in range(1, n + 1):
        actions[t] = _argmin_aggregate(A, B, lo, hi)
        A, B = _aggregate_update(A, B, float(mu[t - 1]), float(q[t - 1]))

    regret = _regret_from_actions(actions, mu, cfg, q=q)
    return {"actions": actions, "regret": regret}


def run_ogd(mu: Array, cfg: OCOConfig, q: Array | None = None, anytime_lr: bool = False) -> dict[str, Array]:
    n = cfg.n
    lo, hi = cfg.domain
    if q is None:
        q = np.ones(n, dtype=float)

    actions = np.zeros(n + 1, dtype=float)

    for t in range(1, n):
        _, grad_t = _loss_and_grad(actions[t], float(mu[t - 1]), float(q[t - 1]))
        eta_t = (1.0 / np.sqrt(t)) if anytime_lr else cfg.eta_fixed
        actions[t + 1] = _project(actions[t] - eta_t * grad_t, lo, hi)

    regret = _regret_from_actions(actions, mu, cfg, q=q)
    return {"actions": actions, "regret": regret}


def compute_sigma_eq6(mu: Array, cfg: OCOConfig, q: Array | None = None) -> Array:
    n = cfg.n
    lo, hi = cfg.domain
    if q is None:
        q = np.ones(n, dtype=float)

    A = 0.0
    B = 0.0
    a_star = np.zeros(n + 1, dtype=float)
    sigma = np.zeros(n + 1, dtype=float)

    for i in range(1, n + 1):
        A, B = _aggregate_update(A, B, float(mu[i - 1]), float(q[i - 1]))
        a_star[i] = _argmin_aggregate(A, B, lo, hi)
        Li_prev = 0.5 * A * a_star[i - 1] ** 2 + B * a_star[i - 1]
        Li_curr = 0.5 * A * a_star[i] ** 2 + B * a_star[i]
        sigma[i] = sigma[i - 1] + (Li_prev - Li_curr)

    return sigma


def run_smart(mu: Array, cfg: OCOConfig, q: Array | None = None, anytime_lr: bool = False) -> dict[str, Array | float | int]:
    n = cfg.n
    if q is None:
        q = np.ones(n, dtype=float)

    ftl = run_ftl(mu, cfg, q=q)
    sigma = compute_sigma_eq6(mu, cfg, q=q)
    threshold = cfg.threshold_scale * 2.0 * np.sqrt(n)
    lo, hi = cfg.domain

    switch_round = n + 1
    for t in range(1, n + 1):
        if sigma[t] > threshold:
            switch_round = t + 1
            break

    actions = np.zeros(n + 1, dtype=float)
    switched = False
    for t in range(1, n + 1):
        if t < switch_round:
            actions[t] = ftl["actions"][t]
            continue

        if not switched:
            # Reset robust branch at switch point.
            actions[t] = 0.0
            switched = True
            continue

        _, grad_prev = _loss_and_grad(actions[t - 1], float(mu[t - 2]), float(q[t - 2]))
        eta_prev = (1.0 / np.sqrt(t - switch_round + 1)) if anytime_lr else cfg.eta_fixed
        actions[t] = _project(actions[t - 1] - eta_prev * grad_prev, lo, hi)

    regret = _regret_from_actions(actions, mu, cfg, q=q)
    return {
        "actions": actions,
        "regret": regret,
        "sigma_eq6": sigma,
        "threshold": threshold,
        "switch_round": switch_round,
    }
