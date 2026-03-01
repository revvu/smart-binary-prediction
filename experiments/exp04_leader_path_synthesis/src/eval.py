from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def _normalize(v: Array, eps: float = 1e-12) -> Array:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _action_ftl(theta: Array) -> Array:
    return _normalize(-theta)


def _action_ftrl(theta: Array, t: int, eta0: float) -> Array:
    out = -(eta0 / math.sqrt(max(t, 1))) * theta
    n = np.linalg.norm(out)
    if n > 1.0:
        out = out / n
    return out


def _loss(q: float, y: float) -> float:
    return 0.5 * abs(q - y)


def _grad(q: float, y: float) -> float:
    d = q - y
    if d > 0.0:
        return 0.5
    if d < 0.0:
        return -0.5
    return 0.0


@dataclass
class CurveResult:
    regret_ftl: Array
    regret_ftrl: Array
    regret_smart: Array
    sigma: Array
    threshold: float
    switch_round: int


def _prefix_comparator_losses(z: Array, y: Array) -> Array:
    T, d = z.shape
    theta = np.zeros(d, dtype=float)
    comp_prefix = np.zeros(T + 1, dtype=float)

    for t in range(1, T + 1):
        x = _action_ftl(theta)
        q = float(np.dot(z[t - 1], x))
        theta = theta + _grad(q, float(y[t - 1])) * z[t - 1]

        s = _action_ftl(theta)
        q_hist = z[:t] @ s
        comp_prefix[t] = np.sum(0.5 * np.abs(q_hist - y[:t]))

    return comp_prefix


def run_curves(
    z: Array,
    y: Array,
    *,
    eta0: float = math.sqrt(2.0),
    threshold_scale: float = 1.0,
) -> CurveResult:
    T, d = z.shape

    comp_prefix = _prefix_comparator_losses(z, y)

    theta_ftl = np.zeros(d, dtype=float)
    theta_ftrl = np.zeros(d, dtype=float)
    theta_smart_ftl = np.zeros(d, dtype=float)
    theta_smart_ftrl = np.zeros(d, dtype=float)

    cum_ftl = np.zeros(T + 1, dtype=float)
    cum_ftrl = np.zeros(T + 1, dtype=float)
    cum_smart = np.zeros(T + 1, dtype=float)

    sigma = np.zeros(T + 1, dtype=float)
    threshold = threshold_scale * math.sqrt(2.0 * T)
    switched = False
    switch_round = T + 1

    # For SMART's internal FTL-regret estimate
    ftl_internal_cum = 0.0

    for t in range(1, T + 1):
        zt = z[t - 1]
        yt = float(y[t - 1])

        # FTL
        x_ftl = _action_ftl(theta_ftl)
        q_ftl = float(np.dot(zt, x_ftl))
        cum_ftl[t] = cum_ftl[t - 1] + _loss(q_ftl, yt)
        theta_ftl = theta_ftl + _grad(q_ftl, yt) * zt

        # FTRL
        x_ftrl = _action_ftrl(theta_ftrl, t, eta0)
        q_ftrl = float(np.dot(zt, x_ftrl))
        cum_ftrl[t] = cum_ftrl[t - 1] + _loss(q_ftrl, yt)
        theta_ftrl = theta_ftrl + _grad(q_ftrl, yt) * zt

        # SMART internal FTL tracker
        x_sftl = _action_ftl(theta_smart_ftl)
        q_sftl = float(np.dot(zt, x_sftl))
        loss_sftl = _loss(q_sftl, yt)
        ftl_internal_cum += loss_sftl
        theta_smart_ftl = theta_smart_ftl + _grad(q_sftl, yt) * zt

        # prefix comparator for SMART switch signal
        s = _action_ftl(theta_smart_ftl)
        q_hist = z[:t] @ s
        s_loss = np.sum(0.5 * np.abs(q_hist - y[:t]))
        sigma[t] = ftl_internal_cum - s_loss

        if not switched and sigma[t] >= threshold:
            switched = True
            switch_round = t + 1

        if switched:
            x_smart = _action_ftrl(theta_smart_ftrl, t, eta0)
            q_smart = float(np.dot(zt, x_smart))
            cum_smart[t] = cum_smart[t - 1] + _loss(q_smart, yt)
            theta_smart_ftrl = theta_smart_ftrl + _grad(q_smart, yt) * zt
        else:
            cum_smart[t] = cum_smart[t - 1] + loss_sftl

    regret_ftl = cum_ftl - comp_prefix
    regret_ftrl = cum_ftrl - comp_prefix
    regret_smart = cum_smart - comp_prefix

    return CurveResult(
        regret_ftl=regret_ftl,
        regret_ftrl=regret_ftrl,
        regret_smart=regret_smart,
        sigma=sigma,
        threshold=threshold,
        switch_round=switch_round,
    )


def final_regrets(z: Array, y: Array, *, threshold_scale: float = 1.0) -> tuple[float, float, float, int]:
    c = run_curves(z, y, threshold_scale=threshold_scale)
    return float(c.regret_ftl[-1]), float(c.regret_ftrl[-1]), float(c.regret_smart[-1]), int(c.switch_round)
