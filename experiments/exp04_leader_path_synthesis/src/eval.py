from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit

    HAS_NUMBA = True
except Exception:  # pragma: no cover - optional dependency path
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[override]
        def _decorator(fn):
            return fn

        return _decorator


Array = NDArray[np.float64]
_EVAL_MODE = "auto"  # one of {"auto", "python", "numba"}


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


def set_eval_mode(mode: str) -> None:
    global _EVAL_MODE
    if mode not in {"auto", "python", "numba"}:
        raise ValueError("mode must be one of: auto, python, numba")
    _EVAL_MODE = mode


def get_eval_mode() -> str:
    if _EVAL_MODE == "auto":
        return "numba" if HAS_NUMBA else "python"
    if _EVAL_MODE == "numba" and not HAS_NUMBA:
        return "python"
    return _EVAL_MODE


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


@njit(cache=True)
def _normalize_nb(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = 0.0
    for i in range(v.shape[0]):
        n += v[i] * v[i]
    n = math.sqrt(n)
    out = np.zeros_like(v)
    if n < eps:
        return out
    inv = 1.0 / n
    for i in range(v.shape[0]):
        out[i] = v[i] * inv
    return out


@njit(cache=True)
def _dot_nb(a: np.ndarray, b: np.ndarray) -> float:
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i] * b[i]
    return s


@njit(cache=True)
def _loss_nb(q: float, y: float) -> float:
    d = q - y
    if d < 0.0:
        d = -d
    return 0.5 * d


@njit(cache=True)
def _grad_nb(q: float, y: float) -> float:
    d = q - y
    if d > 0.0:
        return 0.5
    if d < 0.0:
        return -0.5
    return 0.0


@njit(cache=True)
def _action_ftrl_nb(theta: np.ndarray, t: int, eta0: float) -> np.ndarray:
    out = np.zeros_like(theta)
    scale = -(eta0 / math.sqrt(max(t, 1)))
    for i in range(theta.shape[0]):
        out[i] = scale * theta[i]
    n = 0.0
    for i in range(out.shape[0]):
        n += out[i] * out[i]
    n = math.sqrt(n)
    if n > 1.0:
        inv = 1.0 / n
        for i in range(out.shape[0]):
            out[i] *= inv
    return out


@njit(cache=True)
def _run_curves_numba(
    z: np.ndarray,
    y: np.ndarray,
    eta0: float,
    threshold_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    T, d = z.shape

    comp_prefix = np.zeros(T + 1, dtype=np.float64)
    theta_tmp = np.zeros(d, dtype=np.float64)
    for t in range(1, T + 1):
        x = _normalize_nb(-theta_tmp)
        q = _dot_nb(z[t - 1], x)
        g = _grad_nb(q, y[t - 1])
        for k in range(d):
            theta_tmp[k] += g * z[t - 1, k]

        s = _normalize_nb(-theta_tmp)
        s_loss = 0.0
        for i in range(t):
            qh = _dot_nb(z[i], s)
            s_loss += _loss_nb(qh, y[i])
        comp_prefix[t] = s_loss

    theta_ftl = np.zeros(d, dtype=np.float64)
    theta_ftrl = np.zeros(d, dtype=np.float64)
    theta_smart_ftl = np.zeros(d, dtype=np.float64)
    theta_smart_ftrl = np.zeros(d, dtype=np.float64)

    cum_ftl = np.zeros(T + 1, dtype=np.float64)
    cum_ftrl = np.zeros(T + 1, dtype=np.float64)
    cum_smart = np.zeros(T + 1, dtype=np.float64)

    sigma = np.zeros(T + 1, dtype=np.float64)
    threshold = threshold_scale * math.sqrt(2.0 * T)
    switched = False
    switch_round = T + 1
    ftl_internal_cum = 0.0

    for t in range(1, T + 1):
        zt = z[t - 1]
        yt = y[t - 1]

        x_ftl = _normalize_nb(-theta_ftl)
        q_ftl = _dot_nb(zt, x_ftl)
        cum_ftl[t] = cum_ftl[t - 1] + _loss_nb(q_ftl, yt)
        g_ftl = _grad_nb(q_ftl, yt)
        for k in range(d):
            theta_ftl[k] += g_ftl * zt[k]

        x_ftrl = _action_ftrl_nb(theta_ftrl, t, eta0)
        q_ftrl = _dot_nb(zt, x_ftrl)
        cum_ftrl[t] = cum_ftrl[t - 1] + _loss_nb(q_ftrl, yt)
        g_ftrl = _grad_nb(q_ftrl, yt)
        for k in range(d):
            theta_ftrl[k] += g_ftrl * zt[k]

        x_sftl = _normalize_nb(-theta_smart_ftl)
        q_sftl = _dot_nb(zt, x_sftl)
        loss_sftl = _loss_nb(q_sftl, yt)
        ftl_internal_cum += loss_sftl
        g_sftl = _grad_nb(q_sftl, yt)
        for k in range(d):
            theta_smart_ftl[k] += g_sftl * zt[k]

        s = _normalize_nb(-theta_smart_ftl)
        s_loss = 0.0
        for i in range(t):
            qh = _dot_nb(z[i], s)
            s_loss += _loss_nb(qh, y[i])
        sigma[t] = ftl_internal_cum - s_loss

        if (not switched) and sigma[t] >= threshold:
            switched = True
            switch_round = t + 1

        if switched:
            x_smart = _action_ftrl_nb(theta_smart_ftrl, t, eta0)
            q_smart = _dot_nb(zt, x_smart)
            cum_smart[t] = cum_smart[t - 1] + _loss_nb(q_smart, yt)
            g_smart = _grad_nb(q_smart, yt)
            for k in range(d):
                theta_smart_ftrl[k] += g_smart * zt[k]
        else:
            cum_smart[t] = cum_smart[t - 1] + loss_sftl

    regret_ftl = cum_ftl - comp_prefix
    regret_ftrl = cum_ftrl - comp_prefix
    regret_smart = cum_smart - comp_prefix
    return regret_ftl, regret_ftrl, regret_smart, sigma, threshold, switch_round


def run_curves(
    z: Array,
    y: Array,
    *,
    eta0: float = math.sqrt(2.0),
    threshold_scale: float = 1.0,
) -> CurveResult:
    mode = get_eval_mode()
    if mode == "numba":
        reg_ftl, reg_ftrl, reg_smart, sigma, threshold, switch_round = _run_curves_numba(
            z.astype(np.float64, copy=False),
            y.astype(np.float64, copy=False),
            eta0,
            threshold_scale,
        )
        return CurveResult(
            regret_ftl=reg_ftl,
            regret_ftrl=reg_ftrl,
            regret_smart=reg_smart,
            sigma=sigma,
            threshold=float(threshold),
            switch_round=int(switch_round),
        )

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
