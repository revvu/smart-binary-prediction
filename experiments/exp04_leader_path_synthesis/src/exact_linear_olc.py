from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class ExactLinearCurves:
    regret_ftl: Array
    regret_ftrl: Array
    regret_smart: Array
    cum_ftl: Array
    cum_ftrl: Array
    cum_smart: Array
    comp_prefix: Array
    sigma: Array
    threshold: float
    switch_round: int
    switched: bool


def _as_float_array(values: NDArray[np.floating] | NDArray[np.integer]) -> Array:
    return np.ascontiguousarray(values, dtype=np.float64)


def _validate_inputs(z: Array, y: Array, *, norm_tol: float) -> None:
    if z.ndim != 2:
        raise ValueError("z must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if z.shape[0] != y.shape[0]:
        raise ValueError("z and y must have matching first dimensions")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(y)):
        raise ValueError("z and y must contain only finite values")
    max_norm = float(np.linalg.norm(z, axis=1).max(initial=0.0))
    if max_norm > 1.0 + norm_tol:
        raise ValueError(f"linear closed form requires ||z_t|| <= 1; max norm is {max_norm:.6g}")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("y must contain only -1 and +1 labels")


def _unit(v: Array, eps: float = 1e-12) -> Array:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def ftl_action(moment: Array) -> Array:
    """Exact FTL action for losses 0.5 * (1 - y_t <z_t, x>)."""
    return _unit(moment)


def ftrl_action(moment: Array, round_index: int, *, eta0: float = math.sqrt(2.0)) -> Array:
    """
    Quadratic FTRL action for linear losses c_t = -0.5 y_t z_t.

    The update is equivalent to projecting
    (eta0 / (2 * sqrt(round_index))) * sum_{i<t} y_i z_i
    onto the unit Euclidean ball.
    """
    scale = eta0 / (2.0 * math.sqrt(max(1, int(round_index))))
    x = scale * moment
    n = float(np.linalg.norm(x))
    if n > 1.0:
        return x / n
    return x


def linear_surrogate_loss(z_t: Array, y_t: float, x_t: Array) -> float:
    """Loss value under the globally linear OLC surrogate."""
    return 0.5 * (1.0 - float(y_t) * float(np.dot(z_t, x_t)))


def comparator_loss_from_moment(t: int, moment: Array) -> float:
    """Best fixed-action loss on a prefix of length t."""
    return 0.5 * float(t) - 0.5 * float(np.linalg.norm(moment))


def run_curves_exact_linear(
    z: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    eta0: float = math.sqrt(2.0),
    threshold: float | None = None,
    threshold_scale: float = 1.0,
    switch_on_equal: bool = True,
    norm_tol: float = 1e-6,
) -> ExactLinearCurves:
    """
    Evaluate true FTL, quadratic FTRL, and causal single-switch SMART.

    This evaluator is exact for the repository's OLC surrogate when
    ||z_t|| <= 1 and y_t in {-1, +1}, because the absolute loss is then
    globally linear on the unit action ball.
    """
    z_arr = _as_float_array(z)
    y_arr = _as_float_array(y)
    _validate_inputs(z_arr, y_arr, norm_tol=norm_tol)

    T, d = z_arr.shape
    threshold_value = float(threshold) if threshold is not None else float(threshold_scale * math.sqrt(2.0 * T))

    moment_ftl = np.zeros(d, dtype=np.float64)
    moment_ftrl = np.zeros(d, dtype=np.float64)
    moment_smart_ftrl = np.zeros(d, dtype=np.float64)
    suffix_rounds_seen = 0

    cum_ftl = np.zeros(T + 1, dtype=np.float64)
    cum_ftrl = np.zeros(T + 1, dtype=np.float64)
    cum_smart = np.zeros(T + 1, dtype=np.float64)
    comp_prefix = np.zeros(T + 1, dtype=np.float64)
    sigma = np.zeros(T + 1, dtype=np.float64)

    switched = False
    switch_round = T + 1

    for t in range(1, T + 1):
        z_t = z_arr[t - 1]
        y_t = float(y_arr[t - 1])

        x_ftl = ftl_action(moment_ftl)
        x_ftrl = ftrl_action(moment_ftrl, t, eta0=eta0)

        if switched:
            x_smart = ftrl_action(moment_smart_ftrl, suffix_rounds_seen + 1, eta0=eta0)
        else:
            x_smart = x_ftl

        cum_ftl[t] = cum_ftl[t - 1] + linear_surrogate_loss(z_t, y_t, x_ftl)
        cum_ftrl[t] = cum_ftrl[t - 1] + linear_surrogate_loss(z_t, y_t, x_ftrl)
        cum_smart[t] = cum_smart[t - 1] + linear_surrogate_loss(z_t, y_t, x_smart)

        moment_update = y_t * z_t
        moment_ftl += moment_update
        moment_ftrl += moment_update
        if switched:
            moment_smart_ftrl += moment_update
            suffix_rounds_seen += 1

        comp_prefix[t] = comparator_loss_from_moment(t, moment_ftl)
        sigma[t] = cum_ftl[t] - comp_prefix[t]

        crosses = sigma[t] >= threshold_value if switch_on_equal else sigma[t] > threshold_value
        if (not switched) and crosses:
            switched = True
            switch_round = t + 1
            moment_smart_ftrl.fill(0.0)
            suffix_rounds_seen = 0

    regret_ftl = cum_ftl - comp_prefix
    regret_ftrl = cum_ftrl - comp_prefix
    regret_smart = cum_smart - comp_prefix

    return ExactLinearCurves(
        regret_ftl=regret_ftl,
        regret_ftrl=regret_ftrl,
        regret_smart=regret_smart,
        cum_ftl=cum_ftl,
        cum_ftrl=cum_ftrl,
        cum_smart=cum_smart,
        comp_prefix=comp_prefix,
        sigma=sigma,
        threshold=threshold_value,
        switch_round=switch_round,
        switched=switched,
    )


def final_regrets_exact_linear(
    z: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    eta0: float = math.sqrt(2.0),
    threshold: float | None = None,
    threshold_scale: float = 1.0,
) -> tuple[float, float, float, int]:
    curves = run_curves_exact_linear(
        z,
        y,
        eta0=eta0,
        threshold=threshold,
        threshold_scale=threshold_scale,
    )
    return (
        float(curves.regret_ftl[-1]),
        float(curves.regret_ftrl[-1]),
        float(curves.regret_smart[-1]),
        int(curves.switch_round),
    )
