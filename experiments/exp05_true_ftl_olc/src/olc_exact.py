from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class OLCConfig:
    horizon: int
    eta0: float = math.sqrt(2.0)
    threshold: float | None = None
    threshold_scale: float = 1.0
    switch_on_equal: bool = True
    norm_tol: float = 1e-6

    @property
    def threshold_value(self) -> float:
        if self.threshold is not None:
            return float(self.threshold)
        return float(self.threshold_scale * math.sqrt(2.0 * self.horizon))


@dataclass(frozen=True)
class CurveResult:
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


def _as_float64(arr: NDArray[np.floating] | NDArray[np.integer]) -> Array:
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_inputs(z: Array, y: Array, cfg: OLCConfig) -> None:
    if z.ndim != 2:
        raise ValueError("z must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if z.shape[0] != y.shape[0]:
        raise ValueError("z and y must have the same number of rounds")
    if z.shape[0] != cfg.horizon:
        raise ValueError(f"config horizon {cfg.horizon} does not match data length {z.shape[0]}")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(y)):
        raise ValueError("z and y must contain only finite values")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("labels must be exactly -1 or +1")

    row_norms = np.linalg.norm(z, axis=1)
    max_norm = float(row_norms.max(initial=0.0))
    if max_norm > 1.0 + cfg.norm_tol:
        raise ValueError(f"closed-form OLC evaluator requires ||z_t|| <= 1; max norm is {max_norm:.6g}")


def _unit(v: Array, eps: float = 1e-12) -> Array:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def ftl_action(moment: Array) -> Array:
    """True FTL action for the linearized OLC loss."""
    return _unit(moment)


def ftrl_action(moment: Array, local_round: int, eta0: float) -> Array:
    """
    Quadratic FTRL for losses c_t = -0.5 y_t z_t on the unit ball.

    The unconstrained action is eta0 / (2 sqrt(local_round)) times the
    cumulative signed feature vector, then projected to the unit ball.
    """
    x = (eta0 / (2.0 * math.sqrt(max(1, int(local_round))))) * moment
    n = float(np.linalg.norm(x))
    if n > 1.0:
        return x / n
    return x


def loss_value(z_t: Array, y_t: float, x_t: Array) -> float:
    return 0.5 * (1.0 - float(y_t) * float(np.dot(z_t, x_t)))


def comparator_loss(prefix_len: int, moment: Array) -> float:
    return 0.5 * float(prefix_len) - 0.5 * float(np.linalg.norm(moment))


def run_curves(z: NDArray[np.floating], y: NDArray[np.floating], cfg: OLCConfig) -> CurveResult:
    """
    Run true FTL, quadratic FTRL, and single-switch SMART on one sequence.

    This is exact for the repository's OLC surrogate because y_t in {+-1}
    and ||z_t|| <= 1 imply
    0.5 * |<z_t, x> - y_t| = 0.5 * (1 - y_t <z_t, x>)
    for every unit-ball action x.
    """
    z_arr = _as_float64(z)
    y_arr = _as_float64(y)
    _validate_inputs(z_arr, y_arr, cfg)

    T, d = z_arr.shape
    moment_ftl = np.zeros(d, dtype=np.float64)
    moment_ftrl = np.zeros(d, dtype=np.float64)
    moment_smart_suffix = np.zeros(d, dtype=np.float64)
    suffix_rounds = 0

    cum_ftl = np.zeros(T + 1, dtype=np.float64)
    cum_ftrl = np.zeros(T + 1, dtype=np.float64)
    cum_smart = np.zeros(T + 1, dtype=np.float64)
    comp_prefix = np.zeros(T + 1, dtype=np.float64)
    sigma = np.zeros(T + 1, dtype=np.float64)

    switched = False
    switch_round = T + 1
    threshold = cfg.threshold_value

    for t in range(1, T + 1):
        z_t = z_arr[t - 1]
        y_t = float(y_arr[t - 1])

        x_ftl = ftl_action(moment_ftl)
        x_ftrl = ftrl_action(moment_ftrl, t, cfg.eta0)
        if switched:
            x_smart = ftrl_action(moment_smart_suffix, suffix_rounds + 1, cfg.eta0)
        else:
            x_smart = x_ftl

        cum_ftl[t] = cum_ftl[t - 1] + loss_value(z_t, y_t, x_ftl)
        cum_ftrl[t] = cum_ftrl[t - 1] + loss_value(z_t, y_t, x_ftrl)
        cum_smart[t] = cum_smart[t - 1] + loss_value(z_t, y_t, x_smart)

        update = y_t * z_t
        moment_ftl += update
        moment_ftrl += update
        if switched:
            moment_smart_suffix += update
            suffix_rounds += 1

        comp_prefix[t] = comparator_loss(t, moment_ftl)
        sigma[t] = cum_ftl[t] - comp_prefix[t]

        crosses = sigma[t] >= threshold if cfg.switch_on_equal else sigma[t] > threshold
        if (not switched) and crosses:
            switched = True
            switch_round = t + 1
            moment_smart_suffix.fill(0.0)
            suffix_rounds = 0

    return CurveResult(
        regret_ftl=cum_ftl - comp_prefix,
        regret_ftrl=cum_ftrl - comp_prefix,
        regret_smart=cum_smart - comp_prefix,
        cum_ftl=cum_ftl,
        cum_ftrl=cum_ftrl,
        cum_smart=cum_smart,
        comp_prefix=comp_prefix,
        sigma=sigma,
        threshold=threshold,
        switch_round=switch_round,
    )


def assert_trace_invariants(result: CurveResult, *, tol: float = 1e-8) -> None:
    min_delta = float(np.min(np.diff(result.sigma))) if result.sigma.size > 1 else 0.0
    if min_delta < -tol:
        raise AssertionError(f"Sigma trace is not monotone: min increment {min_delta}")
    final_gap = abs(float(result.sigma[-1] - result.regret_ftl[-1]))
    if final_gap > tol:
        raise AssertionError(f"Sigma_T does not match FTL regret: gap {final_gap}")
    if float(np.min(result.regret_ftl)) < -tol:
        raise AssertionError("FTL regret trace should be nonnegative in this exact linear setting")
