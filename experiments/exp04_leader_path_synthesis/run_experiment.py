from __future__ import annotations

import argparse
import concurrent.futures as cf
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from src.eval import final_regrets, get_eval_mode, set_eval_mode
from src.synthesis import SynthesisConfig, regime_names, synthesize_sequence


@dataclass(frozen=True)
class RunConfig:
    d: int = 5
    t_max: int = 1000
    t_step: int = 25
    runs: int = 48
    max_delta_norm: float = 0.45
    label_mismatch_prob: float = 0.02
    threshold_scale: float = 0.01
    direction_noise_scale: float = 0.35
    seed: int = 7


@dataclass(frozen=True)
class SelectedConfig:
    max_delta_norm: float
    label_mismatch_prob: float
    score: float
    mean_ftl: float
    mean_ftrl: float
    mean_smart: float
    curve_ftl: np.ndarray
    curve_ftrl: np.ndarray
    curve_smart: np.ndarray
    mean_ci_halfwidth: float


@dataclass(frozen=True)
class RegimeVisualConfig:
    max_delta_norm: float
    label_mismatch_prob: float
    threshold_scale: float
    direction_noise_scale: float


@dataclass(frozen=True)
class RegimeEvalConfig:
    d: int
    max_delta_norm: float
    label_mismatch_prob: float
    hard_window_mismatch_boost: float
    direction_noise_scale: float
    threshold_scale: float
    seed: int


def _hard_window_boost(base_mismatch: float) -> float:
    # Keep hard-window corruption comparable between config search and final plotting.
    return max(0.06, float(base_mismatch) + 0.04)


def _regime_title(name: str) -> str:
    mapping = {
        "stable_benign": "Stable Benign Sequence",
        "persistent_shift": "Persistent Shift Sequence",
        "delayed_hardening": "Delayed Hardening Sequence",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _smooth_curve(curve: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or curve.size < 3:
        return curve.copy()
    w = int(max(1, window))
    if w % 2 == 0:
        w += 1
    if w > curve.size:
        w = curve.size if curve.size % 2 == 1 else curve.size - 1
    if w <= 1:
        return curve.copy()
    pad = w // 2
    padded = np.pad(curve, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(padded, kernel, mode="valid")


def _plot_regret_grid(t_grid, stats_by_regime, out_path: Path, *, smooth_window: int) -> None:
    regimes = list(stats_by_regime.keys())
    cols = 2
    rows = (len(regimes) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows), squeeze=False)
    axes = axes.flatten()

    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        stats = stats_by_regime[regime]

        for key, label in (("ftl", "FTL"), ("ftrl", "FTRL"), ("smart", "SMART")):
            mean = _smooth_curve(stats[key]["mean"], smooth_window)
            lo = _smooth_curve(stats[key]["lo"], smooth_window)
            hi = _smooth_curve(stats[key]["hi"], smooth_window)
            lo = np.minimum(lo, mean)
            hi = np.maximum(hi, mean)
            line = ax.plot(t_grid, mean, linewidth=2, label=label)[0]
            if np.any(hi > lo):
                ax.fill_between(t_grid, lo, hi, alpha=0.18, color=line.get_color())

        ax.set_title(_regime_title(regime))
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Regret")
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="best")

    for j in range(len(regimes), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Online Linear Classification: Regret by Horizon", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _summary_stats(vals: np.ndarray) -> dict[str, np.ndarray]:
    mean = vals.mean(axis=0)
    if vals.shape[0] <= 1:
        return {"mean": mean, "lo": mean, "hi": mean}
    se = vals.std(axis=0, ddof=1) / np.sqrt(vals.shape[0])
    return {"mean": mean, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se}


def _curve_roughness(curve: np.ndarray) -> float:
    if curve.size < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(curve, n=2))))


def _monotone_drop_penalty(curve: np.ndarray) -> float:
    if curve.size < 2:
        return 0.0
    d = np.diff(curve)
    return float(np.sum(np.maximum(0.0, -d)))


def _selection_score(
    regime: str,
    curve_ftl: np.ndarray,
    curve_ftrl: np.ndarray,
    curve_smart: np.ndarray,
    mean_ci_halfwidth: float,
) -> float:
    """
    Score candidate parameterizations based on paper-facing behavior goals:
    - stable: SMART ~= FTL, and FTL should be at least as good as robust.
    - hard: SMART should materially improve over FTL.
    - mixed: SMART should improve over FTL while staying in a plausible middle regime.
    """
    mean_ftl = float(np.mean(curve_ftl))
    mean_ftrl = float(np.mean(curve_ftrl))
    mean_smart = float(np.mean(curve_smart))
    ftl_last = float(curve_ftl[-1])
    ftrl_last = float(curve_ftrl[-1])
    smart_last = float(curve_smart[-1])
    rough_pen = _curve_roughness(curve_smart) + 0.5 * _curve_roughness(curve_ftl)
    drop_pen = _monotone_drop_penalty(curve_smart) + 0.5 * _monotone_drop_penalty(curve_ftl)
    ci_pen = 1.5 * mean_ci_halfwidth
    negative_pen = 4.0 * max(0.0, -float(np.min(curve_smart)))

    if regime == "stable_benign":
        return (
            -abs(smart_last - ftl_last)
            - 0.75 * float(np.mean(np.abs(curve_smart - curve_ftl)))
            - 2.0 * max(0.0, ftl_last - ftrl_last)
            - rough_pen
            - drop_pen
            - ci_pen
            - negative_pen
        )

    if regime == "persistent_shift":
        return (
            (ftl_last - smart_last)
            + 0.40 * (mean_ftl - mean_smart)
            - 0.40 * max(0.0, smart_last - ftrl_last)
            - 1.2 * rough_pen
            - 1.2 * drop_pen
            - ci_pen
            - negative_pen
        )

    # delayed_hardening
    midpoint = 0.5 * (curve_ftl + curve_ftrl)
    mid_pen = float(np.mean(np.abs(curve_smart - midpoint)))
    return (
        0.8 * (ftl_last - smart_last)
        + 0.4 * (mean_ftl - mean_smart)
        - 0.6 * mid_pen
        - 0.8 * rough_pen
        - 0.8 * drop_pen
        - ci_pen
        - negative_pen
    )


def _evaluate_candidate(
    *,
    regime: str,
    horizons: list[int],
    runs: int,
    threshold_scale: float,
    d: int,
    seed: int,
    max_delta_norm: float,
    label_mismatch_prob: float,
    direction_noise_scale: float,
) -> SelectedConfig:
    H = len(horizons)
    ftl_vals = np.zeros((runs, H), dtype=float)
    ftrl_vals = np.zeros((runs, H), dtype=float)
    smart_vals = np.zeros((runs, H), dtype=float)

    for r in range(runs):
        for j, h in enumerate(horizons):
            synth_cfg = SynthesisConfig(
                d=d,
                max_delta_norm=max_delta_norm,
                label_mismatch_prob=label_mismatch_prob,
                hard_window_mismatch_boost=_hard_window_boost(label_mismatch_prob),
                direction_noise_scale=direction_noise_scale,
                seed=seed + 5000 * r + 37 * h,
            )
            seq = synthesize_sequence(T=h, regime=regime, cfg=synth_cfg)
            ftl_reg, ftrl_reg, smart_reg, _ = final_regrets(seq.z, seq.y, threshold_scale=threshold_scale)
            ftl_vals[r, j] = ftl_reg
            ftrl_vals[r, j] = ftrl_reg
            smart_vals[r, j] = smart_reg

    curve_ftl = np.mean(ftl_vals, axis=0)
    curve_ftrl = np.mean(ftrl_vals, axis=0)
    curve_smart = np.mean(smart_vals, axis=0)
    mean_ftl = float(np.mean(curve_ftl))
    mean_ftrl = float(np.mean(curve_ftrl))
    mean_smart = float(np.mean(curve_smart))

    if runs > 1:
        se_ftl = np.std(ftl_vals, axis=0, ddof=1) / np.sqrt(runs)
        se_ftrl = np.std(ftrl_vals, axis=0, ddof=1) / np.sqrt(runs)
        se_smart = np.std(smart_vals, axis=0, ddof=1) / np.sqrt(runs)
        mean_ci_halfwidth = float(np.mean(1.96 * (se_ftl + se_ftrl + se_smart) / 3.0))
    else:
        mean_ci_halfwidth = 0.0

    score = _selection_score(regime, curve_ftl, curve_ftrl, curve_smart, mean_ci_halfwidth)

    return SelectedConfig(
        max_delta_norm=max_delta_norm,
        label_mismatch_prob=label_mismatch_prob,
        score=score,
        mean_ftl=mean_ftl,
        mean_ftrl=mean_ftrl,
        mean_smart=mean_smart,
        curve_ftl=curve_ftl,
        curve_ftrl=curve_ftrl,
        curve_smart=curve_smart,
        mean_ci_halfwidth=mean_ci_halfwidth,
    )


def _select_configs(
    regimes: list[str],
    cfg: RunConfig,
) -> dict[str, SelectedConfig]:
    # Small, fast search grid chosen for reproducibility and interpretability.
    delta_grid = [0.35, 0.45, 0.55]
    mismatch_grid = [0.00, 0.01, 0.02, 0.04, 0.06]
    probe_horizons = list(np.arange(max(cfg.t_step, 200), cfg.t_max + 1, max(cfg.t_step, 200), dtype=int))
    probe_runs = min(6, cfg.runs)

    selected: dict[str, SelectedConfig] = {}
    for regime in regimes:
        regime_seed_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(regime))
        best: SelectedConfig | None = None
        for delta in delta_grid:
            for mismatch in mismatch_grid:
                cand = _evaluate_candidate(
                    regime=regime,
                    horizons=probe_horizons,
                    runs=probe_runs,
                    threshold_scale=cfg.threshold_scale,
                    d=cfg.d,
                    seed=cfg.seed + regime_seed_offset,
                    max_delta_norm=delta,
                    label_mismatch_prob=mismatch,
                    direction_noise_scale=cfg.direction_noise_scale,
                )
                if best is None or cand.score > best.score:
                    best = cand
        assert best is not None
        selected[regime] = best
    return selected


def _simulate_one_run(
    regime: str,
    t_grid: np.ndarray,
    coupled_horizons: bool,
    run_idx: int,
    cfg: RegimeEvalConfig,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ftl_row = np.zeros(len(t_grid), dtype=float)
    ftrl_row = np.zeros(len(t_grid), dtype=float)
    smart_row = np.zeros(len(t_grid), dtype=float)
    switch_row = np.zeros(len(t_grid), dtype=float)

    if not coupled_horizons:
        for i, T in enumerate(t_grid):
            synth_cfg = SynthesisConfig(
                d=cfg.d,
                max_delta_norm=cfg.max_delta_norm,
                label_mismatch_prob=cfg.label_mismatch_prob,
                hard_window_mismatch_boost=cfg.hard_window_mismatch_boost,
                direction_noise_scale=cfg.direction_noise_scale,
                seed=cfg.seed + 1000 * run_idx + 17 * int(T),
            )
            seq = synthesize_sequence(T=int(T), regime=regime, cfg=synth_cfg)
            ftl_reg, ftrl_reg, smart_reg, switch = final_regrets(
                seq.z,
                seq.y,
                threshold_scale=cfg.threshold_scale,
            )
            ftl_row[i] = ftl_reg
            ftrl_row[i] = ftrl_reg
            smart_row[i] = smart_reg
            switch_row[i] = switch
    else:
        t_max = int(t_grid[-1])
        synth_cfg = SynthesisConfig(
            d=cfg.d,
            max_delta_norm=cfg.max_delta_norm,
            label_mismatch_prob=cfg.label_mismatch_prob,
            hard_window_mismatch_boost=cfg.hard_window_mismatch_boost,
            direction_noise_scale=cfg.direction_noise_scale,
            seed=cfg.seed + 1000 * run_idx + 17 * t_max,
        )
        seq_full = synthesize_sequence(T=t_max, regime=regime, cfg=synth_cfg)
        for i, T in enumerate(t_grid):
            z = seq_full.z[: int(T)]
            y = seq_full.y[: int(T)]
            ftl_reg, ftrl_reg, smart_reg, switch = final_regrets(
                z,
                y,
                threshold_scale=cfg.threshold_scale,
            )
            ftl_row[i] = ftl_reg
            ftrl_row[i] = ftrl_reg
            smart_row[i] = smart_reg
            switch_row[i] = switch

    return run_idx, ftl_row, ftrl_row, smart_row, switch_row


def _illustrative_regime_configs(base_threshold_scale: float) -> dict[str, RegimeVisualConfig]:
    # Fixed profiles chosen to create clear, visually distinct narratives:
    # 1) benign: FTL/SMART strong, robust baseline conservative,
    # 2) hard/reverse: robust baseline strong, FTL weak, SMART helpful,
    # 3) mixed: SMART sits between optimistic and robust extremes.
    return {
        "stable_benign": RegimeVisualConfig(
            max_delta_norm=0.35,
            label_mismatch_prob=0.00,
            threshold_scale=max(base_threshold_scale, 0.015),
            direction_noise_scale=0.65,
        ),
        "persistent_shift": RegimeVisualConfig(
            max_delta_norm=0.45,
            label_mismatch_prob=0.05,
            threshold_scale=0.006,
            direction_noise_scale=1.20,
        ),
        "delayed_hardening": RegimeVisualConfig(
            max_delta_norm=0.45,
            label_mismatch_prob=0.12,
            threshold_scale=0.005,
            direction_noise_scale=0.50,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Metric-driven sequence synthesis experiment for SMART in online linear classification.")
    parser.add_argument("--t-max", type=int, default=1000)
    parser.add_argument("--t-step", type=int, default=25)
    parser.add_argument("--runs", type=int, default=48)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers over independent runs.")
    parser.add_argument("--engine", choices=["auto", "python", "numba"], default="auto", help="Evaluation engine.")
    parser.add_argument("--max-delta-norm", type=float, default=0.45)
    parser.add_argument("--label-mismatch-prob", type=float, default=0.02)
    parser.add_argument("--threshold-scale", type=float, default=0.01)
    parser.add_argument("--direction-noise-scale", type=float, default=0.35)
    parser.add_argument("--smooth-window", type=int, default=1, help="Moving-average window for plotted means/CI.")
    parser.add_argument(
        "--coupled-horizons",
        action="store_true",
        default=False,
        help="Use one long sequence per run and horizon prefixes (trace-like; not primary paper protocol).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--auto-select",
        action="store_true",
        default=False,
        help="Enable metric-driven per-regime parameter selection.",
    )
    parser.add_argument(
        "--no-auto-select",
        action="store_false",
        dest="auto_select",
        help="Disable metric-driven selection and use provided base parameters directly.",
    )
    parser.add_argument("--regime", nargs="*", default=regime_names())
    args = parser.parse_args()
    set_eval_mode(args.engine)

    cfg = RunConfig(
        d=args.d,
        t_max=args.t_max,
        t_step=args.t_step,
        runs=args.runs,
        max_delta_norm=args.max_delta_norm,
        label_mismatch_prob=args.label_mismatch_prob,
        threshold_scale=args.threshold_scale,
        direction_noise_scale=args.direction_noise_scale,
        seed=args.seed,
    )
    if cfg.t_max % cfg.t_step != 0:
        raise ValueError("--t-max must be divisible by --t-step for consistent coupled-horizon evaluation.")

    out_dir = Path(__file__).resolve().parent / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_grid = np.arange(cfg.t_step, cfg.t_max + 1, cfg.t_step, dtype=int)
    regimes = list(args.regime)
    print(f"Evaluation engine: requested={args.engine}, active={get_eval_mode()}")
    print(f"Parallel jobs: {max(1, int(args.jobs))}")

    stats_by_regime = {}
    selected_cfgs: dict[str, SelectedConfig] = {}
    visual_cfgs = _illustrative_regime_configs(cfg.threshold_scale)
    if args.auto_select:
        selected_cfgs = _select_configs(regimes, cfg)
        print("Metric-driven selected sequence parameters:")
        for regime in regimes:
            s = selected_cfgs[regime]
            print(
                f"- {regime:16s} "
                f"max_delta={s.max_delta_norm:.2f} "
                f"mismatch={s.label_mismatch_prob:.2f} "
                f"score={s.score:.3f} "
                f"[probe mean regrets FTL={s.mean_ftl:.3f}, FTRL={s.mean_ftrl:.3f}, SMART={s.mean_smart:.3f}, "
                f"mean_ci={s.mean_ci_halfwidth:.3f}]"
            )
    else:
        print("Using fixed illustrative regime parameters:")
        for regime in regimes:
            if regime in visual_cfgs:
                v = visual_cfgs[regime]
                print(
                    f"- {regime:16s} "
                    f"max_delta={v.max_delta_norm:.2f} "
                    f"mismatch={v.label_mismatch_prob:.2f} "
                    f"threshold_scale={v.threshold_scale:.3f} "
                    f"dir_noise={v.direction_noise_scale:.2f}"
                )

    for regime in regimes:
        thr_scale = cfg.threshold_scale
        ftl_runs = np.zeros((cfg.runs, len(t_grid)), dtype=float)
        ftrl_runs = np.zeros((cfg.runs, len(t_grid)), dtype=float)
        smart_runs = np.zeros((cfg.runs, len(t_grid)), dtype=float)
        switch_runs = np.zeros((cfg.runs, len(t_grid)), dtype=float)

        if args.auto_select:
            regime_delta = selected_cfgs[regime].max_delta_norm
            regime_mismatch = selected_cfgs[regime].label_mismatch_prob
            thr_scale = cfg.threshold_scale
            regime_noise_scale = cfg.direction_noise_scale
        else:
            if regime in visual_cfgs:
                regime_delta = visual_cfgs[regime].max_delta_norm
                regime_mismatch = visual_cfgs[regime].label_mismatch_prob
                thr_scale = visual_cfgs[regime].threshold_scale
                regime_noise_scale = visual_cfgs[regime].direction_noise_scale
            else:
                regime_delta = cfg.max_delta_norm
                regime_mismatch = cfg.label_mismatch_prob
                thr_scale = cfg.threshold_scale
                regime_noise_scale = cfg.direction_noise_scale

        eval_cfg = RegimeEvalConfig(
            d=cfg.d,
            max_delta_norm=regime_delta,
            label_mismatch_prob=regime_mismatch,
            hard_window_mismatch_boost=_hard_window_boost(regime_mismatch),
            direction_noise_scale=regime_noise_scale,
            threshold_scale=thr_scale,
            seed=cfg.seed,
        )

        if args.jobs <= 1:
            for r in range(cfg.runs):
                run_idx, ftl_row, ftrl_row, smart_row, switch_row = _simulate_one_run(
                    regime,
                    t_grid,
                    args.coupled_horizons,
                    r,
                    eval_cfg,
                )
                ftl_runs[run_idx] = ftl_row
                ftrl_runs[run_idx] = ftrl_row
                smart_runs[run_idx] = smart_row
                switch_runs[run_idx] = switch_row
        else:
            backend = "process"
            t0 = perf_counter()
            try:
                with cf.ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
                    futs = [
                        ex.submit(
                            _simulate_one_run,
                            regime,
                            t_grid,
                            args.coupled_horizons,
                            r,
                            eval_cfg,
                        )
                        for r in range(cfg.runs)
                    ]
                    for fut in cf.as_completed(futs):
                        run_idx, ftl_row, ftrl_row, smart_row, switch_row = fut.result()
                        ftl_runs[run_idx] = ftl_row
                        ftrl_runs[run_idx] = ftrl_row
                        smart_runs[run_idx] = smart_row
                        switch_runs[run_idx] = switch_row
            except (PermissionError, OSError):
                # Some environments disallow multiprocessing semaphores.
                # Thread fallback can be significantly slower for this workload, so use serial fallback.
                backend = "serial-fallback"
                for r in range(cfg.runs):
                    run_idx, ftl_row, ftrl_row, smart_row, switch_row = _simulate_one_run(
                        regime,
                        t_grid,
                        args.coupled_horizons,
                        r,
                        eval_cfg,
                    )
                    ftl_runs[run_idx] = ftl_row
                    ftrl_runs[run_idx] = ftrl_row
                    smart_runs[run_idx] = smart_row
                    switch_runs[run_idx] = switch_row
            print(f"[{regime}] parallel backend={backend}, elapsed={perf_counter()-t0:.2f}s")

        stats_by_regime[regime] = {
            "ftl": _summary_stats(ftl_runs),
            "ftrl": _summary_stats(ftrl_runs),
            "smart": _summary_stats(smart_runs),
            "switch_mean": switch_runs.mean(axis=0),
        }

    _plot_regret_grid(
        t_grid,
        stats_by_regime,
        out_dir / "exp04_olc_regret_by_horizon.png",
        smooth_window=max(1, int(args.smooth_window)),
    )

    print("Regret summary (at max horizon)")
    for regime in args.regime:
        s = stats_by_regime[regime]
        print(
            f"{regime:18s} "
            f"FTL={s['ftl']['mean'][-1]:.3f} "
            f"FTRL={s['ftrl']['mean'][-1]:.3f} "
            f"SMART={s['smart']['mean'][-1]:.3f} "
            f"switch_mean={s['switch_mean'][-1]:.1f}"
        )


if __name__ == "__main__":
    main()
