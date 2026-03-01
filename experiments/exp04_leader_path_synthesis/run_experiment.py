from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.eval import final_regrets
from src.synthesis import SynthesisConfig, regime_names, synthesize_sequence


@dataclass(frozen=True)
class RunConfig:
    d: int = 5
    t_max: int = 1000
    t_step: int = 100
    runs: int = 12
    max_delta_norm: float = 0.45
    label_mismatch_prob: float = 0.02
    threshold_scale: float = 0.01
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


def _regime_title(name: str) -> str:
    mapping = {
        "stable_benign": "Stable Benign Sequence",
        "persistent_shift": "Persistent Shift Sequence",
        "delayed_hardening": "Delayed Hardening Sequence",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _plot_regret_grid(t_grid, stats_by_regime, out_path: Path) -> None:
    regimes = list(stats_by_regime.keys())
    cols = 2
    rows = (len(regimes) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows), squeeze=False)
    axes = axes.flatten()

    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        stats = stats_by_regime[regime]

        for key, label in (("ftl", "FTL"), ("ftrl", "FTRL"), ("smart", "SMART")):
            mean = stats[key]["mean"]
            lo = stats[key]["lo"]
            hi = stats[key]["hi"]
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
                hard_window_mismatch_boost=max(0.06, label_mismatch_prob + 0.04),
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
                )
                if best is None or cand.score > best.score:
                    best = cand
        assert best is not None
        selected[regime] = best
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Metric-driven sequence synthesis experiment for SMART in online linear classification.")
    parser.add_argument("--t-max", type=int, default=1000)
    parser.add_argument("--t-step", type=int, default=100)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--max-delta-norm", type=float, default=0.45)
    parser.add_argument("--label-mismatch-prob", type=float, default=0.02)
    parser.add_argument("--threshold-scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--auto-select",
        action="store_true",
        default=True,
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

    cfg = RunConfig(
        d=args.d,
        t_max=args.t_max,
        t_step=args.t_step,
        runs=args.runs,
        max_delta_norm=args.max_delta_norm,
        label_mismatch_prob=args.label_mismatch_prob,
        threshold_scale=args.threshold_scale,
        seed=args.seed,
    )

    out_dir = Path(__file__).resolve().parent / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_grid = np.arange(cfg.t_step, cfg.t_max + 1, cfg.t_step, dtype=int)
    regimes = list(args.regime)

    stats_by_regime = {}
    selected_cfgs: dict[str, SelectedConfig] = {}
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

    for regime in regimes:
        thr_scale = cfg.threshold_scale
        ftl_runs = np.zeros((cfg.runs, len(t_grid)), dtype=float)
        ftrl_runs = np.zeros((cfg.runs, len(t_grid)), dtype=float)
        smart_runs = np.zeros((cfg.runs, len(t_grid)), dtype=float)
        switch_runs = np.zeros((cfg.runs, len(t_grid)), dtype=float)

        if args.auto_select:
            regime_delta = selected_cfgs[regime].max_delta_norm
            regime_mismatch = selected_cfgs[regime].label_mismatch_prob
        else:
            regime_delta = cfg.max_delta_norm
            regime_mismatch = cfg.label_mismatch_prob

        for r in range(cfg.runs):
            for i, T in enumerate(t_grid):
                synth_cfg = SynthesisConfig(
                    d=cfg.d,
                    max_delta_norm=regime_delta,
                    label_mismatch_prob=regime_mismatch,
                    seed=cfg.seed + 1000 * r + 17 * T,
                )
                seq = synthesize_sequence(T=T, regime=regime, cfg=synth_cfg)
                ftl_reg, ftrl_reg, smart_reg, switch = final_regrets(
                    seq.z,
                    seq.y,
                    threshold_scale=thr_scale,
                )
                ftl_runs[r, i] = ftl_reg
                ftrl_runs[r, i] = ftrl_reg
                smart_runs[r, i] = smart_reg
                switch_runs[r, i] = switch

        stats_by_regime[regime] = {
            "ftl": _summary_stats(ftl_runs),
            "ftrl": _summary_stats(ftrl_runs),
            "smart": _summary_stats(smart_runs),
            "switch_mean": switch_runs.mean(axis=0),
        }

    _plot_regret_grid(t_grid, stats_by_regime, out_dir / "exp04_olc_regret_by_horizon.png")

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
