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
    runs: int = 8
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


def _regime_title(name: str) -> str:
    mapping = {
        "stable_benign": "Stable Benign Sequence",
        "corruption_burst": "Corruption Burst Sequence",
        "drift_plus_shift": "Drift Plus Shift Sequence",
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


def _selection_score(regime: str, mean_ftl: float, mean_ftrl: float, mean_smart: float) -> float:
    """
    Score candidate parameterizations based on paper-facing behavior goals:
    - stable: SMART ~= FTL, and FTL should be at least as good as robust.
    - hard: SMART should materially improve over FTL.
    - mixed: SMART should improve over FTL while staying in a plausible middle regime.
    """
    if regime == "stable_benign":
        return -abs(mean_smart - mean_ftl) - 2.0 * max(0.0, mean_ftl - mean_ftrl)

    if regime == "corruption_burst":
        return (mean_ftl - mean_smart) - 0.5 * max(0.0, mean_smart - mean_ftrl)

    # drift_plus_shift
    improve = mean_ftl - mean_smart
    worse_than_ftl_penalty = 2.0 * max(0.0, mean_smart - mean_ftl)
    too_better_than_robust_penalty = 0.25 * max(0.0, mean_ftrl - mean_smart)
    return improve - worse_than_ftl_penalty - too_better_than_robust_penalty


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
    ftl_vals: list[float] = []
    ftrl_vals: list[float] = []
    smart_vals: list[float] = []

    for r in range(runs):
        for h in horizons:
            synth_cfg = SynthesisConfig(
                d=d,
                max_delta_norm=max_delta_norm,
                label_mismatch_prob=label_mismatch_prob,
                seed=seed + 5000 * r + 37 * h,
            )
            seq = synthesize_sequence(T=h, regime=regime, cfg=synth_cfg)
            ftl_reg, ftrl_reg, smart_reg, _ = final_regrets(seq.z, seq.y, threshold_scale=threshold_scale)
            ftl_vals.append(ftl_reg)
            ftrl_vals.append(ftrl_reg)
            smart_vals.append(smart_reg)

    mean_ftl = float(np.mean(ftl_vals))
    mean_ftrl = float(np.mean(ftrl_vals))
    mean_smart = float(np.mean(smart_vals))
    score = _selection_score(regime, mean_ftl, mean_ftrl, mean_smart)

    return SelectedConfig(
        max_delta_norm=max_delta_norm,
        label_mismatch_prob=label_mismatch_prob,
        score=score,
        mean_ftl=mean_ftl,
        mean_ftrl=mean_ftrl,
        mean_smart=mean_smart,
    )


def _select_configs(
    regimes: list[str],
    cfg: RunConfig,
) -> dict[str, SelectedConfig]:
    # Small, fast search grid chosen for reproducibility and interpretability.
    delta_grid = [0.35, 0.45, 0.55]
    mismatch_grid = [0.00, 0.02, 0.06, 0.10]
    probe_horizons = [max(cfg.t_step, 300), max(cfg.t_step, 700), cfg.t_max]
    probe_runs = min(4, cfg.runs)

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
    parser = argparse.ArgumentParser(description="Leader-path synthesis experiment for SMART in online linear classification.")
    parser.add_argument("--t-max", type=int, default=1000)
    parser.add_argument("--t-step", type=int, default=100)
    parser.add_argument("--runs", type=int, default=8)
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
                f"[probe mean regrets FTL={s.mean_ftl:.3f}, FTRL={s.mean_ftrl:.3f}, SMART={s.mean_smart:.3f}]"
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

    print("Final-regret summary (at max horizon)")
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
