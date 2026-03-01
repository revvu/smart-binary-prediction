from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.eval import final_regrets, run_curves
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

        ax.set_title(regime.replace("_", " "))
        ax.set_xlabel("T")
        ax.set_ylabel("Final regret")
        ax.legend(loc="best")

    for j in range(len(regimes), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Leader-synthesized realistic regimes: final regret vs horizon", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_single_sequence_diagnostics(regime: str, seq, curves, out_dir: Path) -> None:
    T = seq.z.shape[0]
    t = np.arange(1, T + 1)

    # Leader alignment
    cos_sim = np.sum(seq.target_leaders * seq.realized_leaders, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(t, curves.regret_ftl[1:], label="FTL", linewidth=2)
    axes[0].plot(t, curves.regret_ftrl[1:], label="FTRL", linewidth=2)
    axes[0].plot(t, curves.regret_smart[1:], label="SMART", linewidth=2)
    axes[0].set_ylabel("Prefix regret")
    axes[0].set_title(f"{regime.replace('_', ' ')}: regret trajectories")
    axes[0].legend(loc="best")

    axes[1].plot(t, curves.sigma[1:], label=r"$\Sigma_t$", linewidth=2)
    axes[1].axhline(curves.threshold, linestyle="--", linewidth=1.5, label="threshold")
    if 1 <= curves.switch_round <= T:
        axes[1].axvline(curves.switch_round, linestyle=":", linewidth=1.5, label=f"switch={curves.switch_round}")
    axes[1].set_ylabel("Switch statistic")
    axes[1].legend(loc="best")

    axes[2].plot(t, cos_sim, linewidth=2)
    axes[2].set_ylabel("cos(target, realized)")
    axes[2].set_xlabel("t")
    axes[2].set_ylim(-1.05, 1.05)

    fig.tight_layout()
    fig.savefig(out_dir / f"{regime}_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _summary_stats(vals: np.ndarray) -> dict[str, np.ndarray]:
    mean = vals.mean(axis=0)
    if vals.shape[0] <= 1:
        return {"mean": mean, "lo": mean, "hi": mean}
    se = vals.std(axis=0, ddof=1) / np.sqrt(vals.shape[0])
    return {"mean": mean, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se}


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

    stats_by_regime = {}

    for regime in args.regime:
        thr_scale = cfg.threshold_scale
        ftl_runs = np.zeros((cfg.runs, len(t_grid)), dtype=float)
        ftrl_runs = np.zeros((cfg.runs, len(t_grid)), dtype=float)
        smart_runs = np.zeros((cfg.runs, len(t_grid)), dtype=float)
        switch_runs = np.zeros((cfg.runs, len(t_grid)), dtype=float)

        for r in range(cfg.runs):
            for i, T in enumerate(t_grid):
                synth_cfg = SynthesisConfig(
                    d=cfg.d,
                    max_delta_norm=cfg.max_delta_norm,
                    label_mismatch_prob=cfg.label_mismatch_prob,
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

        # Diagnostic plot at max horizon using first run seed.
        diag_cfg = SynthesisConfig(
            d=cfg.d,
            max_delta_norm=cfg.max_delta_norm,
            label_mismatch_prob=cfg.label_mismatch_prob,
            seed=cfg.seed + 99991,
        )
        seq_diag = synthesize_sequence(T=cfg.t_max, regime=regime, cfg=diag_cfg)
        curves = run_curves(seq_diag.z, seq_diag.y, threshold_scale=thr_scale)
        _plot_single_sequence_diagnostics(regime, seq_diag, curves, out_dir)

    _plot_regret_grid(t_grid, stats_by_regime, out_dir / "regret_vs_horizon_by_regime.png")

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
