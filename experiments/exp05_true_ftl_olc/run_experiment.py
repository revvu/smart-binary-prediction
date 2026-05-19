from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.olc_exact import OLCConfig, assert_trace_invariants, run_curves
from src.sequences import (
    available_generators,
    hard_calibration_scenarios,
    primary_scenarios,
)


Array = NDArray[np.float64]
ALGO_ORDER = ("ftl", "ftrl", "smart_theory", "smart_calibrated")
ALGO_LABELS = {
    "ftl": "FTL",
    "ftrl": "FTRL",
    "smart_theory": "SMART (theory)",
    "smart_calibrated": "SMART (calibrated)",
}


@dataclass(frozen=True)
class ScenarioStats:
    regret: dict[str, dict[str, Array]]
    switch_theory: dict[str, Array]
    switch_calibrated: dict[str, Array]


def _stable_offset(text: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text))


def _rng(seed: int, *, scenario: str, horizon: int, trial: int, d: int, stream: int = 0) -> np.random.Generator:
    ss = np.random.SeedSequence([seed, _stable_offset(scenario), int(horizon), int(trial), int(d), int(stream)])
    return np.random.default_rng(ss)


def _summary(vals: Array) -> dict[str, Array]:
    mean = vals.mean(axis=0)
    if vals.shape[0] <= 1:
        return {"mean": mean, "lo": mean, "hi": mean}
    se = vals.std(axis=0, ddof=1) / math.sqrt(vals.shape[0])
    return {"mean": mean, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se}


def _scenario_title(name: str) -> str:
    mapping = {
        "iid_separable_margin": "I.I.D. Separable Margin",
        "massart_10": "Massart Noise (10%)",
        "alternating_antileader": "Alternating Anti-Leader",
        "switching_leaders": "Switching Leaders (Exp02)",
        "benign_to_hard_suffix": "Benign Prefix to Hard Suffix",
        "separator_drift": "Separator Drift Diagnostic",
        "random_labels_isotropic": "Random Labels",
    }
    return mapping.get(name, name.replace("_", " ").title())


def estimate_empirical_g(
    horizons: Array,
    *,
    d: int,
    rho: float,
    trials: int,
    seed: int,
) -> dict[int, float]:
    generators = available_generators()
    hard_names = hard_calibration_scenarios()
    g_emp: dict[int, float] = {}

    for horizon in horizons:
        T = int(horizon)
        max_regret = 0.0
        for scenario in hard_names:
            gen = generators[scenario]
            for trial in range(trials):
                rng = _rng(seed, scenario=scenario, horizon=T, trial=trial, d=d, stream=11)
                seq = gen(T, d, rho, rng)
                curves = run_curves(seq.z, seq.y, OLCConfig(horizon=T, threshold=math.inf))
                max_regret = max(max_regret, float(curves.regret_ftrl[-1]))
        g_emp[T] = max(max_regret, 1e-12)

    return g_emp


def evaluate_scenario(
    scenario: str,
    horizons: Array,
    g_emp: dict[int, float],
    *,
    d: int,
    rho: float,
    trials: int,
    seed: int,
    threshold_scale: float,
) -> ScenarioStats:
    generators = available_generators()
    gen = generators[scenario]
    H = horizons.size
    regrets = {key: np.zeros((trials, H), dtype=np.float64) for key in ALGO_ORDER}
    switches_theory = np.zeros((trials, H), dtype=np.float64)
    switches_cal = np.zeros((trials, H), dtype=np.float64)

    for trial in range(trials):
        for j, horizon in enumerate(horizons):
            T = int(horizon)
            rng = _rng(seed, scenario=scenario, horizon=T, trial=trial, d=d)
            seq = gen(T, d, rho, rng)

            theory = run_curves(
                seq.z,
                seq.y,
                OLCConfig(horizon=T, threshold_scale=threshold_scale),
            )
            calibrated = run_curves(
                seq.z,
                seq.y,
                OLCConfig(horizon=T, threshold=threshold_scale * g_emp[T]),
            )
            assert_trace_invariants(theory, tol=1e-7)

            regrets["ftl"][trial, j] = float(theory.regret_ftl[-1])
            regrets["ftrl"][trial, j] = float(theory.regret_ftrl[-1])
            regrets["smart_theory"][trial, j] = float(theory.regret_smart[-1])
            regrets["smart_calibrated"][trial, j] = float(calibrated.regret_smart[-1])
            switches_theory[trial, j] = float(theory.switch_round)
            switches_cal[trial, j] = float(calibrated.switch_round)

    return ScenarioStats(
        regret={key: _summary(vals) for key, vals in regrets.items()},
        switch_theory=_summary(switches_theory),
        switch_calibrated=_summary(switches_cal),
    )


def _plot_with_band(ax: plt.Axes, x: Array, stats: dict[str, Array], label: str) -> None:
    line = ax.plot(x, stats["mean"], linewidth=2.0, label=label)[0]
    if np.any(stats["hi"] > stats["lo"]):
        ax.fill_between(x, stats["lo"], stats["hi"], color=line.get_color(), alpha=0.18, linewidth=0)


def plot_regret_grid(horizons: Array, stats_by_scenario: dict[str, ScenarioStats], out_path: Path) -> None:
    scenarios = list(stats_by_scenario)
    cols = 2
    rows = int(math.ceil(len(scenarios) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12.4, 4.1 * rows), squeeze=False)
    axes = axes.flatten()

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        stats = stats_by_scenario[scenario]
        for key in ALGO_ORDER:
            _plot_with_band(ax, horizons, stats.regret[key], ALGO_LABELS[key])
        min_lo = min(float(np.min(stats.regret[key]["lo"])) for key in ALGO_ORDER)
        max_hi = max(float(np.max(stats.regret[key]["hi"])) for key in ALGO_ORDER)
        pad = 0.05 * max(1.0, max_hi - min_lo)
        ax.set_title(_scenario_title(scenario))
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Regret")
        ax.set_ylim(bottom=min(0.0, min_lo - pad))
        ax.legend(loc="best", fontsize=9)

    for idx in range(len(scenarios), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("True-FTL Online Linear Classification: Regret by Horizon", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_empirical_g(horizons: Array, g_emp: dict[int, float], out_path: Path) -> None:
    g_vals = np.array([g_emp[int(h)] for h in horizons], dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(horizons, g_vals, marker="o", linewidth=2, label="Empirical FTRL g(T)")
    ax.plot(horizons, np.sqrt(2.0 * horizons), linestyle="--", linewidth=2, label=r"$\sqrt{2T}$")
    ax.set_title("Empirical Robust Threshold for True-FTL OLC")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Threshold")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_switch_diagnostics(
    *,
    t_max: int,
    d: int,
    rho: float,
    seed: int,
    g_emp: dict[int, float],
    out_path: Path,
) -> None:
    scenarios = ["iid_separable_margin", "benign_to_hard_suffix", "alternating_antileader", "switching_leaders"]
    generators = available_generators()
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(12.4, 3.6 * len(scenarios)), squeeze=False)

    for row, scenario in enumerate(scenarios):
        rng = _rng(seed, scenario=scenario, horizon=t_max, trial=0, d=d, stream=23)
        seq = generators[scenario](t_max, d, rho, rng)
        curves = run_curves(seq.z, seq.y, OLCConfig(horizon=t_max, threshold=g_emp[t_max]))
        rounds = np.arange(t_max + 1)

        ax_sigma = axes[row, 0]
        ax_sigma.plot(rounds, curves.sigma, linewidth=2, label=r"$\Sigma_t$")
        ax_sigma.axhline(curves.threshold, color="black", linestyle="--", linewidth=1.6, label="Threshold")
        if curves.switch_round <= t_max:
            ax_sigma.axvline(curves.switch_round, color="tab:red", linestyle=":", linewidth=1.8, label="Switch")
        ax_sigma.set_title(f"{_scenario_title(scenario)}: SMART Trace")
        ax_sigma.set_xlabel("Round")
        ax_sigma.set_ylabel(r"$\Sigma_t$")
        ax_sigma.legend(loc="best", fontsize=9)

        ax_reg = axes[row, 1]
        ax_reg.plot(rounds, curves.regret_ftl, linewidth=2, label="FTL")
        ax_reg.plot(rounds, curves.regret_ftrl, linewidth=2, label="FTRL")
        ax_reg.plot(rounds, curves.regret_smart, linewidth=2, label="SMART")
        if curves.switch_round <= t_max:
            ax_reg.axvline(curves.switch_round, color="tab:red", linestyle=":", linewidth=1.8)
        ax_reg.set_title(f"{_scenario_title(scenario)}: Prefix Regret")
        ax_reg.set_xlabel("Round")
        ax_reg.set_ylabel("Regret")
        min_reg = min(
            float(np.min(curves.regret_ftl)),
            float(np.min(curves.regret_ftrl)),
            float(np.min(curves.regret_smart)),
        )
        ax_reg.set_ylim(bottom=min(0.0, min_reg - 0.5))
        ax_reg.legend(loc="best", fontsize=9)

    fig.suptitle("SMART Switch Diagnostics in True-FTL OLC", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_calibration(
    *,
    t_max: int,
    d: int,
    rho: float,
    seed: int,
    base_threshold: float,
    trials: int,
    out_path: Path,
) -> None:
    scenarios = ["iid_separable_margin", "benign_to_hard_suffix"]
    generators = available_generators()
    scales = np.array([0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0, 3.0], dtype=float)
    smart_by_scenario: dict[str, dict[str, Array]] = {}
    switch_by_scenario: dict[str, dict[str, Array]] = {}

    for scenario in scenarios:
        gen = generators[scenario]
        smart_vals = np.zeros((trials, scales.size), dtype=float)
        switch_vals = np.zeros((trials, scales.size), dtype=float)
        for trial in range(trials):
            rng = _rng(seed, scenario=scenario, horizon=t_max, trial=trial, d=d, stream=31)
            seq = gen(t_max, d, rho, rng)
            for idx, scale in enumerate(scales):
                curves = run_curves(seq.z, seq.y, OLCConfig(horizon=t_max, threshold=scale * base_threshold))
                smart_vals[trial, idx] = float(curves.regret_smart[-1])
                switch_vals[trial, idx] = float(curves.switch_round)
        smart_by_scenario[scenario] = _summary(smart_vals)
        switch_by_scenario[scenario] = _summary(switch_vals)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    ax = axes[0]
    for scenario in scenarios:
        _plot_with_band(ax, scales, smart_by_scenario[scenario], _scenario_title(scenario))
    ax.set_title("Final Regret vs Threshold Scale")
    ax.set_xlabel("Scale times empirical g(T)")
    ax.set_ylabel("Regret at horizon")
    min_y = min(float(np.min(stats["lo"])) for stats in smart_by_scenario.values())
    ax.set_ylim(bottom=min(0.0, min_y - 0.5))
    ax.legend(loc="best")

    ax = axes[1]
    for scenario in scenarios:
        _plot_with_band(ax, scales, switch_by_scenario[scenario], _scenario_title(scenario))
    ax.axhline(0.45 * t_max, color="black", linestyle=":", linewidth=1.6, label="Hard suffix begins")
    ax.set_title("Switch Timing vs Threshold Scale")
    ax.set_xlabel("Scale times empirical g(T)")
    ax.set_ylabel("Mean switch round")
    ax.set_ylim(bottom=0.0, top=t_max + 1.0)
    ax.legend(loc="best")

    fig.suptitle("SMART Threshold Calibration Across OLC Regimes", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_dimension_sweep(
    *,
    t_max: int,
    dims: list[int],
    rho: float,
    seed: int,
    trials: int,
    g_trials: int,
    out_path: Path,
) -> None:
    scenario = "benign_to_hard_suffix"
    gen = available_generators()[scenario]
    vals = {key: np.zeros((len(dims), trials), dtype=float) for key in ("ftl", "ftrl", "smart_calibrated")}

    for i, d in enumerate(dims):
        g_emp = estimate_empirical_g(np.array([t_max], dtype=int), d=d, rho=rho, trials=g_trials, seed=seed + 7000 + d)
        threshold = g_emp[t_max]
        for trial in range(trials):
            rng = _rng(seed, scenario=scenario, horizon=t_max, trial=trial, d=d, stream=41)
            seq = gen(t_max, d, rho, rng)
            curves = run_curves(seq.z, seq.y, OLCConfig(horizon=t_max, threshold=threshold))
            vals["ftl"][i, trial] = float(curves.regret_ftl[-1])
            vals["ftrl"][i, trial] = float(curves.regret_ftrl[-1])
            vals["smart_calibrated"][i, trial] = float(curves.regret_smart[-1])

    means = {key: arr.mean(axis=1) for key, arr in vals.items()}
    cis = {
        key: 1.96 * arr.std(axis=1, ddof=1) / math.sqrt(arr.shape[1]) if arr.shape[1] > 1 else np.zeros(len(dims))
        for key, arr in vals.items()
    }

    x = np.arange(len(dims), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for offset, key in zip((-width, 0.0, width), ("ftl", "ftrl", "smart_calibrated")):
        ax.bar(x + offset, means[key], width=width, yerr=cis[key], capsize=3, label=ALGO_LABELS[key])
    ax.set_title("Dimension Robustness on Benign-to-Hard OLC")
    ax.set_xlabel("Feature dimension")
    ax.set_ylabel("Regret at horizon")
    ax.set_xticks(x, [str(d) for d in dims])
    min_y = min(float(np.min(means[key] - cis[key])) for key in means)
    ax.set_ylim(bottom=min(0.0, min_y - 0.5))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(out_path: Path, horizons: Array, stats_by_scenario: dict[str, ScenarioStats]) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "horizon", "algorithm", "mean_regret", "ci_lo", "ci_hi"])
        for scenario, stats in stats_by_scenario.items():
            for j, horizon in enumerate(horizons):
                for algo in ALGO_ORDER:
                    s = stats.regret[algo]
                    writer.writerow([scenario, int(horizon), algo, s["mean"][j], s["lo"][j], s["hi"][j]])


def write_g_csv(out_path: Path, horizons: Array, g_emp: dict[int, float]) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["horizon", "g_empirical", "sqrt_2T"])
        for horizon in horizons:
            T = int(horizon)
            writer.writerow([T, g_emp[T], math.sqrt(2.0 * T)])


def curate_figures(exp_dir: Path, generated: dict[str, tuple[Path, str, str]]) -> None:
    figures_dir = exp_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    index_lines = ["# Figure Index (exp05)", ""]
    for label, (source, title, curated_name) in generated.items():
        target = figures_dir / curated_name
        shutil.copy2(source, target)
        index_lines.extend(
            [
                f"- Label: `{label}`",
                f"  Title: `{title}`",
                f"  File: `{curated_name}`",
                f"  Source: `outputs/figures/{source.name}`",
                "",
            ]
        )

    (figures_dir / "INDEX.md").write_text("\n".join(index_lines).rstrip() + "\n", encoding="utf-8")


def run_self_test() -> None:
    rng = np.random.default_rng(0)
    gen = available_generators()["alternating_antileader"]
    seq = gen(80, 5, 0.8, rng)
    curves = run_curves(seq.z, seq.y, OLCConfig(horizon=80, threshold=math.inf))
    assert_trace_invariants(curves, tol=1e-8)

    high = run_curves(seq.z, seq.y, OLCConfig(horizon=80, threshold=1e9))
    if abs(float(high.regret_smart[-1] - high.regret_ftl[-1])) > 1e-9:
        raise AssertionError("high threshold should make SMART match FTL")

    low = run_curves(seq.z, seq.y, OLCConfig(horizon=80, threshold=0.0))
    if low.switch_round != 2:
        raise AssertionError(f"zero threshold should switch after round 1, got {low.switch_round}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact true-FTL SMART experiments for online linear classification.")
    parser.add_argument("--t-max", type=int, default=1000)
    parser.add_argument("--t-step", type=int, default=100)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--g-trials", type=int, default=8)
    parser.add_argument("--calibration-trials", type=int, default=16)
    parser.add_argument("--dimension-trials", type=int, default=12)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    parser.add_argument("--scenario", nargs="*", default=primary_scenarios())
    parser.add_argument("--skip-dimension-sweep", action="store_true")
    parser.add_argument("--paper-profile", action="store_true", help="Use the heavier T=2000, 64-trial profile.")
    parser.add_argument("--quick", action="store_true", help="Small run for smoke testing.")
    parser.add_argument("--self-test", action="store_true", help="Run invariant checks and exit.")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        print("Self-test passed.")
        return

    if args.paper_profile:
        args.t_max = 2000
        args.t_step = 100
        args.trials = 64
        args.g_trials = 24
        args.calibration_trials = 32
        args.dimension_trials = 32

    if args.quick:
        args.t_max = min(args.t_max, 500)
        args.t_step = max(args.t_step, 100)
        args.trials = min(args.trials, 8)
        args.g_trials = min(args.g_trials, 6)
        args.calibration_trials = min(args.calibration_trials, 8)
        args.dimension_trials = min(args.dimension_trials, 6)

    if not (0.0 < args.rho <= 1.0):
        raise ValueError("--rho must be in (0, 1]")
    if args.t_max % args.t_step != 0:
        raise ValueError("--t-max must be divisible by --t-step")

    exp_dir = Path(__file__).resolve().parent
    output_dir = exp_dir / "outputs"
    figures_out = output_dir / "figures"
    figures_out.mkdir(parents=True, exist_ok=True)

    horizons = np.arange(args.t_step, args.t_max + 1, args.t_step, dtype=int)
    generators = available_generators()
    scenarios = list(args.scenario)
    for scenario in scenarios:
        if scenario not in generators:
            raise ValueError(f"Unknown scenario '{scenario}'. Valid: {sorted(generators)}")

    print("Estimating empirical robust threshold g(T)...")
    g_emp = estimate_empirical_g(horizons, d=args.d, rho=args.rho, trials=args.g_trials, seed=args.seed)
    write_g_csv(output_dir / "empirical_g.csv", horizons, g_emp)

    stats_by_scenario: dict[str, ScenarioStats] = {}
    print("Running horizon sweeps:")
    for scenario in scenarios:
        print(f"- {scenario}")
        stats_by_scenario[scenario] = evaluate_scenario(
            scenario,
            horizons,
            g_emp,
            d=args.d,
            rho=args.rho,
            trials=args.trials,
            seed=args.seed,
            threshold_scale=args.threshold_scale,
        )

    write_summary_csv(output_dir / "summary_regret.csv", horizons, stats_by_scenario)

    regret_path = figures_out / "exp05_olc_regret_by_horizon.png"
    g_path = figures_out / "exp05_olc_empirical_threshold.png"
    switch_path = figures_out / "exp05_olc_switch_diagnostics.png"
    calibration_path = figures_out / "exp05_olc_threshold_calibration.png"
    dimension_path = figures_out / "exp05_olc_dimension_sweep.png"

    plot_regret_grid(horizons, stats_by_scenario, regret_path)
    plot_empirical_g(horizons, g_emp, g_path)
    plot_switch_diagnostics(t_max=args.t_max, d=args.d, rho=args.rho, seed=args.seed, g_emp=g_emp, out_path=switch_path)
    plot_threshold_calibration(
        t_max=args.t_max,
        d=args.d,
        rho=args.rho,
        seed=args.seed,
        base_threshold=g_emp[args.t_max],
        trials=args.calibration_trials,
        out_path=calibration_path,
    )
    generated = {
        "fig:exp05_olc_regret_horizon": (
            regret_path,
            "True-FTL Online Linear Classification: Regret by Horizon",
            "fig_exp05_olc_regret_by_horizon.png",
        ),
        "fig:exp05_olc_empirical_threshold": (
            g_path,
            "Empirical Robust Threshold for True-FTL OLC",
            "fig_exp05_olc_empirical_threshold.png",
        ),
        "fig:exp05_olc_switch_diagnostics": (
            switch_path,
            "SMART Switch Diagnostics in True-FTL OLC",
            "fig_exp05_olc_switch_diagnostics.png",
        ),
        "fig:exp05_olc_threshold_calibration": (
            calibration_path,
            "SMART Threshold Calibration Across OLC Regimes",
            "fig_exp05_olc_threshold_calibration.png",
        ),
    }

    if not args.skip_dimension_sweep:
        plot_dimension_sweep(
            t_max=args.t_max,
            dims=[5, 20, 100],
            rho=args.rho,
            seed=args.seed,
            trials=args.dimension_trials,
            g_trials=args.g_trials,
            out_path=dimension_path,
        )
        generated["fig:exp05_olc_dimension_sweep"] = (
            dimension_path,
            "Dimension Robustness on Benign-to-Hard OLC",
            "fig_exp05_olc_dimension_sweep.png",
        )

    curate_figures(exp_dir, generated)

    print("\nFinal-regret summary at max horizon")
    last = horizons.size - 1
    for scenario in scenarios:
        stats = stats_by_scenario[scenario]
        parts = [f"{scenario:24s}"]
        for algo in ALGO_ORDER:
            parts.append(f"{ALGO_LABELS[algo]}={stats.regret[algo]['mean'][last]:.3f}")
        parts.append(f"switch_cal={stats.switch_calibrated['mean'][last]:.1f}")
        print(" ".join(parts))

    print(f"\nWrote figures to {figures_out}")
    print(f"Curated dashboard figures in {exp_dir / 'figures'}")


if __name__ == "__main__":
    main()
