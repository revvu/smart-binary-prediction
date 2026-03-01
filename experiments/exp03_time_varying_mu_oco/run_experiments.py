from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.oco_smart import OCOConfig, run_ftl, run_ogd, run_smart
from src.sequence_generators import available_generators

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Scenario:
    name: str
    generator: Callable[[int, np.random.Generator], Array]


def _summary_curves(vals: Array) -> dict[str, Array]:
    mean = vals.mean(axis=0)
    if vals.shape[0] <= 1:
        return {"mean": mean, "lo": mean, "hi": mean}
    se = vals.std(axis=0, ddof=1) / np.sqrt(vals.shape[0])
    return {"mean": mean, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se}


def _scenario_title(name: str) -> str:
    mapping = {
        "stable_benign": "Stable Benign Sequence",
        "corruption_burst": "Corruption Burst Sequence",
        "drift_plus_shift": "Drift Plus Shift Sequence",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _plot_regret_grid(horizons: Array, stats_by_scenario: dict[str, dict[str, dict[str, Array]]], out_path: Path) -> None:
    scenarios = list(stats_by_scenario.keys())
    cols = 2
    rows = (len(scenarios) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows), squeeze=False)
    axes = axes.flatten()

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        stats = stats_by_scenario[scenario]
        for key, label in (("ftl", "FTL"), ("ogd", "OGD"), ("smart", "SMART")):
            mean = stats[key]["mean"]
            lo = stats[key]["lo"]
            hi = stats[key]["hi"]
            line = ax.plot(horizons, mean, linewidth=2, label=label)[0]
            if np.any(hi > lo):
                ax.fill_between(horizons, lo, hi, alpha=0.18, color=line.get_color())

        ax.set_title(_scenario_title(scenario))
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Final Regret")
        ax.legend(loc="best")

    for j in range(len(scenarios), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Quadratic OCO: Final Regret by Horizon", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_scenario(
    scenario: Scenario,
    horizons: Array,
    *,
    trials: int,
    seed: int,
    threshold_scale: float,
    anytime_lr: bool,
) -> dict[str, dict[str, Array]]:
    ftl_runs = np.zeros((trials, horizons.size), dtype=float)
    ogd_runs = np.zeros((trials, horizons.size), dtype=float)
    smart_runs = np.zeros((trials, horizons.size), dtype=float)
    switch_runs = np.zeros((trials, horizons.size), dtype=float)

    for r in range(trials):
        for i, n in enumerate(horizons):
            rng = np.random.default_rng(seed + 10007 * r + 97 * int(n))
            mu = scenario.generator(int(n), rng)
            cfg = OCOConfig(n=int(n), seed=seed + 10007 * r + 97 * int(n), threshold_scale=threshold_scale)

            ftl = run_ftl(mu, cfg)
            ogd = run_ogd(mu, cfg, anytime_lr=anytime_lr)
            smart = run_smart(mu, cfg, anytime_lr=anytime_lr)

            ftl_runs[r, i] = float(ftl["regret"][-1])
            ogd_runs[r, i] = float(ogd["regret"][-1])
            smart_runs[r, i] = float(smart["regret"][-1])
            switch_runs[r, i] = float(smart["switch_round"])

    return {
        "ftl": _summary_curves(ftl_runs),
        "ogd": _summary_curves(ogd_runs),
        "smart": _summary_curves(smart_runs),
        "switch": _summary_curves(switch_runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run time-varying mu_t OCO experiments as final-regret-vs-horizon curves.")
    parser.add_argument("--n-max", type=int, default=1000, help="Maximum horizon")
    parser.add_argument("--n-step", type=int, default=100, help="Horizon step")
    parser.add_argument("--trials", type=int, default=30, help="Fresh sequences per horizon")
    parser.add_argument("--seed", type=int, default=0, help="Master RNG seed")
    parser.add_argument("--anytime-lr", action="store_true", help="Use eta_t=1/sqrt(t) for OGD")
    parser.add_argument("--threshold-scale", type=float, default=0.0035, help="Multiplier on SMART threshold 2*sqrt(n)")
    parser.add_argument(
        "--scenario",
        nargs="*",
        default=["stable_benign", "corruption_burst", "drift_plus_shift"],
        help="Subset of scenarios to run",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = np.arange(args.n_step, args.n_max + 1, args.n_step, dtype=int)
    generators = available_generators()

    scenarios: list[Scenario] = []
    for name in args.scenario:
        if name not in generators:
            raise ValueError(f"Unknown scenario '{name}'. Valid: {sorted(generators)}")
        scenarios.append(Scenario(name=name, generator=generators[name]))

    print("Running scenarios:")
    for sc in scenarios:
        print(f"- {sc.name}")

    stats_by_scenario: dict[str, dict[str, dict[str, Array]]] = {}
    switch_stats: dict[str, dict[str, Array]] = {}
    for sc in scenarios:
        stats = run_scenario(
            sc,
            horizons,
            trials=args.trials,
            seed=args.seed,
            threshold_scale=args.threshold_scale,
            anytime_lr=args.anytime_lr,
        )
        stats_by_scenario[sc.name] = {
            "ftl": stats["ftl"],
            "ogd": stats["ogd"],
            "smart": stats["smart"],
        }
        switch_stats[sc.name] = stats["switch"]

    _plot_regret_grid(horizons, stats_by_scenario, out_dir / "exp03_quadratic_oco_final_regret_by_horizon.png")
    print("\nFinal-regret summary at max horizon")
    for sc in scenarios:
        s = stats_by_scenario[sc.name]
        sw = switch_stats[sc.name]
        print(
            f"{sc.name:18s} "
            f"FTL={s['ftl']['mean'][-1]:.3f} "
            f"OGD={s['ogd']['mean'][-1]:.3f} "
            f"SMART={s['smart']['mean'][-1]:.3f} "
            f"switch_mean={sw['mean'][-1]:.1f}"
        )


if __name__ == "__main__":
    main()
