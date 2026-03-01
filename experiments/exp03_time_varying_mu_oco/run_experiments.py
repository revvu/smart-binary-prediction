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
    trials: int


def _summary_curves(regrets: Array) -> tuple[Array, Array, Array]:
    mean = regrets.mean(axis=0)
    if regrets.shape[0] <= 1:
        return mean, mean, mean
    se = regrets.std(axis=0, ddof=1) / np.sqrt(regrets.shape[0])
    lo = mean - 1.96 * se
    hi = mean + 1.96 * se
    return mean, lo, hi


def _plot_mu(mu: Array, out_path: Path, title: str) -> None:
    t = np.arange(1, mu.size + 1)
    plt.figure(figsize=(8.0, 4.5))
    plt.plot(t, mu, linewidth=2)
    plt.xlabel("t")
    plt.ylabel(r"$\mu_t$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_regret(
    t: Array,
    stats_ftl: tuple[Array, Array, Array],
    stats_ogd: tuple[Array, Array, Array],
    stats_smart: tuple[Array, Array, Array],
    out_path: Path,
    title: str,
    ogd_label: str,
) -> None:
    m_ftl, lo_ftl, hi_ftl = stats_ftl
    m_ogd, lo_ogd, hi_ogd = stats_ogd
    m_smart, lo_smart, hi_smart = stats_smart

    plt.figure(figsize=(8.0, 5.0))
    plt.plot(t, m_ftl, label="FTL", linewidth=2)
    plt.plot(t, m_ogd, label=ogd_label, linewidth=2)
    plt.plot(t, m_smart, label="SMART", linewidth=2)

    if np.any(hi_ftl > lo_ftl):
        plt.fill_between(t, lo_ftl, hi_ftl, alpha=0.2)
        plt.fill_between(t, lo_ogd, hi_ogd, alpha=0.2)
        plt.fill_between(t, lo_smart, hi_smart, alpha=0.2)

    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_diagnostics(
    t: Array,
    stats_ftl: tuple[Array, Array, Array],
    stats_ogd: tuple[Array, Array, Array],
    stats_smart: tuple[Array, Array, Array],
    sigma: Array,
    threshold: float,
    switch_round: float,
    out_path: Path,
    title: str,
    ogd_label: str,
) -> None:
    m_ftl, _, _ = stats_ftl
    m_ogd, _, _ = stats_ogd
    m_smart, _, _ = stats_smart

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True)

    axes[0].plot(t, m_ftl, label="FTL", linewidth=2)
    axes[0].plot(t, m_ogd, label=ogd_label, linewidth=2)
    axes[0].plot(t, m_smart, label="SMART", linewidth=2)
    axes[0].set_ylabel("Regret")
    axes[0].set_title(title)
    axes[0].legend(loc="best")

    axes[1].plot(t, sigma[1:], linewidth=2, label=r"$\Sigma_t$")
    axes[1].axhline(threshold, linestyle="--", linewidth=1.5, label="threshold")
    if 1 <= switch_round <= t.size:
        axes[1].axvline(switch_round, linestyle=":", linewidth=1.5, label=f"switch={int(switch_round)}")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Switch statistic")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_scenario(cfg: OCOConfig, scenario: Scenario, out_dir: Path, anytime_lr: bool) -> dict[str, float]:
    rng_master = np.random.default_rng(cfg.seed)

    regrets_ftl = np.zeros((scenario.trials, cfg.n), dtype=float)
    regrets_ogd = np.zeros((scenario.trials, cfg.n), dtype=float)
    regrets_smart = np.zeros((scenario.trials, cfg.n), dtype=float)
    switches = np.zeros(scenario.trials, dtype=float)
    sigma_last: Array | None = None
    threshold_last = 0.0

    first_mu: Array | None = None
    for k in range(scenario.trials):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        mu = scenario.generator(cfg.n, rng)
        if first_mu is None:
            first_mu = mu

        ftl = run_ftl(mu, cfg)
        ogd = run_ogd(mu, cfg, anytime_lr=anytime_lr)
        smart = run_smart(mu, cfg, anytime_lr=anytime_lr)

        regrets_ftl[k, :] = ftl["regret"][1:]
        regrets_ogd[k, :] = ogd["regret"][1:]
        regrets_smart[k, :] = smart["regret"][1:]
        switches[k] = float(smart["switch_round"])
        sigma_last = np.asarray(smart["sigma_eq6"], dtype=float)
        threshold_last = float(smart["threshold"])

    assert first_mu is not None
    assert sigma_last is not None

    t = np.arange(1, cfg.n + 1, dtype=float)
    stats_ftl = _summary_curves(regrets_ftl)
    stats_ogd = _summary_curves(regrets_ogd)
    stats_smart = _summary_curves(regrets_smart)

    _plot_mu(first_mu, out_dir / f"{scenario.name}_mu.png", rf"$\mu_t$ pattern: {scenario.name}")
    ogd_label = r"OGD ($\eta_t=1/\sqrt{t}$)" if anytime_lr else r"OGD ($\eta=2/\sqrt{n}$)"
    _plot_regret(
        t,
        stats_ftl,
        stats_ogd,
        stats_smart,
        out_dir / f"{scenario.name}_regret.png",
        f"Regret comparison: {scenario.name}",
        ogd_label,
    )
    _plot_diagnostics(
        t,
        stats_ftl,
        stats_ogd,
        stats_smart,
        sigma_last,
        threshold_last,
        float(np.mean(switches)),
        out_dir / f"{scenario.name}_diagnostics.png",
        f"Diagnostics: {scenario.name}",
        ogd_label,
    )

    return {
        "ftl_final": float(stats_ftl[0][-1]),
        "ogd_final": float(stats_ogd[0][-1]),
        "smart_final": float(stats_smart[0][-1]),
        "smart_switch_mean": float(np.mean(switches)),
        "sigma_max": float(np.max(sigma_last)),
        "threshold": threshold_last,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SMART OCO experiments with time-varying mu_t.")
    parser.add_argument("--n", type=int, default=1000, help="Horizon length")
    parser.add_argument("--seed", type=int, default=0, help="Master RNG seed")
    parser.add_argument("--trials", type=int, default=50, help="Trials for stochastic scenarios")
    parser.add_argument("--anytime-lr", action="store_true", help="Use eta_t=1/sqrt(t) for OGD branch")
    parser.add_argument(
        "--threshold-scale",
        type=float,
        default=0.0035,
        help="Multiplier on SMART threshold 2*sqrt(n).",
    )
    parser.add_argument(
        "--scenario",
        nargs="*",
        default=["stable_benign", "corruption_burst", "drift_plus_shift"],
        help="Subset of scenarios to run",
    )
    args = parser.parse_args()

    cfg = OCOConfig(n=args.n, seed=args.seed, threshold_scale=args.threshold_scale)
    out_dir = Path(__file__).resolve().parent / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    generators = available_generators()
    scenarios: list[Scenario] = []
    for name in args.scenario:
        if name not in generators:
            raise ValueError(f"Unknown scenario '{name}'. Valid: {sorted(generators)}")
        trials = args.trials
        scenarios.append(Scenario(name=name, generator=generators[name], trials=trials))

    print("Running scenarios:")
    for sc in scenarios:
        print(f"- {sc.name} (trials={sc.trials})")

    results = {}
    for sc in scenarios:
        results[sc.name] = run_scenario(cfg, sc, out_dir, anytime_lr=args.anytime_lr)

    print("\nFinal-time regret summary")
    for name, r in results.items():
        print(
            f"{name:22s} "
            f"FTL={r['ftl_final']:.3f} "
            f"OGD={r['ogd_final']:.3f} "
            f"SMART={r['smart_final']:.3f} "
            f"switch_mean={r['smart_switch_mean']:.2f} "
            f"sigma_max={r['sigma_max']:.3f} "
            f"threshold={r['threshold']:.3f}"
        )


if __name__ == "__main__":
    main()
