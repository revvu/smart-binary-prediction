from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from Scripts.algorithms import (  # noqa: E402
    Cover_binary,
    SMART_det_binary,
    SMART_random_binary,
    adahedge,
    calc_regret,
)


TargetFn = Callable[[int, int], float]


def _loss_based_sequence_generation(n: int, target_loss_function: TargetFn) -> np.ndarray:
    loss_0 = 0
    loss_1 = 0
    sequence = []
    for t in range(n):
        diff = target_loss_function(t, n)
        next_bit = 0 if abs(diff - (loss_0 - loss_1 - 1)) <= abs(diff - (loss_0 - loss_1 + 1)) else 1
        sequence.append(next_bit)
        loss_1 += next_bit == 0
        loss_0 += next_bit == 1
    return np.asarray(sequence)


def _loss_difference(sequence: np.ndarray) -> np.ndarray:
    walk = [0]
    for bit in sequence:
        walk.append(walk[-1] + (1 if bit == 1 else -1))
    return np.asarray(walk, dtype=float)


def _regret_curves(n_values: list[int], target: TargetFn) -> dict[str, np.ndarray]:
    algorithms = {
        r"Reg($SMART$)": SMART_det_binary,
        r"Reg($Cover$)": Cover_binary,
        r"Reg($RandomizedSMART$)": SMART_random_binary,
        r"Reg($AdaHedge$)": adahedge,
    }
    curves = {name: np.zeros(len(n_values), dtype=float) for name in algorithms}

    for idx, n in enumerate(n_values):
        seq = _loss_based_sequence_generation(n, target)
        for name, algorithm in algorithms.items():
            curves[name][idx] = calc_regret(seq, algorithm(seq))

    return curves


def make_figure() -> None:
    target = lambda t, n: float(t**0.4)
    horizon = 1000
    n_values = list(range(10, horizon, 50))

    full_sequence = _loss_based_sequence_generation(horizon, target)
    rounds = np.arange(horizon + 1)
    true_loss_diff = _loss_difference(full_sequence)
    target_loss_diff = np.array([target(int(t), horizon) for t in rounds], dtype=float)
    regret_curves = _regret_curves(n_values, target)

    out_dir = EXP_DIR / "Graphs" / "Paper Figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "Loss t0.4 side by side paper figure.png"
    pdf_path = out_dir / "Loss t0.4 side by side paper figure.pdf"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)

    ax = axes[0]
    ax.step(rounds, true_loss_diff, where="post", color="#7db5e8", linewidth=2.0, label="True Loss Difference")
    ax.plot(rounds, target_loss_diff, color="#ff7f0e", linestyle="--", linewidth=2.0, label="Target Loss Difference")
    ax.set_title(r"Generating Loss $t^{0.4}$ Sequences", color="black", fontsize=18)
    ax.set_xlabel(r"Length of sequence ($t$)", color="black", fontsize=16)
    ax.set_ylabel("Difference in Loss Accrued between Experts", color="black", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.legend(loc="upper left", fontsize=12, frameon=True)

    ax = axes[1]
    colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
    for (label, values), color in zip(regret_curves.items(), colors):
        ax.plot(n_values, values, ".-", label=label, linewidth=1.4, markersize=4.0, color=color)
    ax.set_title(r"Binary prediction on the Loss $n^{0.4}$ sequence", fontsize=18)
    ax.set_xlabel("Length of sequence (n)", fontsize=16)
    ax.set_ylabel("Regret", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.legend(loc="upper left", fontsize=14, frameon=True)

    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    make_figure()
