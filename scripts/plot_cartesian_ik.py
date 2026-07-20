"""Plot Cartesian IK convergence and joint updates from a debug CSV log."""

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a Cartesian IK debug log.")
    parser.add_argument("log_file", type=Path, help="CSV created by debug_cartesian_ik.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/cartesian_ik_debug.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def values(rows, column):
    return [float(row[column]) for row in rows]


def main():
    args = parse_args()
    rows = list(csv.DictReader(args.log_file.open()))
    if not rows:
        raise ValueError(f"No IK iterations found in {args.log_file}")

    import matplotlib.pyplot as plt

    iterations = values(rows, "iteration")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle("Cartesian IK Debugging")

    axes[0].plot(iterations, values(rows, "error_after"), marker="o", color="#dc2626")
    axes[0].axhline(0.001, color="#111827", linestyle="--", label="1 mm tolerance")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("error (m)")
    axes[0].set_title("Position Error")
    axes[0].legend()

    axis_colors = {"x": "#dc2626", "y": "#16a34a", "z": "#2563eb"}
    for axis_name, color in axis_colors.items():
        axes[1].plot(
            iterations,
            values(rows, f"target_{axis_name}"),
            linestyle="--",
            color=color,
            label=f"target {axis_name}",
        )
        axes[1].plot(
            iterations,
            values(rows, f"actual_{axis_name}"),
            color=color,
            label=f"actual {axis_name}",
        )
    axes[1].set_ylabel("position (m)")
    axes[1].set_title("Target vs Actual Position")
    axes[1].legend(ncol=3)

    for index in range(1, 8):
        axes[2].plot(iterations, values(rows, f"jl{index}"), label=f"jl{index}")
    axes[2].set_xlabel("IK iteration")
    axes[2].set_ylabel("joint angle (rad)")
    axes[2].set_title("Left-Arm Joint Solution")
    axes[2].legend(ncol=4)

    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
