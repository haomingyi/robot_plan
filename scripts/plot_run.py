"""Plot a single manipulation benchmark CSV log."""

import argparse
import csv
from pathlib import Path


PHASE_COLORS = {
    "smoke": "#9ca3af",
    "approach": "#60a5fa",
    "hover": "#38bdf8",
    "descend": "#f59e0b",
    "grasp": "#ef4444",
    "lift": "#22c55e",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a manipulation benchmark run log.")
    parser.add_argument("log_file", type=Path, help="CSV file created by manipulation_benchmark.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to the input path with a .png suffix.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title. Defaults to the input file name.",
    )
    return parser.parse_args()


def read_rows(log_file):
    rows = list(csv.DictReader(log_file.open()))
    if not rows:
        raise ValueError(f"No rows found in {log_file}")
    return rows


def float_column(rows, name):
    values = []
    for row in rows:
        value = row.get(name, "")
        values.append(float(value) if value not in ("", None) else None)
    return values


def phase_spans(steps, phases):
    if not steps:
        return []

    spans = []
    start_step = steps[0]
    current_phase = phases[0]
    previous_step = steps[0]
    for step, phase in zip(steps[1:], phases[1:]):
        if phase != current_phase:
            spans.append((start_step, previous_step, current_phase))
            start_step = step
            current_phase = phase
        previous_step = step
    spans.append((start_step, previous_step, current_phase))
    return spans


def add_phase_background(axis, steps, phases):
    for start, end, phase in phase_spans(steps, phases):
        axis.axvspan(
            start,
            end,
            color=PHASE_COLORS.get(phase, "#e5e7eb"),
            alpha=0.12,
            linewidth=0,
        )


def plot_run(rows, output_file, title):
    import matplotlib.pyplot as plt

    steps = [int(row["step"]) for row in rows]
    phases = [row.get("phase", "") for row in rows]
    reward = float_column(rows, "reward")
    cube_z = float_column(rows, "cube_z")
    distance = float_column(rows, "eef_cube_distance")
    xy_distance = float_column(rows, "eef_cube_xy_distance")
    z_gap = float_column(rows, "cube_minus_eef_z")

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(title)

    axes[0].plot(steps, reward, color="#2563eb", label="reward")
    axes[0].set_ylabel("reward")
    axes[0].set_ylim(bottom=-0.05)

    axes[1].plot(steps, cube_z, color="#16a34a", label="cube_z")
    axes[1].axhline(0.90, color="#dc2626", linestyle="--", linewidth=1, label="success z")
    axes[1].set_ylabel("cube z")

    axes[2].plot(steps, distance, color="#7c3aed", label="3D distance")
    axes[2].plot(steps, xy_distance, color="#ea580c", label="XY distance")
    axes[2].set_ylabel("distance")

    axes[3].plot(steps, z_gap, color="#0891b2", label="cube - eef z")
    axes[3].axhline(0.0, color="#111827", linestyle="--", linewidth=1)
    axes[3].set_ylabel("z gap")
    axes[3].set_xlabel("step")

    for axis in axes:
        add_phase_background(axis, steps, phases)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=160)
    print(f"Saved plot to {output_file}")


def main():
    args = parse_args()
    output_file = args.output or args.log_file.with_suffix(".png")
    title = args.title or args.log_file.name
    rows = read_rows(args.log_file)
    plot_run(rows, output_file, title)


if __name__ == "__main__":
    main()
