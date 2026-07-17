"""Plot diagnostics for a wheeled dual-arm CSV run log."""

import argparse
import csv
from pathlib import Path


STAGE_COLORS = {
    "drive_forward": "#60a5fa",
    "approach_objects": "#f59e0b",
    "close_grippers": "#ef4444",
    "lift_objects": "#22c55e",
    "align_over_tray": "#8b5cf6",
    "lower_into_tray": "#14b8a6",
    "release_workpiece": "#64748b",
    "left_retreat": "#94a3b8",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a wheeled dual-arm CSV run log.")
    parser.add_argument("log_file", type=Path, help="CSV file created by scripts/dual_arm_pick_place.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to the input path with a .png suffix.",
    )
    parser.add_argument("--title", default=None, help="Optional figure title. Defaults to the input file name.")
    return parser.parse_args()


def read_rows(log_file):
    rows = list(csv.DictReader(log_file.open()))
    if not rows:
        raise ValueError(f"No rows found in {log_file}")
    return rows


def float_column(rows, name):
    return [float(row[name]) for row in rows]


def stage_spans(times, stages):
    spans = []
    start_time = times[0]
    current_stage = stages[0]
    previous_time = times[0]
    for time_value, stage in zip(times[1:], stages[1:]):
        if stage != current_stage:
            spans.append((start_time, previous_time, current_stage))
            start_time = time_value
            current_stage = stage
        previous_time = time_value
    spans.append((start_time, previous_time, current_stage))
    return spans


def add_stage_background(axis, times, stages):
    for start, end, stage in stage_spans(times, stages):
        axis.axvspan(start, end, color=STAGE_COLORS.get(stage, "#e5e7eb"), alpha=0.12, linewidth=0)


def plot_run(rows, output_file, title):
    import matplotlib.pyplot as plt

    times = float_column(rows, "time")
    stages = [row["stage"] for row in rows]
    placement_error = float_column(rows, "placement_error")
    workpiece_z = float_column(rows, "workpiece_z")
    base_y = float_column(rows, "base_y")
    left_y = float_column(rows, "left_gripper_y")
    right_y = float_column(rows, "right_gripper_y")
    workpiece_y = float_column(rows, "workpiece_y")
    target_y = float_column(rows, "target_y")

    left_x = float_column(rows, "left_gripper_x")
    right_x = float_column(rows, "right_gripper_x")
    workpiece_x = float_column(rows, "workpiece_x")
    target_x = float_column(rows, "target_x")

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=False)
    fig.suptitle(title)

    axes[0].plot(times, placement_error, color="#dc2626", label="placement error")
    axes[0].axhline(0.03, color="#111827", linestyle="--", linewidth=1, label="success threshold")
    axes[0].set_ylabel("meters")
    axes[0].set_title("Placement Error")

    axes[1].plot(times, workpiece_z, color="#16a34a", label="workpiece z")
    axes[1].set_ylabel("meters")
    axes[1].set_title("Workpiece Height")

    axes[2].plot(times, base_y, color="#2563eb", label="base y")
    axes[2].plot(times, left_y, color="#f97316", label="left gripper y")
    axes[2].plot(times, right_y, color="#7c3aed", label="right gripper y")
    axes[2].plot(times, workpiece_y, color="#059669", label="workpiece y")
    axes[2].plot(times, target_y, color="#111827", linestyle="--", linewidth=1, label="target y")
    axes[2].set_ylabel("meters")
    axes[2].set_title("Forward Motion")

    axes[3].plot(left_x, left_y, color="#f97316", label="left gripper")
    axes[3].plot(right_x, right_y, color="#7c3aed", label="right gripper")
    axes[3].plot(workpiece_x, workpiece_y, color="#059669", label="workpiece")
    axes[3].scatter(target_x[-1], target_y[-1], color="#dc2626", marker="x", s=70, label="final target")
    axes[3].set_xlabel("x position")
    axes[3].set_ylabel("y position")
    axes[3].set_title("Top-Down Trajectory")
    axes[3].axis("equal")

    for axis in axes[:3]:
        add_stage_background(axis, times, stages)
        axis.set_xlabel("time (s)")

    for axis in axes:
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
