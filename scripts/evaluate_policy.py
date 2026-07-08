"""Run repeated headless Panda Lift policy evaluations."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate panda_pick.py over repeated runs.")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs to execute.")
    parser.add_argument("--steps", type=int, default=400, help="Steps per run.")
    parser.add_argument("--policy", default="pick", choices=("smoke", "approach", "pick"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/eval_pick"))
    parser.add_argument("--summary-file", type=Path, default=Path("logs/eval_pick_summary.csv"))
    parser.add_argument("--success-cube-z", type=float, default=0.90)
    return parser.parse_args()


def read_run_result(log_file, success_cube_z):
    rows = list(csv.DictReader(log_file.open()))
    if not rows:
        return {"max_reward": 0.0, "max_cube_z": 0.0, "success": False}

    max_reward = max(float(row["reward"]) for row in rows)
    max_cube_z = max(float(row["cube_z"]) for row in rows if row["cube_z"])
    return {
        "max_reward": max_reward,
        "max_cube_z": max_cube_z,
        "success": max_reward >= 1.0 or max_cube_z >= success_cube_z,
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for run_idx in range(args.runs):
        log_file = args.output_dir / f"run_{run_idx:03d}.csv"
        cmd = [
            sys.executable,
            str(repo_root / "panda_pick.py"),
            "--no-render",
            "--steps",
            str(args.steps),
            "--print-every",
            "0",
            "--policy",
            args.policy,
            "--success-cube-z",
            str(args.success_cube_z),
            "--log-file",
            str(log_file),
        ]
        print(f"Run {run_idx + 1}/{args.runs}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=repo_root)
        result = read_run_result(log_file, args.success_cube_z)
        result["run"] = run_idx
        result["log_file"] = str(log_file)
        results.append(result)
        print(
            "  success={success} max_reward={max_reward:.3f} max_cube_z={max_cube_z:.3f}".format(
                **result
            ),
            flush=True,
        )

    with args.summary_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "success", "max_reward", "max_cube_z", "log_file"])
        writer.writeheader()
        writer.writerows(results)

    successes = sum(1 for result in results if result["success"])
    print("\nEvaluation summary")
    print(f"Runs: {args.runs}")
    print(f"Successes: {successes}")
    print(f"Success rate: {successes / args.runs:.2%}")
    print(f"Summary CSV: {args.summary_file}")


if __name__ == "__main__":
    main()
