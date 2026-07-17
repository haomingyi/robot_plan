# Contact-Rich Manipulation Benchmark

MuJoCo / robosuite tools for debugging and evaluating contact-rich robotic
manipulation policies.

The current benchmark uses robosuite's `Lift` task with a Franka Panda arm as a
controlled manipulation platform. The public project focus is policy debugging,
contact visualization, logging, and repeatable evaluation rather than the robot
brand itself.

## Current Files

- `manipulation_benchmark.py`: creates a robosuite `Lift` environment, then runs
  selectable scripted policies with MuJoCo viewer support, diagnostics, and
  optional CSV logs.
- `scripts/evaluate_policy.py`: runs repeated headless policy checks and writes a
  summary CSV with success rate, max reward, and max cube height.
- `scripts/plot_run.py`: turns a per-step CSV log into a diagnostic plot for
  reward, object height, end-effector distance, and policy phases.
- `run_viewer.sh`: starts the official MuJoCo viewer with the benchmark policy
  and keeps the window open for inspection or reruns after the first demo.

## Capabilities

- Scripted manipulation baselines with explicit policy phases.
- Official MuJoCo viewer integration for pause, stepping, contact inspection,
  rendering debug overlays, and repeated demos from the same viewer window.
- Per-step CSV logging for reward, end-effector pose, object pose, distance, and
  policy phase.
- Plot generation for single-run logs, including phase backgrounds, reward,
  object height, and distance diagnostics.
- Repeated headless evaluation for success-rate tracking.
- Tunable viewer diagnostics such as coordinate-frame length and width.

## Environment Status

Checked on 2026-07-08:

- System `python3` is available, but does not have the simulation dependencies.
- Conda environment `mujoco` has `robosuite` and can run a short no-render
  smoke test.
- `libGL.so.1` was installed into the `mujoco` conda environment with
  `conda install -n mujoco -c conda-forge libgl -y`.

Run the demo with:

```bash
source /home/hzm/anaconda3/etc/profile.d/conda.sh
conda activate mujoco
python manipulation_benchmark.py --steps 500
```

For a non-rendering smoke test:

```bash
python manipulation_benchmark.py --no-render --steps 100 --print-every 20
```

To save per-step rewards and positions for later analysis:

```bash
python manipulation_benchmark.py --no-render --steps 100 --print-every 20 --log-file logs/pick.csv
```

To run the simple approach policy:

```bash
python manipulation_benchmark.py --no-render --steps 200 --print-every 20 --policy approach --log-file logs/approach.csv
```

To run and verify a staged pick attempt without relying on the MuJoCo viewer:

```bash
python manipulation_benchmark.py --no-render --steps 400 --print-every 50 --policy pick --log-file logs/pick.csv
```

A successful pick should show reward increasing to `1.0`, `cube_z` increasing in the CSV log, and a final `Success: True` summary. The policy phase should progress through `hover`, `descend`, `grasp`, and `lift`.

To run repeated headless evaluations:

```bash
python scripts/evaluate_policy.py --runs 5 --steps 400 --policy pick
```

This writes per-run logs under `logs/eval_pick/` and a summary CSV at `logs/eval_pick_summary.csv`.

To visualize a single run log:

```bash
python scripts/plot_run.py logs/pick.csv --output outputs/pick_diagnostics.png
```

## Roadmap

1. Run the existing demo and understand the robosuite environment lifecycle:
   `suite.make()`, `reset()`, `step()`, `render()`.
2. Inspect observations such as end-effector position, gripper state, cube
   position, and reward.
3. Understand the action vector:
   arm control dimensions plus the final gripper command.
4. Replace the fixed action script with staged behavior:
   move above cube, descend, close gripper, lift.
5. Add logging for observations, rewards, and success signals.
6. Add plots and summary artifacts for experiment review.
7. Try different controllers and compare behavior.
8. Add camera observations and save images or videos.
9. Use the same task for imitation learning or reinforcement learning
   experiments.

## Script Behavior

`manipulation_benchmark.py` currently:

1. Creates `Lift` with robot `Panda`.
2. Prints `env.action_dim`, render mode, step count, and observation keys.
3. Runs for a configurable number of steps or keeps the MuJoCo viewer open for
   post-run inspection.
4. Supports three scripted policies: `smoke` for environment checks, `approach`
   for moving above the cube, and `pick` for a staged lift attempt.
5. Exposes the staged policy phases: `hover`, `descend`, `grasp`, and `lift`.
6. Supports `--no-render`, `--viewer`, `--pause-at-start`, `--keep-viewer-open`,
   `--debug`, `--log-file`, and success-threshold options.
7. Can save per-step reward, done flag, end-effector position, cube position,
   distance metrics, and policy phase to CSV.
8. Reports run-level success using max reward and max cube height.
9. Supports repeated headless evaluation for basic success-rate tracking.

This is a debuggable benchmark baseline, not yet a general pick-and-place
planner.
