# Wheeled Dual-Arm Pick-and-Place

This project is a MuJoCo simulation benchmark for wheeled dual-arm mobile manipulation. It is designed to be practical, debuggable, and suitable for portfolio or resume presentation.

## Final Effect

The current demo shows a complete wheeled dual-arm coordination sequence in the official MuJoCo Viewer:

- The mobile base drives toward the table.
- The left arm grasps the workpiece.
- The right arm grasps the tray.
- Both arms lift together and place the workpiece into the target position on the tray.

The goal is not only to play an animation, but to grow this into a debuggable, measurable, and extensible mobile manipulation benchmark.

## Project Structure

- `models/wheeled_dual_arm/`: wheeled dual-arm robot model, table, tray, workpiece, and required STL meshes.
- `scripts/dual_arm_pick_place.py`: the main demo script with staged control for the base, dual arms, grippers, tray, and workpiece.
- `run_dual_arm.sh`: one-command launcher for the MuJoCo Viewer.
- `manipulation_benchmark.py`: legacy single-arm robosuite baseline kept for comparison and learning.
- `scripts/evaluate_policy.py`: repeated evaluation script for the legacy baseline.
- `scripts/plot_run.py`: diagnostic plot generator for legacy baseline CSV logs.

## Environment

Activate the MuJoCo conda environment before running the project:

```bash
cd /path/to/wheeled-dual-arm-pick-place
source /home/hzm/anaconda3/etc/profile.d/conda.sh
conda activate mujoco
```

The `source /home/hzm/anaconda3/etc/profile.d/conda.sh` command loads conda's shell integration into the current terminal. After that, `conda activate mujoco` can switch this terminal into the MuJoCo environment.

## Run The Main Demo

Open the official MuJoCo Viewer:

```bash
./run_dual_arm.sh
```

Run a headless check without opening a window:

```bash
python scripts/dual_arm_pick_place.py --headless
```

Change playback speed:

```bash
./run_dual_arm.sh --realtime 0.5
./run_dual_arm.sh --realtime 2.0
```

The viewer loops the full sequence and does not close automatically. Pause the Viewer when you want to inspect the scene, and close the Viewer window when you are done.

## Learning Path

1. Run `./run_dual_arm.sh` and observe the full sequence: base motion, dual-arm approach, gripper closing, lifting, alignment, and release.
2. Open `scripts/dual_arm_pick_place.py` and inspect `STAGES`. Each row defines one phase of the task, including base position, left and right arm joint targets, gripper commands, and object attachment state.
3. Read `sample_stage()` to understand how `smoothstep()` interpolates between stages.
4. Read `set_base_pose()` to understand how the mobile base pose and wheel angles are set.
5. Read `set_joint_pose()` to understand the actuator commands for the two 7-DoF arms and both grippers.
6. Read `set_workpiece_pose()` and `set_tray_pose()` to understand how the current scripted demo stabilizes the object and tray motion.
7. Replace scripted object attachment with more realistic contact, grasping, sensing, and controller logic.

## MuJoCo Viewer Usage

The left panel is mainly for simulation and visualization controls, such as pause, step, speed, rendering modes, contact points, frames, and cameras.

The right panel is mainly for model inspection and parameter tuning, such as body, joint, actuator, sensor, camera, and option settings. It is especially useful for checking joint names, actuator names, coordinate frames, and contact configuration.

If coordinate-frame cylinders or site markers look too large, adjust the related frame or site visualization options in the Viewer, or reduce the corresponding `site` `size` values in the XML model.

## Current Baseline

The main dual-arm demo currently uses a staged scripted controller. It is useful for learning model structure, Viewer debugging, trajectory flow, and task definition. It is not yet a full physics-based grasp planner.

The legacy single-arm baseline is still available:

```bash
./run_single_arm_baseline.sh
python manipulation_benchmark.py --no-render --steps 400 --policy pick --log-file logs/pick.csv
python scripts/evaluate_policy.py --runs 5 --steps 400 --policy pick
python scripts/plot_run.py logs/pick.csv --output outputs/pick_diagnostics.png
```

## Roadmap

1. Document the wheeled dual-arm model, joint table, actuator table, and coordinate frames.
2. Add CSV logging for the dual-arm demo: stage, base pose, gripper poses, tray pose, and workpiece pose.
3. Add a dual-arm evaluation script for task completion, grasp state, placement accuracy, and final error.
4. Add diagnostic plots for base trajectory, gripper trajectory, workpiece height, and stage timeline.
5. Replace scripted object attachment with more realistic contact-based grasping and constraints.
6. Add cameras, sensors, and dataset export for imitation learning or reinforcement learning.
7. Prepare a demo GIF, metrics summary, and resume-ready project description.
