# Wheeled Dual-Arm Pick-and-Place

This project is a MuJoCo simulation benchmark for wheeled dual-arm mobile manipulation. It is designed to be practical, debuggable, and suitable for portfolio or resume presentation.

## Project Status

The current version is a stable learning baseline. The main MuJoCo task, repeated evaluation, contact diagnostics, Cartesian IK debugger, Gazebo smoke test, and Isaac Sim smoke test have been validated on the target workstation. Future work can build on this baseline without changing its documented run commands.

## Final Effect

The current demo shows a complete wheeled dual-arm coordination sequence in the official MuJoCo Viewer:

- The mobile base drives toward the table.
- The left arm grasps the workpiece.
- The right arm approaches from the tray's right side and grips a dedicated side tab.
- The workpiece lifts first, the tray lifts second, and the left arm inserts the workpiece into a tray pocket.

The goal is not only to play an animation, but to grow this into a debuggable, measurable, and extensible mobile manipulation benchmark.

## Project Structure

- `models/wheeled_dual_arm/`: wheeled dual-arm robot model, table, tray, workpiece, and required STL meshes.
- `scripts/dual_arm_pick_place.py`: the main demo script with staged control for the base, dual arms, grippers, tray, and workpiece.
- `scripts/evaluate_dual_arm.py`: repeated headless evaluation for the wheeled dual-arm task.
- `scripts/plot_dual_arm_run.py`: diagnostic plotting for dual-arm CSV logs.
- `scripts/debug_cartesian_ik.py`: interactive Cartesian IK debugger with target and measured-position markers.
- `scripts/ik_utils.py`: damped least-squares position IK utilities.
- `scripts/plot_cartesian_ik.py`: IK convergence and joint-solution plotting.
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

Save per-step diagnostics to CSV:

```bash
python scripts/dual_arm_pick_place.py --headless --log-file logs/dual_arm_run.csv
```

Plot a dual-arm run:

```bash
python scripts/plot_dual_arm_run.py logs/dual_arm_run.csv --output outputs/dual_arm_diagnostics.png
```

Run repeated headless evaluations:

```bash
python scripts/evaluate_dual_arm.py --runs 5
```

Change playback speed:

```bash
./run_dual_arm.sh --realtime 0.5
./run_dual_arm.sh --realtime 2.0
```

The viewer loops the full sequence and does not close automatically. Pause the Viewer when you want to inspect the scene, and close the Viewer window when you are done.

## Contact And Physics Checks

The table, tray structure, workpiece, gripper fingers, wheels, and mobile-base shell have collision geometry. The tray has four physical pockets, a supporting bottom, and a dedicated grip tab on its right side. The workpiece is released into the target pocket under gravity; the left gripper clears upward before retreating.

Enable `Contact Point` and `Contact Force` in the Viewer to inspect contacts. The main CSV log includes `contact_count`, `workpiece_tray_contacts`, and `base_table_contacts` for headless verification.

Green sites named `left_grip_center` and `right_grip_center` are end-effector reference points used for IK and logging. Red sites mark finger tips and the workpiece reference point. Sites are debug markers, not collision geometry.

The current benchmark still uses scripted attachment while carrying the workpiece and tray. Finger motion and release contacts are simulated, but carrying is not yet a friction-only grasp.

## Debug Cartesian IK

Open the interactive left-arm IK debugger:

```bash
python scripts/debug_cartesian_ik.py
```

The red sphere is the Cartesian target, the green sphere is the measured gripper position, and the yellow line is the current position error. Use `A/D` for world X, `S/W` for world Y, and `Q/E` for world Z. Press `N` to execute exactly one IK iteration, `G` to solve automatically, and `R` to reset the experiment.

Run the same 5 cm Z-axis target without a window:

```bash
python scripts/debug_cartesian_ik.py --headless --axis z --distance 0.05
```

Each IK iteration is written to `logs/cartesian_ik_debug.csv`, including target position, measured position, error, joint update, Jacobian minimum singular value, and all seven left-arm joint positions.

Plot the convergence after closing the Viewer or running headless:

```bash
python scripts/plot_cartesian_ik.py logs/cartesian_ik_debug.csv --output outputs/cartesian_ik_debug.png
```

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

The dual-arm evaluation reports success, final placement error, maximum workpiece height, final base error, and the per-run CSV log path. The plotting script visualizes placement error, workpiece height, forward motion, and top-down trajectories.

The legacy single-arm baseline is still available:

```bash
./run_single_arm_baseline.sh
python manipulation_benchmark.py --no-render --steps 400 --policy pick --log-file logs/pick.csv
python scripts/evaluate_policy.py --runs 5 --steps 400 --policy pick
python scripts/plot_run.py logs/pick.csv --output outputs/pick_diagnostics.png
```

## Roadmap

1. Document the wheeled dual-arm model, joint table, actuator table, and coordinate frames.
2. Replace scripted object attachment with more realistic contact-based grasping and constraints.
3. Add cameras, sensors, and dataset export for imitation learning or reinforcement learning.
4. Prepare a demo GIF, metrics summary, and resume-ready project description.
