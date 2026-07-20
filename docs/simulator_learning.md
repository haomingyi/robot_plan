# Simulator Learning Plan

This project uses MuJoCo as the main benchmark. Gazebo and Isaac Sim are used as learning comparisons.

## Current Machine

- Ubuntu: 24.04 Noble
- ROS 2: Jazzy
- GPU: NVIDIA GeForce RTX 5070
- NVIDIA driver: 580.159.03
- Isaac Sim: installed at `/home/hzm/isaacsim`
- Gazebo Sim: installed through the ROS 2 Jazzy vendor packages; the minimal server and GUI launcher are available

## Step 1: MuJoCo Baseline

Run the main project benchmark:

```bash
cd /home/hzm/yyy/wheeled-dual-arm-pick-place
source /home/hzm/anaconda3/etc/profile.d/conda.sh
conda activate mujoco
./run_dual_arm.sh
```

Generate logs and plots:

```bash
python scripts/dual_arm_pick_place.py --headless --log-file logs/dual_arm_run.csv
python scripts/plot_dual_arm_run.py logs/dual_arm_run.csv --output outputs/dual_arm_diagnostics.png
python scripts/evaluate_dual_arm.py --runs 5
```

Learn: MuJoCo XML, bodies, joints, actuators, sites, scripted control, logging, and evaluation.

## Step 2: Gazebo Minimal Test

Install Gazebo ROS integration for ROS 2 Jazzy:

```bash
sudo apt update
sudo apt install -y ros-jazzy-ros-gz
```

Refresh the shell:

```bash
source /opt/ros/jazzy/setup.bash
```

Check Gazebo:

```bash
which gz
gz sim --versions
./run_gazebo_minimal.sh
```

If the GUI opens and shows a simple two-wheel base on a ground plane, the minimal Gazebo test passed.

If you launch the project from a Snap-packaged editor terminal and see a `libpthread.so.0` / `GLIBC_PRIVATE` symbol lookup error, use `./run_gazebo_minimal.sh` instead of calling `gz sim` directly. The wrapper clears Snap-specific environment variables and adds the Gazebo vendor library paths installed by ROS 2 Jazzy.

For a no-GUI server check:

```bash
./run_gazebo_minimal.sh --server-only
```

Learn: SDF, world, model, link, joint, plugin, ROS-Gazebo integration.

## Step 3: Isaac Sim Minimal Test

Run the headless Isaac smoke test:

```bash
/home/hzm/isaacsim/python.sh sim_isaac/check_isaac_headless.py
```

Expected output:

```text
isaac_headless_ok final_cube_z=...
```

Open the Isaac Sim GUI:

```bash
/home/hzm/isaacsim/isaac-sim.sh
```

Learn: USD stage, prim, articulation, physics scene, Isaac Python API, rendering, synthetic data.

## How To Compare The Three

| Simulator | Best For | What To Test Here |
| --- | --- | --- |
| MuJoCo | Fast control and manipulation benchmarks | Main wheeled dual-arm scripted benchmark |
| Gazebo | ROS 2 integration, sensors, robot system testing | Minimal mobile base SDF world |
| Isaac Sim | High-fidelity rendering, GPU workflows, synthetic data | Headless cube physics and GUI basics |

Do not migrate the full dual-arm robot to all simulators immediately. First make each simulator pass a minimal test, then decide which features are worth porting.
