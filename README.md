# Panda Pick Robot Simulation Learning

Robot task planning and MuJoCo simulation tools for mobile manipulation research.

This folder is a small robosuite learning workspace built around a Panda arm in
the `Lift` task.

## Current Files

- `panda_pick.py`: creates a robosuite `Lift` environment with a Panda robot,
  prints observation keys, then runs a simple action loop.

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
python panda_pick.py --steps 500
```

For a non-rendering smoke test:

```bash
python panda_pick.py --no-render --steps 100 --print-every 20
```

To save per-step rewards and positions for later analysis:

```bash
python panda_pick.py --no-render --steps 100 --print-every 20 --log-file logs/panda_pick.csv
```

To run the simple approach policy:

```bash
python panda_pick.py --no-render --steps 200 --print-every 20 --policy approach --log-file logs/panda_approach.csv
```

To run and verify a staged pick attempt without relying on the MuJoCo viewer:

```bash
python panda_pick.py --no-render --steps 400 --print-every 50 --policy pick --log-file logs/panda_pick.csv
```

A successful pick should show reward increasing to `1.0` and `cube_z` increasing in the CSV log.

## Learning Roadmap

1. Run the existing demo and understand the robosuite environment lifecycle:
   `suite.make()`, `reset()`, `step()`, `render()`.
2. Inspect observations such as end-effector position, gripper state, cube
   position, and reward.
3. Understand the action vector:
   arm control dimensions plus the final gripper command.
4. Replace the fixed action script with staged behavior:
   move above cube, descend, close gripper, lift.
5. Add logging for observations, rewards, and success signals.
6. Try different controllers and compare behavior.
7. Add camera observations and save images or videos.
8. Use the same task for simple imitation learning or reinforcement learning
   experiments.

## First Script Behavior

`panda_pick.py` currently:

1. Creates `Lift` with robot `Panda`.
2. Prints `env.action_dim`, render mode, step count, and observation keys.
3. Runs for a configurable number of steps instead of looping forever.
4. Sends zero arm actions at first.
5. Closes the gripper after step 100.
6. Moves one action dimension between steps 200 and 300.
7. Supports `--no-render`, `--steps`, `--print-every`, `--sleep`, `--reset-on-done`, `--log-file`, and `--policy`.
8. Can save per-step reward, done flag, end-effector position, and cube position to CSV.
9. Includes three scripted policies: `smoke` for environment checks, `approach` for moving above the cube, and `pick` for a staged lift attempt.
10. Prints and logs distance diagnostics so policy behavior can be checked without relying on the MuJoCo viewer.

This is useful as a smoke test, but it is not yet a real pick-and-place policy.
