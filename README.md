# Panda Pick Robot Simulation Learning

This folder is a small robosuite learning workspace built around a Panda arm in
the `Lift` task.

## Current Files

- `panda_pick.py`: creates a robosuite `Lift` environment with a Panda robot,
  prints observation keys, then runs a simple action loop.

## Environment Status

Checked on 2026-07-08:

- System `python3` is available, but does not have `robosuite` installed.
- Conda environment `mujoco` has `robosuite`, but importing it currently fails
  because `libGL.so.1` is missing.

Typical fix on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y libgl1
```

After that, run:

```bash
source /home/hzm/anaconda3/etc/profile.d/conda.sh
conda activate mujoco
python panda_pick.py
```

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
2. Prints `env.action_dim` and observation keys.
3. Sends zero actions at first.
4. Closes the gripper after step 100.
5. Moves one action dimension between steps 200 and 300.
6. Renders continuously.

This is useful as a smoke test, but it is not yet a real pick-and-place policy.
