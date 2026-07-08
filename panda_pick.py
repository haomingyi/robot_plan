"""Run a small robosuite Panda Lift demo.

This script is intentionally simple: it is a smoke test for the environment and
an entry point for learning how actions, observations, rewards, and rendering
fit together in robosuite.
"""

import argparse
import csv
import time
from pathlib import Path


GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description="Run a robosuite Panda Lift demo.")
    parser.add_argument("--steps", type=int, default=500, help="Number of control steps to run.")
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print observation and reward information every N steps.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.02,
        help="Seconds to sleep after each step when rendering.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run without the interactive robosuite renderer.",
    )
    parser.add_argument(
        "--reset-on-done",
        action="store_true",
        help="Reset and continue if the environment reports done before --steps.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional CSV file path for per-step reward and position logging.",
    )
    parser.add_argument(
        "--policy",
        choices=("smoke", "approach", "pick"),
        default="smoke",
        help="Scripted policy to run: smoke checks the environment; approach moves above the cube; pick approaches, descends, grasps, and lifts.",
    )
    return parser.parse_args()


def make_env(has_renderer):
    import robosuite as suite

    return suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=has_renderer,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )


def zero_action(action_dim):
    import numpy as np

    return np.zeros(action_dim)


def smoke_policy(action_dim, step, obs):
    """Return the original simple hand-written smoke-test action."""
    action = zero_action(action_dim)

    if step > 100:
        action[-1] = GRIPPER_CLOSE

    if 200 < step < 300:
        action[0] = 0.2

    return action


def relative_cube_offset(obs):
    offset = obs.get("gripper_to_cube_pos")
    if offset is None or len(offset) < 3:
        return None

    import numpy as np

    return np.asarray(offset, dtype=float)


def move_to_relative_offset(action_dim, offset, desired_cube_below_gripper):
    import numpy as np

    action = zero_action(action_dim)
    error = offset - np.asarray(desired_cube_below_gripper, dtype=float)
    action[:3] = np.clip(error * 2.0, -0.25, 0.25)
    return action


def approach_policy(action_dim, step, obs):
    """Move the end effector to a hover pose above the cube."""
    offset = relative_cube_offset(obs)
    if offset is None:
        return smoke_policy(action_dim, step, obs)

    action = move_to_relative_offset(action_dim, offset, desired_cube_below_gripper=[0.0, 0.0, -0.05])
    action[-1] = GRIPPER_OPEN
    return action


def pick_policy(action_dim, step, obs):
    """Simple staged pick attempt: hover, descend, close gripper, then lift."""
    import numpy as np

    offset = relative_cube_offset(obs)
    if offset is None:
        return smoke_policy(action_dim, step, obs)

    xy_error = np.linalg.norm(offset[:2])
    z_gap = offset[2]

    if xy_error > 0.025 or z_gap < -0.065:
        action = move_to_relative_offset(action_dim, offset, desired_cube_below_gripper=[0.0, 0.0, -0.05])
        action[-1] = GRIPPER_OPEN
        return action

    if step < 150:
        action = move_to_relative_offset(action_dim, offset, desired_cube_below_gripper=[0.0, 0.0, 0.0])
        action[-1] = GRIPPER_OPEN
        return action

    if step < 220:
        action = zero_action(action_dim)
        action[-1] = GRIPPER_CLOSE
        return action

    action = zero_action(action_dim)
    action[2] = 0.25
    action[-1] = GRIPPER_CLOSE
    return action


def scripted_action(action_dim, step, obs, policy):
    if policy == "approach":
        return approach_policy(action_dim, step, obs)
    if policy == "pick":
        return pick_policy(action_dim, step, obs)
    return smoke_policy(action_dim, step, obs)


def obs_metrics(obs):
    import math

    eef_x, eef_y, eef_z = vector_components(obs, "robot0_eef_pos")
    cube_x, cube_y, cube_z = vector_components(obs, "cube_pos")
    if None in (eef_x, eef_y, eef_z, cube_x, cube_y, cube_z):
        return None, None, None

    dx = cube_x - eef_x
    dy = cube_y - eef_y
    dz = cube_z - eef_z
    xy_distance = math.sqrt(dx * dx + dy * dy)
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    return distance, xy_distance, dz


def print_step_summary(step, obs, reward):
    print(f"\nStep {step}")

    if "robot0_eef_pos" in obs:
        print("EEF position:", obs["robot0_eef_pos"])

    if "cube_pos" in obs:
        print("Cube position:", obs["cube_pos"])

    distance, xy_distance, z_gap = obs_metrics(obs)
    if distance is not None:
        print("EEF-cube distance:", round(distance, 4))
        print("EEF-cube XY distance:", round(xy_distance, 4))
        print("Cube minus EEF Z:", round(z_gap, 4))

    print("Reward:", reward)


def make_log_writer(log_file):
    if log_file is None:
        return None, None

    log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = log_file.open("w", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "step",
            "reward",
            "done",
            "eef_x",
            "eef_y",
            "eef_z",
            "cube_x",
            "cube_y",
            "cube_z",
            "eef_cube_distance",
            "eef_cube_xy_distance",
            "cube_minus_eef_z",
        ],
    )
    writer.writeheader()
    return handle, writer


def vector_components(obs, key):
    value = obs.get(key)
    if value is None or len(value) < 3:
        return None, None, None
    return float(value[0]), float(value[1]), float(value[2])


def write_step_log(writer, step, obs, reward, done):
    if writer is None:
        return

    eef_x, eef_y, eef_z = vector_components(obs, "robot0_eef_pos")
    cube_x, cube_y, cube_z = vector_components(obs, "cube_pos")
    distance, xy_distance, z_gap = obs_metrics(obs)
    writer.writerow(
        {
            "step": step,
            "reward": float(reward),
            "done": bool(done),
            "eef_x": eef_x,
            "eef_y": eef_y,
            "eef_z": eef_z,
            "cube_x": cube_x,
            "cube_y": cube_y,
            "cube_z": cube_z,
            "eef_cube_distance": distance,
            "eef_cube_xy_distance": xy_distance,
            "cube_minus_eef_z": z_gap,
        }
    )


def main():
    args = parse_args()
    env = make_env(has_renderer=not args.no_render)
    log_handle, log_writer = make_log_writer(args.log_file)

    try:
        obs = env.reset()

        print("==============================")
        print("Environment loaded")
        print("Action dimension:", env.action_dim)
        print("Render enabled:", not args.no_render)
        print("Steps:", args.steps)
        print("Policy:", args.policy)
        print("==============================")

        print("\nObservation keys:")
        for key in obs.keys():
            print("-", key)

        for step in range(args.steps):
            action = scripted_action(env.action_dim, step, obs, args.policy)
            obs, reward, done, info = env.step(action)

            if not args.no_render:
                env.render()

            if args.print_every > 0 and step % args.print_every == 0:
                print_step_summary(step, obs, reward)

            write_step_log(log_writer, step, obs, reward, done)

            if done:
                print(f"\nEnvironment returned done at step {step}.")
                if not args.reset_on_done:
                    break
                obs = env.reset()

            if not args.no_render and args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        if log_handle is not None:
            log_handle.close()
            print(f"Saved step log to {args.log_file}")
        env.close()


if __name__ == "__main__":
    main()
