"""Run and inspect simple robosuite Panda Lift policies.

The script is intentionally lightweight, but it is structured for debugging:
policies are selectable, per-step diagnostics can be logged, and the staged pick
policy exposes its internal phase.
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
    parser.add_argument(
        "--success-cube-z",
        type=float,
        default=0.90,
        help="Cube height threshold used for the final success summary.",
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


def vector_components(obs, key):
    value = obs.get(key)
    if value is None or len(value) < 3:
        return None, None, None
    return float(value[0]), float(value[1]), float(value[2])


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


def relative_cube_offset(obs):
    offset = obs.get("gripper_to_cube_pos")
    if offset is None or len(offset) < 3:
        return None

    import numpy as np

    return np.asarray(offset, dtype=float)


def move_to_relative_offset(action_dim, offset, desired_cube_below_gripper, gain=2.2, limit=0.25):
    import numpy as np

    action = zero_action(action_dim)
    error = offset - np.asarray(desired_cube_below_gripper, dtype=float)
    action[:3] = np.clip(error * gain, -limit, limit)
    return action


class SmokePolicy:
    phase = "smoke"

    def act(self, action_dim, step, obs):
        action = zero_action(action_dim)

        if step > 100:
            action[-1] = GRIPPER_CLOSE

        if 200 < step < 300:
            action[0] = 0.2

        return action


class ApproachPolicy:
    phase = "approach"

    def act(self, action_dim, step, obs):
        offset = relative_cube_offset(obs)
        if offset is None:
            return SmokePolicy().act(action_dim, step, obs)

        action = move_to_relative_offset(action_dim, offset, desired_cube_below_gripper=[0.0, 0.0, -0.05])
        action[-1] = GRIPPER_OPEN
        return action


class PickPolicy:
    """Observation-driven staged pick policy.

    This is still a scripted baseline, not a general manipulation planner. The
    important part is that phase transitions are based on measured pose error
    and gripper timing instead of fixed global step numbers.
    """

    def __init__(self):
        self.phase = "hover"
        self.phase_steps = 0

    def set_phase(self, phase):
        if self.phase != phase:
            self.phase = phase
            self.phase_steps = 0

    def act(self, action_dim, step, obs):
        import numpy as np

        offset = relative_cube_offset(obs)
        if offset is None:
            return SmokePolicy().act(action_dim, step, obs)

        xy_error = float(np.linalg.norm(offset[:2]))
        z_gap = float(offset[2])

        if self.phase == "hover":
            if xy_error < 0.018 and -0.060 <= z_gap <= -0.035:
                self.set_phase("descend")
            action = move_to_relative_offset(action_dim, offset, desired_cube_below_gripper=[0.0, 0.0, -0.05])
            action[-1] = GRIPPER_OPEN

        elif self.phase == "descend":
            if xy_error < 0.018 and abs(z_gap) < 0.014:
                self.set_phase("grasp")
            action = move_to_relative_offset(action_dim, offset, desired_cube_below_gripper=[0.0, 0.0, 0.0], gain=2.0)
            action[-1] = GRIPPER_OPEN

        elif self.phase == "grasp":
            if self.phase_steps >= 55:
                self.set_phase("lift")
            action = zero_action(action_dim)
            action[-1] = GRIPPER_CLOSE

        else:
            action = zero_action(action_dim)
            action[2] = 0.25
            action[-1] = GRIPPER_CLOSE

        self.phase_steps += 1
        return action


def make_policy(name):
    if name == "approach":
        return ApproachPolicy()
    if name == "pick":
        return PickPolicy()
    return SmokePolicy()


def print_step_summary(step, obs, reward, policy):
    print(f"\nStep {step}")

    if hasattr(policy, "phase"):
        print("Policy phase:", policy.phase)

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
            "phase",
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


def write_step_log(writer, step, obs, reward, done, policy):
    if writer is None:
        return

    eef_x, eef_y, eef_z = vector_components(obs, "robot0_eef_pos")
    cube_x, cube_y, cube_z = vector_components(obs, "cube_pos")
    distance, xy_distance, z_gap = obs_metrics(obs)
    writer.writerow(
        {
            "step": step,
            "phase": getattr(policy, "phase", ""),
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


def update_summary(summary, obs, reward):
    cube_z = vector_components(obs, "cube_pos")[2]
    if cube_z is not None:
        summary["max_cube_z"] = max(summary["max_cube_z"], cube_z)
    summary["max_reward"] = max(summary["max_reward"], float(reward))


def print_final_summary(summary, success_cube_z):
    success = summary["max_reward"] >= 1.0 or summary["max_cube_z"] >= success_cube_z
    print("\n==============================")
    print("Run summary")
    print("Max reward:", round(summary["max_reward"], 4))
    print("Max cube_z:", round(summary["max_cube_z"], 4))
    print("Success threshold cube_z:", success_cube_z)
    print("Success:", success)
    print("==============================")


def main():
    args = parse_args()
    env = make_env(has_renderer=not args.no_render)
    policy = make_policy(args.policy)
    log_handle, log_writer = make_log_writer(args.log_file)
    summary = {"max_reward": 0.0, "max_cube_z": float("-inf")}

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
            action = policy.act(env.action_dim, step, obs)
            obs, reward, done, info = env.step(action)
            update_summary(summary, obs, reward)

            if not args.no_render:
                env.render()

            if args.print_every > 0 and step % args.print_every == 0:
                print_step_summary(step, obs, reward, policy)

            write_step_log(log_writer, step, obs, reward, done, policy)

            if done:
                print(f"\nEnvironment returned done at step {step}.")
                if not args.reset_on_done:
                    break
                obs = env.reset()
                policy = make_policy(args.policy)

            if not args.no_render and args.sleep > 0:
                time.sleep(args.sleep)

        print_final_summary(summary, args.success_cube_z)
    finally:
        if log_handle is not None:
            log_handle.close()
            print(f"Saved step log to {args.log_file}")
        env.close()


if __name__ == "__main__":
    main()
