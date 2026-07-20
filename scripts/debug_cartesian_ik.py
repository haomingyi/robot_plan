import argparse
import csv
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from dual_arm_pick_place import (
    APPROACH_L,
    DRIVE_READY_R,
    LEFT_JOINTS,
    MODEL_PATH,
    OPEN_GRIPPER,
    build_maps,
    set_base_pose,
    set_joint_pose,
)
from ik_utils import position_ik_step, site_position

SITE_NAME = "left_grip_center"
DEFAULT_LOG = Path(__file__).resolve().parents[1] / "logs" / "cartesian_ik_debug.csv"
TARGET_STEP = 0.01
POSITION_TOLERANCE = 0.001


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Cartesian IK debugger for the left arm.")
    parser.add_argument("--headless", action="store_true", help="Solve once without opening the Viewer.")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z", help="Initial target direction.")
    parser.add_argument("--distance", type=float, default=0.05, help="Initial target distance in meters.")
    parser.add_argument("--damping", type=float, default=0.03, help="Damped least-squares coefficient.")
    parser.add_argument("--max-joint-step", type=float, default=0.05, help="Maximum joint update per IK iteration.")
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum iterations for automatic solve.")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG, help="CSV output path.")
    return parser.parse_args()


def initialize(args):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    joint_qpos, actuator_id = build_maps(model)
    set_base_pose(model, data, joint_qpos, actuator_id, 0.25, 0.0)
    set_joint_pose(
        data,
        joint_qpos,
        actuator_id,
        APPROACH_L,
        DRIVE_READY_R,
        OPEN_GRIPPER,
        OPEN_GRIPPER,
    )
    mujoco.mj_forward(model, data)

    initial_qpos = data.qpos.copy()
    initial_ctrl = data.ctrl.copy()
    start = site_position(model, data, SITE_NAME)
    direction = np.zeros(3)
    direction["xyz".index(args.axis)] = args.distance
    target = start + direction
    return model, data, initial_qpos, initial_ctrl, start, target


def add_marker(scene, geom_type, size, pos, rgba):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.asarray(size, dtype=float),
        np.asarray(pos, dtype=float),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return geom


def update_debug_geometry(viewer, current, target):
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        add_marker(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012] * 3, target, [1.0, 0.1, 0.1, 0.9])
        add_marker(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.009] * 3, current, [0.0, 0.9, 0.3, 0.9])
        line = add_marker(scene, mujoco.mjtGeom.mjGEOM_LINE, [1.0, 1.0, 1.0], current, [1.0, 0.8, 0.0, 1.0])
        mujoco.mjv_connector(line, mujoco.mjtGeom.mjGEOM_LINE, 3.0, current, target)


def write_header(writer):
    writer.writerow(
        [
            "iteration",
            "target_x",
            "target_y",
            "target_z",
            "actual_x",
            "actual_y",
            "actual_z",
            "error_before",
            "error_after",
            "dq_norm",
            "min_singular_value",
            *LEFT_JOINTS,
        ]
    )


def write_result(writer, iteration, target, result):
    writer.writerow(
        [
            iteration,
            *target,
            *result.current_position,
            result.error_before,
            result.error_after,
            result.dq_norm,
            result.min_singular_value,
            *result.joint_positions,
        ]
    )


def solve_step(args, model, data, target):
    return position_ik_step(
        model,
        data,
        SITE_NAME,
        LEFT_JOINTS,
        target,
        damping=args.damping,
        max_joint_step=args.max_joint_step,
    )


def run_headless(args, model, data, target, writer, log_handle):
    result = None
    for iteration in range(1, args.max_iterations + 1):
        result = solve_step(args, model, data, target)
        write_result(writer, iteration, target, result)
        log_handle.flush()
        print(
            f"iteration={iteration:03d} error={result.error_after:.6f} "
            f"dq={result.dq_norm:.6f} sigma_min={result.min_singular_value:.6f}"
        )
        if result.error_after <= POSITION_TOLERANCE:
            break
    print(
        f"ik_debug_ok converged={result.error_after <= POSITION_TOLERANCE} "
        f"iterations={iteration} final_error={result.error_after:.6f}"
    )


def run_viewer(args, model, data, initial_qpos, initial_ctrl, target, writer, log_handle):
    state = {
        "target": target.copy(),
        "step_requested": False,
        "auto_solve": False,
        "reset_requested": False,
        "iteration": 0,
        "last": None,
    }

    def key_callback(keycode):
        movement = {
            65: np.array([-TARGET_STEP, 0.0, 0.0]),
            68: np.array([TARGET_STEP, 0.0, 0.0]),
            87: np.array([0.0, TARGET_STEP, 0.0]),
            83: np.array([0.0, -TARGET_STEP, 0.0]),
            81: np.array([0.0, 0.0, -TARGET_STEP]),
            69: np.array([0.0, 0.0, TARGET_STEP]),
        }
        upper_key = keycode - 32 if 97 <= keycode <= 122 else keycode
        if upper_key in movement:
            state["target"] += movement[upper_key]
            state["auto_solve"] = False
            print("target =", np.round(state["target"], 4))
        elif upper_key == 78:
            state["step_requested"] = True
            state["auto_solve"] = False
        elif upper_key == 71:
            state["auto_solve"] = True
        elif upper_key == 82:
            state["reset_requested"] = True

    print("Cartesian IK Viewer")
    print("Red sphere: target | Green sphere: current gripper | Yellow line: position error")
    print("A/D: X-/X+ | S/W: Y-/Y+ | Q/E: Z-/Z+ | N: one IK step | G: solve | R: reset")

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        viewer.cam.lookat[:] = [-0.05, 0.80, 0.55]
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18

        while viewer.is_running():
            if state["reset_requested"]:
                data.qpos[:] = initial_qpos
                data.ctrl[:] = initial_ctrl
                mujoco.mj_forward(model, data)
                state["target"] = target.copy()
                state["iteration"] = 0
                state["last"] = None
                state["auto_solve"] = False
                state["reset_requested"] = False
                print("reset: target =", np.round(state["target"], 4))

            current = site_position(model, data, SITE_NAME)
            error = float(np.linalg.norm(state["target"] - current))
            should_step = state["step_requested"] or (
                state["auto_solve"]
                and error > POSITION_TOLERANCE
                and state["iteration"] < args.max_iterations
            )
            if should_step:
                state["iteration"] += 1
                state["last"] = solve_step(args, model, data, state["target"])
                state["step_requested"] = False
                write_result(writer, state["iteration"], state["target"], state["last"])
                log_handle.flush()
                print(
                    f"iteration={state['iteration']:03d} "
                    f"error {state['last'].error_before:.6f} -> {state['last'].error_after:.6f} m | "
                    f"dq={state['last'].dq_norm:.6f} | sigma_min={state['last'].min_singular_value:.6f}"
                )
                if state["last"].error_after <= POSITION_TOLERANCE:
                    state["auto_solve"] = False
                    print("target reached: error <= 0.001 m")
                elif state["iteration"] >= args.max_iterations:
                    state["auto_solve"] = False
                    print("solve stopped: maximum iterations reached")

            current = site_position(model, data, SITE_NAME)
            error = float(np.linalg.norm(state["target"] - current))
            update_debug_geometry(viewer, current, state["target"])
            status = "SOLVING" if state["auto_solve"] else "READY"
            viewer.set_texts(
                [
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "Cartesian IK",
                        f"{status} | iteration {state['iteration']} | error {error:.6f} m",
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                        "Keys",
                        "A/D X | S/W Y | Q/E Z | N one step | G solve | R reset",
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                        "Target",
                        f"x {state['target'][0]:.3f} | y {state['target'][1]:.3f} | z {state['target'][2]:.3f}",
                    ),
                ]
            )
            viewer.sync()
            time.sleep(0.05)


def main():
    args = parse_args()
    model, data, initial_qpos, initial_ctrl, start, target = initialize(args)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    print("start  =", np.round(start, 6))
    print("target =", np.round(target, 6))
    print("log    =", args.log_file)

    with args.log_file.open("w", newline="") as log_handle:
        writer = csv.writer(log_handle)
        write_header(writer)
        log_handle.flush()
        if args.headless:
            run_headless(args, model, data, target, writer, log_handle)
        else:
            run_viewer(
                args,
                model,
                data,
                initial_qpos,
                initial_ctrl,
                target,
                writer,
                log_handle,
            )


if __name__ == "__main__":
    main()
