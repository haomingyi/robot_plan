from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class IKStepResult:
    error_before: float
    error_after: float
    dq_norm: float
    min_singular_value: float
    current_position: np.ndarray
    joint_positions: np.ndarray


def joint_indices(model, joint_names):
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_names
    ]
    if any(joint_id < 0 for joint_id in joint_ids):
        missing = [name for name, joint_id in zip(joint_names, joint_ids) if joint_id < 0]
        raise ValueError(f"Unknown joints: {', '.join(missing)}")

    qpos_indices = np.array([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
    dof_indices = np.array([model.jnt_dofadr[joint_id] for joint_id in joint_ids])
    return joint_ids, qpos_indices, dof_indices


def site_position(model, data, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Unknown site: {site_name}")
    return data.site_xpos[site_id].copy()


def position_ik_step(
    model,
    data,
    site_name,
    joint_names,
    target,
    *,
    damping=0.03,
    max_joint_step=0.05,
):
    joint_ids, qpos_indices, dof_indices = joint_indices(model, joint_names)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Unknown site: {site_name}")

    mujoco.mj_forward(model, data)
    current = data.site_xpos[site_id].copy()
    error = np.asarray(target, dtype=float) - current

    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))
    mujoco.mj_jacSite(
        model,
        data,
        jacobian_position,
        jacobian_rotation,
        site_id,
    )
    jacobian = jacobian_position[:, dof_indices]
    singular_values = np.linalg.svd(jacobian, compute_uv=False)

    regularized = jacobian @ jacobian.T + damping**2 * np.eye(3)
    joint_delta = jacobian.T @ np.linalg.solve(regularized, error)
    joint_delta = np.clip(joint_delta, -max_joint_step, max_joint_step)

    new_positions = data.qpos[qpos_indices].copy() + joint_delta
    for index, joint_id in enumerate(joint_ids):
        if model.jnt_limited[joint_id]:
            low, high = model.jnt_range[joint_id]
            new_positions[index] = np.clip(new_positions[index], low, high)
        data.qpos[qpos_indices[index]] = new_positions[index]

        actuator_name = f"{joint_names[index]}_pos"
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )
        if actuator_id >= 0:
            data.ctrl[actuator_id] = new_positions[index]

    mujoco.mj_forward(model, data)
    updated_position = data.site_xpos[site_id].copy()
    return IKStepResult(
        error_before=float(np.linalg.norm(error)),
        error_after=float(np.linalg.norm(np.asarray(target) - updated_position)),
        dq_norm=float(np.linalg.norm(joint_delta)),
        min_singular_value=float(singular_values[-1]),
        current_position=updated_position,
        joint_positions=new_positions,
    )
