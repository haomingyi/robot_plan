"""Minimal Isaac Sim headless smoke test.

Run with:
    /home/hzm/isaacsim/python.sh sim_isaac/check_isaac_headless.py
"""

import numpy as np
from pathlib import Path
from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": True})

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid  # noqa: E402


def main():
    repo_root = Path(__file__).resolve().parents[1]
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/test_cube",
            name="test_cube",
            position=np.array([0.0, 0.0, 1.0]),
            scale=np.array([0.2, 0.2, 0.2]),
            color=np.array([0.1, 0.4, 1.0]),
        )
    )

    world.reset()
    for _ in range(120):
        world.step(render=False)

    position, _ = cube.get_world_pose()
    message = "isaac_headless_ok final_cube_z={:.4f}".format(float(position[2]))
    output_file = repo_root / "outputs" / "isaac_headless_check.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(message + "\n")
    print(message, flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
