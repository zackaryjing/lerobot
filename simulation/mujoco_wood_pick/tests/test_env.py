from __future__ import annotations

import unittest

import mujoco
import numpy as np

from mujoco_wood_pick import WoodPickEnv
from mujoco_wood_pick.env import HOME_QPOS
from mujoco_wood_pick.scripted_control import SCRIPTED_HOME


class WoodPickEnvTest(unittest.TestCase):
    def test_model_compiles_and_reset_matches_home(self) -> None:
        env = WoodPickEnv(render_images=False)
        try:
            observation, info = env.reset(seed=0)
            np.testing.assert_allclose(observation["observation.state"], HOME_QPOS, atol=1e-6)
            self.assertEqual(env.model.nu, 6)
            self.assertFalse(info["success"])
        finally:
            env.close()

    def test_one_control_step_is_30_hz(self) -> None:
        env = WoodPickEnv(render_images=False)
        try:
            env.reset(seed=0)
            _, _, terminated, truncated, info = env.step(HOME_QPOS)
            self.assertTrue(np.isclose(info["sim_time"], 1.0 / 30.0))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        finally:
            env.close()

    def test_scripted_tcp_marks_closed_tip_midpoint(self) -> None:
        env = WoodPickEnv(render_images=False)
        try:
            env.reset(seed=0)
            env.data.qpos[env._qpos_addresses[-1]] = np.deg2rad(-3.1)
            import mujoco

            mujoco.mj_forward(env.model, env.data)
            tcp = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_SITE, "scripted_tcp"
            )
            fixed = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_sph_tip1"
            )
            moving = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_sph_tip1"
            )
            expected = 0.5 * (
                env.data.geom_xpos[fixed] + env.data.geom_xpos[moving]
            )
            np.testing.assert_allclose(env.data.site_xpos[tcp], expected, atol=1e-8)
        finally:
            env.close()

    def test_stick_uses_full_block_mesh_collision_proxy(self) -> None:
        """Keep the detailed toy visual, but collide as an uncut block."""
        env = WoodPickEnv(render_images=False)
        try:
            stick_geom = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_GEOM, "stick_geom"
            )
            self.assertEqual(env.model.geom_type[stick_geom], mujoco.mjtGeom.mjGEOM_MESH)
            mesh = int(env.model.geom_dataid[stick_geom])
            vertex_start = int(env.model.mesh_vertadr[mesh])
            vertex_count = int(env.model.mesh_vertnum[mesh])
            vertices = env.model.mesh_vert[vertex_start : vertex_start + vertex_count]
            self.assertEqual(vertex_count, 8)
            np.testing.assert_allclose(
                vertices.min(axis=0), (-0.00575, -0.00575, -0.040), atol=1e-8
            )
            np.testing.assert_allclose(
                vertices.max(axis=0), (0.00575, 0.00575, 0.040), atol=1e-8
            )
        finally:
            env.close()

    def test_oblique_stick_rest_does_not_create_deep_platform_contact(self) -> None:
        """Regression for the primitive box-box false manifold seen at t=0.646 s."""
        env = WoodPickEnv(render_images=False)
        try:
            # This exact pose used to acquire an 85.878 mm platform penetration
            # while sitting still, launching the 30 g stick at roughly 4.3 m/s.
            position = np.array((0.14939604528579054, 0.3604297096644056, 0.05875))
            quaternion = np.array(
                (0.6355762273493627, 0.3099078237611166,
                 0.6355762273493627, -0.3099078237611166)
            )
            env.reset(
                options={
                    "joint_positions": SCRIPTED_HOME,
                    "stick_position": position,
                    "stick_quaternion": quaternion,
                }
            )
            stick_geom = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_GEOM, "stick_geom"
            )
            worst_penetration = 0.0
            for _ in range(600):
                mujoco.mj_step(env.model, env.data)
                for contact in env.data.contact:
                    if contact.geom1 == stick_geom or contact.geom2 == stick_geom:
                        worst_penetration = max(worst_penetration, -float(contact.dist))

            displacement = np.linalg.norm(env.data.xpos[env._stick_body_id] - position)
            self.assertLess(worst_penetration, 1.0e-4)
            self.assertLess(displacement, 1.0e-4)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
