"""Runtime visual-material overrides for instanced USD assets."""

import isaaclab.sim as sim_utils

from .scene_parameters import SCENE


def apply_scene_materials(env, env_ids) -> None:
    """Apply measured colours to the robot and imported wooden target.

    The SO-101 asset contains instanceable visual subtrees. Applying a new
    material binding below ``gripper/visuals`` or ``jaw/visuals`` at runtime
    attempts to author on an instance proxy, which USD forbids. A strong
    binding on each non-instanced link root safely overrides the white material
    within those two visual instances without recolouring the rest of the arm.
    """
    from pxr import Gf, UsdShade

    stage = sim_utils.get_current_stage()
    if env_ids is None or isinstance(env_ids, slice):
        env_indices = range(env.scene.num_envs)
    else:
        env_indices = [int(index) for index in env_ids]

    for index in env_indices:
        robot_path = f"{env.scene.env_prim_paths[index]}/Robot"
        body_shader = UsdShade.Shader(
            stage.GetPrimAtPath(f"{robot_path}/Looks/material_a_3d_printed/Shader")
        )
        black_shader = UsdShade.Shader(
            stage.GetPrimAtPath(f"{robot_path}/Looks/material_sts3215/Shader")
        )
        black_material = UsdShade.Material(
            stage.GetPrimAtPath(f"{robot_path}/Looks/material_sts3215")
        )
        if not body_shader or not black_shader or not black_material:
            raise RuntimeError(f"SO-101 material hierarchy not found below {robot_path}")

        body_shader.GetInput("diffuse_color_constant").Set(Gf.Vec3f(*SCENE.robot_body_color))
        black_shader.GetInput("diffuse_color_constant").Set(Gf.Vec3f(*SCENE.robot_black_color))

        for link_name in ("gripper", "jaw"):
            link_prim = stage.GetPrimAtPath(f"{robot_path}/{link_name}")
            if not link_prim:
                raise RuntimeError(f"SO-101 link not found: {robot_path}/{link_name}")
            binding_api = UsdShade.MaterialBindingAPI.Apply(link_prim)
            binding_api.Bind(
                black_material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )

        # The converted wooden-target USD has its own MDL shader directly on
        # the child mesh, so a spawn-time root material does not affect it.
        # Its material is unique to this imported target; editing that shader
        # input recolours only the stick and leaves the source USD unchanged.
        stick_shader_path = (
            f"{env.scene.env_prim_paths[index]}"
            "/Stick/geometry/Looks/DefaultMaterial/DefaultMaterial"
        )
        stick_shader = UsdShade.Shader(stage.GetPrimAtPath(stick_shader_path))
        if not stick_shader:
            raise RuntimeError(f"Wooden target shader not found: {stick_shader_path}")
        stick_shader.GetInput("diffuse_color_constant").Set(Gf.Vec3f(*SCENE.stick_color))
