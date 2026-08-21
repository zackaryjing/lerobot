"""Manager-based LeIsaac scene for the SO-101 wood-pick task."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass
from leisaac.tasks.template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from leisaac.tasks.template.single_arm_env_cfg import SingleArmEventCfg

from . import mdp
from .materials import apply_scene_materials
from .scene_parameters import SCENE


def _static_cuboid(
    prim_name: str,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
    color: tuple[float, float, float],
    roughness: float,
    collision_enabled: bool = True,
) -> AssetBaseCfg:
    """Create one static, collidable scene component."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision_enabled),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=roughness,
                metallic=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=center),
    )


def _box_component_geometry() -> dict[str, tuple[float, float, float]]:
    """Return sizes and centres for the open box's floor and four walls."""
    x0, y0, _ = SCENE.box_lower_left
    length, width, height = SCENE.box_size
    wall = SCENE.box_wall_thickness
    cx = x0 + length / 2.0
    cy = y0 + width / 2.0
    return {
        "floor_size": (length - 2.0 * wall, width - 2.0 * wall, wall),
        "floor_center": (cx, cy, wall / 2.0),
        "left_size": (wall, width, height),
        "left_center": (x0 + wall / 2.0, cy, height / 2.0),
        "right_size": (wall, width, height),
        "right_center": (x0 + length - wall / 2.0, cy, height / 2.0),
        "near_size": (length - 2.0 * wall, wall, height),
        "near_center": (cx, y0 + wall / 2.0, height / 2.0),
        "far_size": (length - 2.0 * wall, wall, height),
        "far_center": (cx, y0 + width - wall / 2.0, height / 2.0),
    }


_BOX = _box_component_geometry()


@configclass
class WoodPickSceneCfg(SingleArmTaskSceneCfg):
    """Measured SO-101 workcell with two RGB cameras."""

    # ``scene`` is the required base-scene asset in LeIsaac's task template.
    scene: AssetBaseCfg = _static_cuboid(
        "Table",
        SCENE.table_size,
        SCENE.table_center,
        SCENE.table_color,
        SCENE.table_roughness,
    )

    platform: AssetBaseCfg = _static_cuboid(
        "PlacementPlatform",
        SCENE.platform_size,
        SCENE.platform_center,
        SCENE.platform_color,
        SCENE.platform_roughness,
    )

    # A thin, visual-only skin gives just the world -X side its measured teal
    # colour while retaining one simple white cuboid for platform collision.
    platform_left_face: AssetBaseCfg = _static_cuboid(
        "PlacementPlatformLeftFace",
        (
            SCENE.platform_face_thickness,
            SCENE.platform_size[1],
            SCENE.platform_size[2],
        ),
        (
            SCENE.platform_center[0]
            - SCENE.platform_size[0] / 2.0
            - SCENE.platform_face_thickness / 2.0,
            SCENE.platform_center[1],
            SCENE.platform_center[2],
        ),
        SCENE.platform_left_face_color,
        SCENE.platform_roughness,
        collision_enabled=False,
    )

    box_floor: AssetBaseCfg = _static_cuboid(
        "DestinationBoxFloor",
        _BOX["floor_size"],
        _BOX["floor_center"],
        SCENE.box_bottom_color,
        SCENE.cardboard_roughness,
    )
    box_left: AssetBaseCfg = _static_cuboid(
        "DestinationBoxLeftWall",
        _BOX["left_size"],
        _BOX["left_center"],
        SCENE.cardboard_color,
        SCENE.cardboard_roughness,
    )
    box_right: AssetBaseCfg = _static_cuboid(
        "DestinationBoxRightWall",
        _BOX["right_size"],
        _BOX["right_center"],
        SCENE.cardboard_color,
        SCENE.cardboard_roughness,
    )
    box_near: AssetBaseCfg = _static_cuboid(
        "DestinationBoxNearWall",
        _BOX["near_size"],
        _BOX["near_center"],
        SCENE.cardboard_color,
        SCENE.cardboard_roughness,
    )
    box_far: AssetBaseCfg = _static_cuboid(
        "DestinationBoxFarWall",
        _BOX["far_size"],
        _BOX["far_center"],
        SCENE.cardboard_color,
        SCENE.cardboard_roughness,
    )

    stick: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stick",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCENE.stick_usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=SCENE.stick_mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=SCENE.stick_color,
                roughness=SCENE.stick_roughness,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=SCENE.stick_position,
            rot=SCENE.stick_quaternion,
        ),
    )

    # The measured overhead camera is fixed in the environment, not attached to
    # a moving robot link. Identity OpenGL orientation looks along world -Z.
    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=SCENE.front_camera_position,
            rot=SCENE.front_camera_quaternion,
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=SCENE.front_focal_length,
            focus_distance=0.595,
            horizontal_aperture=SCENE.front_horizontal_aperture,
            clipping_range=(0.01, 5.0),
            lock_camera=True,
        ),
        width=SCENE.camera_width,
        height=SCENE.camera_height,
        update_period=1.0 / SCENE.camera_fps,
    )

    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=SCENE.wrist_camera_position,
            rot=SCENE.wrist_camera_quaternion,
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=SCENE.wrist_focal_length,
            focus_distance=0.25,
            horizontal_aperture=SCENE.wrist_horizontal_aperture,
            clipping_range=(0.01, 5.0),
            lock_camera=True,
        ),
        width=SCENE.camera_width,
        height=SCENE.camera_height,
        update_period=1.0 / SCENE.camera_fps,
    )

    # Record per-link contact forces during learned-policy evaluation.  This
    # makes it possible to distinguish a policy-command reversal from a real
    # collision/depenetration response instead of inferring contact from video.
    robot_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=1,
        track_pose=False,
        track_air_time=False,
    )

    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=SCENE.dome_color,
            intensity=SCENE.dome_intensity,
        ),
    )
    key_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/KeyLight",
        spawn=sim_utils.SphereLightCfg(
            color=SCENE.key_color,
            intensity=SCENE.key_intensity,
            radius=SCENE.key_radius,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=SCENE.key_position),
    )
    fill_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FillLight",
        spawn=sim_utils.DistantLightCfg(
            color=SCENE.fill_color,
            intensity=SCENE.fill_intensity,
            angle=0.70,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(rot=SCENE.fill_quaternion),
    )


@configclass
class WoodPickObservationsCfg(SingleArmObservationsCfg):
    """Use LeIsaac's state, end-effector, front, and wrist observations."""


@configclass
class WoodPickEventsCfg(SingleArmEventCfg):
    """Reset the target to a random, supported and coarsely reachable pose."""

    scene_materials = EventTerm(func=apply_scene_materials, mode="startup")

    reset_stick = EventTerm(
        func=mdp.reset_stick_on_platform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("stick"),
            "platform_center_xy": SCENE.platform_center[:2],
            "platform_size_xy": SCENE.platform_size[:2],
            "platform_top_z": SCENE.platform_size[2],
        },
    )


@configclass
class WoodPickTerminationsCfg(SingleArmTerminationsCfg):
    """Terminate when the whole, settled stick is inside the destination box."""

    success = DoneTerm(
        func=mdp.stick_in_destination_box,
        params={
            "asset_cfg": SceneEntityCfg("stick"),
            "box_lower_left": SCENE.box_lower_left,
            "box_size": SCENE.box_size,
            "wall_thickness": SCENE.box_wall_thickness,
        },
    )


@configclass
class WoodPickEnvCfg(SingleArmTaskEnvCfg):
    """Measured digital-twin configuration."""

    scene: WoodPickSceneCfg = WoodPickSceneCfg(num_envs=1, env_spacing=2.0)
    observations: WoodPickObservationsCfg = WoodPickObservationsCfg()
    events: WoodPickEventsCfg = WoodPickEventsCfg()
    terminations: WoodPickTerminationsCfg = WoodPickTerminationsCfg()
    task_description: str = "Pick up the wooden stick and place it in the box."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot.init_state.pos = SCENE.robot_position
        self.scene.robot.init_state.joint_pos = {
            "shoulder_pan": math.radians(0.0),
            "shoulder_lift": math.radians(-99.0),
            # The stock LeIsaac SO-101 USD caps this joint at +90 degrees.
            "elbow_flex": math.radians(90.0),
            "wrist_flex": math.radians(53.0),
            "wrist_roll": math.radians(0.0),
            "gripper": math.radians(7.0),
        }

        self.episode_length_s = 25.0
        self.viewer.eye = (-0.35, -0.20, 0.70)
        self.viewer.lookat = (0.10, 0.35, 0.05)

        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        # One image per 30 Hz control step. The base template defaults to one
        # render per 120 Hz physics step, redundantly rendering four frames for
        # every action and making offline camera episodes roughly 4x slower.
        self.sim.render_interval = self.decimation
        self.sim.render.enable_translucency = True
        self.sim.render.antialiasing_mode = "FXAA"
        self.sim.render.rendering_mode = "quality"
