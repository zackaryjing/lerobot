"""Measured scene constants.

The physical measurements supplied for this task were in centimetres and are
stored here in SI units. Quaternions use Isaac Lab's ``(w, x, y, z)`` order.

World frame:
    origin: midpoint of the rear edge of the follower-arm base
    +X: right
    +Y: forward
    +Z: up
    Z=0: tabletop surface
"""

from dataclasses import dataclass
from pathlib import Path


PACKAGE_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
WOODEN_TARGET_USD_PATH = PACKAGE_ASSETS_DIR / "wooden_target.usd"


def _srgb_hex_to_linear(value: int) -> tuple[float, float, float]:
    """Convert a display-referred ``0xRRGGBB`` colour to shader-linear RGB."""

    def linear_channel(byte: int) -> float:
        channel = byte / 255.0
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    return tuple(linear_channel((value >> shift) & 0xFF) for shift in (16, 8, 0))


@dataclass(frozen=True)
class SceneParameters:
    # The exact desk outline is outside both camera views. This size safely
    # encloses every measured object while keeping the tabletop surface at Z=0.
    table_size: tuple[float, float, float] = (0.90, 0.80, 0.050)
    table_center: tuple[float, float, float] = (0.10, 0.30, -0.025)
    # Measured from an unobstructed patch in the real front-camera frame:
    # #C0BDB3.
    table_color: tuple[float, float, float] = _srgb_hex_to_linear(0xC0BDB3)
    table_roughness: float = 0.72

    # White object-placement platform: lower-left=(10,31.2)cm, LxW=27x10cm.
    platform_size: tuple[float, float, float] = (0.270, 0.100, 0.053)
    platform_center: tuple[float, float, float] = (0.235, 0.362, 0.0265)
    platform_color: tuple[float, float, float] = (206 / 255, 208 / 255, 205 / 255)
    # Only the platform's world -X (left) face is dark teal: #004A4E.
    platform_left_face_color: tuple[float, float, float] = _srgb_hex_to_linear(0x004A4E)
    platform_face_thickness: float = 0.0005
    platform_roughness: float = 0.62

    # Open destination box: lower-left=(-10.7,38,0)cm, LxWxH=15x10x8.5cm.
    box_lower_left: tuple[float, float, float] = (-0.107, 0.380, 0.0)
    box_size: tuple[float, float, float] = (0.150, 0.100, 0.085)
    box_wall_thickness: float = 0.003
    cardboard_color: tuple[float, float, float] = _srgb_hex_to_linear(0x583E29)
    box_bottom_color: tuple[float, float, float] = _srgb_hex_to_linear(0xA56B12)
    cardboard_roughness: float = 0.88

    # STL bounds are 11.5x11.5x80 mm. The converted USD is centred, scaled to
    # metres, and rotated so its long local-Z axis becomes world +X.
    stick_usd_path: str = str(WOODEN_TARGET_USD_PATH)
    stick_position: tuple[float, float, float] = (0.235, 0.362, 0.05875)
    stick_quaternion: tuple[float, float, float, float] = (
        0.70710678,
        0.0,
        0.70710678,
        0.0,
    )
    stick_mass_kg: float = 0.030
    stick_color: tuple[float, float, float] = _srgb_hex_to_linear(0x804A1B)
    stick_roughness: float = 0.68
    stick_static_friction: float = 0.70
    stick_dynamic_friction: float = 0.58

    # LeIsaac's SO-101 asset is placed at the measured world origin. Its stock
    # asset is yellow, so startup material overrides reproduce the real arm.
    robot_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_body_color: tuple[float, float, float] = (0.82, 0.82, 0.80)
    robot_black_color: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Fixed overhead camera. With the OpenGL convention its optical axis points
    # along world -Z. The real camera is mounted 90 degrees clockwise when seen
    # from above, hence the -90-degree world-Z quaternion below. A physical
    # clockwise sensor rotation produces a counter-clockwise image rotation.
    front_camera_position: tuple[float, float, float] = (0.060, 0.230, 0.595)
    front_camera_quaternion: tuple[float, float, float, float] = (
        0.70710678,
        0.0,
        0.0,
        -0.70710678,
    )
    # The camera specification reports DFOV=100 degrees, EFL=2.8 mm and
    # F/2.2. Comparison with the training frames shows that the real data uses
    # the central 60% in both image dimensions (20% removed from every edge).
    # Multiplying the 4:3 film back by 0.6 reproduces that crop while rendering
    # directly at 640x480. Effective FOV: H=59.54, V=46.44, D=71.13 degrees.
    front_focal_length: float = 2.8
    front_horizontal_aperture: float = 3.2034336
    front_f_number: float = 2.2

    # Keep the official LeIsaac wrist-camera mount until its physical transform
    # is measured. It remains necessary because the real ACT dataset has two
    # visual inputs: front and wrist.
    wrist_camera_position: tuple[float, float, float] = (-0.001, 0.100, -0.040)
    wrist_camera_quaternion: tuple[float, float, float, float] = (
        -0.404379,
        -0.912179,
        -0.0451242,
        0.0486914,
    )
    # Starting from the reported cropped 4:3 DFOV=110 degrees, apply the same
    # central 60% training-frame crop as the front camera. Effective FOV:
    # H=68.86, V=54.42, D=81.19 degrees. Only the aperture/focal ratio matters
    # to this pinhole model; the physical wrist sensor size is not known.
    wrist_focal_length: float = 16.117902
    wrist_horizontal_aperture: float = 22.098

    camera_width: int = 640
    camera_height: int = 480
    camera_fps: float = 30.0

    # Lighting remains tunable because camera exposure/white balance have not
    # yet been calibrated.
    dome_color: tuple[float, float, float] = (0.86, 0.89, 0.84)
    dome_intensity: float = 950.0
    key_color: tuple[float, float, float] = (1.0, 0.93, 0.82)
    # Positional key light: 50 cm to the robot's right, directly above its
    # Y-origin, and 1.2 m above the tabletop. A small sphere gives a practical
    # soft source whose location is visible through the cast-shadow direction.
    key_position: tuple[float, float, float] = (0.50, 0.0, 1.20)
    key_radius: float = 0.12
    key_intensity: float = 30000.0
    fill_color: tuple[float, float, float] = (0.78, 0.88, 1.0)
    fill_intensity: float = 450.0
    fill_quaternion: tuple[float, float, float, float] = (
        0.92388,
        -0.27060,
        0.27060,
        0.0,
    )


SCENE = SceneParameters()
