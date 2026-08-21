"""Convert the measured wooden-target STL to a centred, metre-scale USD."""

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parents[1]
DEFAULT_INPUT = REPOSITORY_DIR / "target" / "wooden_target.stl"
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "source"
    / "leisaac_wood_pick"
    / "leisaac_wood_pick"
    / "assets"
    / "wooden_target.usd"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--force", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg


def main() -> None:
    input_path = args_cli.input.resolve()
    output_path = args_cli.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Wooden-target mesh not found: {input_path}")
    if output_path.exists() and not args_cli.force:
        print(f"Asset already exists: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Measured binary STL bounds: min=(0,0,0), max=(11.5,11.5,80) mm.
    # Centre it before scaling. Orientation is applied in the scene config so
    # object yaw can later be randomized naturally around world Z.
    converter = MeshConverter(
        MeshConverterCfg(
            asset_path=str(input_path),
            usd_dir=str(output_path.parent),
            usd_file_name=output_path.name,
            force_usd_conversion=True,
            # There is only one target and its material is overridden by the
            # scene. Keeping it non-instanceable makes that override reliable.
            make_instanceable=False,
            # MeshConverter applies this transform after ``scale``, so the
            # translation is expressed in stage units (metres), not STL mm.
            translation=(-0.00575, -0.00575, -0.0400),
            rotation=(1.0, 0.0, 0.0, 0.0),
            scale=(0.001, 0.001, 0.001),
            mass_props=schemas_cfg.MassPropertiesCfg(mass=0.030),
            rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
            collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
            mesh_collision_props=schemas_cfg.ConvexHullPropertiesCfg(),
        )
    )
    print(f"Generated {converter.usd_path}")
    print(f"Expected dimensions: 0.0115 x 0.0115 x 0.0800 m")
    print(f"Long-axis scene rotation: +{math.degrees(math.pi / 2):.0f} deg about Y")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
