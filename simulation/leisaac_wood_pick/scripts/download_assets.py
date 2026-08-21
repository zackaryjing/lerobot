"""Download the official LeIsaac SO-101 USD asset from Hugging Face."""

import argparse
import shutil
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "assets" / "leisaac" / "robots" / "so101_follower.usd"
ASSET_URL = (
    "https://huggingface.co/LightwheelAI/leisaac_env/resolve/main/"
    "assets/robots/so101_follower.usd?download=true"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--force", action="store_true")
args = parser.parse_args()


def main() -> None:
    output = args.output.resolve()
    if output.exists() and not args.force:
        print(f"Asset already exists: {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".usd.part")
    try:
        print(f"Downloading official SO-101 asset to {output}")
        with urllib.request.urlopen(ASSET_URL) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if temporary.stat().st_size < 10_000_000:
            raise RuntimeError(
                f"Downloaded file is unexpectedly small ({temporary.stat().st_size} bytes)"
            )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Downloaded {output} ({output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
