import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)

    # Create directories
    night_dir = data_dir / "night"
    day_dir = data_dir / "day"

    night_dir.mkdir(parents=True, exist_ok=True)
    day_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    for file in os.listdir(data_dir):
        if file in {"night", "day"}:
            continue

        if os.path.isdir(data_dir / file):
            dir_path = data_dir / file
            for image in tqdm(os.listdir(dir_path), desc=f"Collecting {file}"):
                new_image_name = f"{file}_{image}"
                if "night" in file:
                    shutil.copyfile(dir_path / image, night_dir / new_image_name)
                else:
                    shutil.copyfile(dir_path / image, day_dir / new_image_name)


if __name__ == "__main__":
    main()
