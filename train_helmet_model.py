import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
YOLO_TRAIN_SCRIPT = ROOT / "yolov5" / "train.py"
DEFAULT_DATASET_CONFIG = ROOT / "configs" / "helmet_dataset.yaml"
DEFAULT_STARTING_WEIGHTS = ROOT / "yolov5s.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv5 helmet detector using your local helmet dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET_CONFIG,
        help="Path to your YOLO dataset YAML file.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_STARTING_WEIGHTS,
        help="Starting YOLO weights for transfer learning.",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--img", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--device", default="", help="Training device. Example: cpu, 0, or 0,1")
    parser.add_argument("--name", default="helmet_detector", help="Run name under runs/train/.")
    parser.add_argument(
        "--project",
        type=Path,
        default=ROOT / "runs" / "train",
        help="Directory where YOLOv5 training outputs will be written.",
    )
    return parser.parse_args()


def validate_dataset_yaml(dataset_yaml_path):
    if not dataset_yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml_path}\n"
            "Update configs/helmet_dataset.yaml or pass --data with the correct dataset YAML path."
        )

    with dataset_yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    required_keys = {"train", "val", "nc", "names"}
    missing_keys = sorted(required_keys - data.keys())
    if missing_keys:
        raise ValueError(f"Dataset YAML is missing required keys: {', '.join(missing_keys)}")

    if not isinstance(data["names"], list) or not data["names"]:
        raise ValueError("Dataset YAML field 'names' must be a non-empty list of class names.")

    if int(data["nc"]) != len(data["names"]):
        raise ValueError("Dataset YAML field 'nc' must match the number of entries in 'names'.")


def main():
    args = parse_args()
    validate_dataset_yaml(args.data)

    if not YOLO_TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"YOLOv5 train.py not found: {YOLO_TRAIN_SCRIPT}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Starting weights not found: {args.weights}")

    args.project.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(YOLO_TRAIN_SCRIPT),
        "--data",
        str(args.data),
        "--weights",
        str(args.weights),
        "--epochs",
        str(args.epochs),
        "--img",
        str(args.img),
        "--batch",
        str(args.batch),
        "--project",
        str(args.project),
        "--name",
        args.name,
    ]
    if args.device:
        command.extend(["--device", args.device])

    print("Starting helmet detector training...")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=ROOT)

    best_weights = args.project / args.name / "weights" / "best.pt"
    print("\nTraining finished.")
    print(f"Best weights should be available at: {best_weights}")
    print("Copy or rename that file to weights/helmet_best.pt for live camera use.")


if __name__ == "__main__":
    main()
