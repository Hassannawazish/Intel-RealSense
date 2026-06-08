import argparse
import csv
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_CONFIG = ROOT / "configs" / "safety_vest_dataset.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "generated_datasets" / "safety_vest_yolo"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the current CSV-based safety vest dataset into YOLO image/label format."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_SOURCE_CONFIG,
        help="Path to the source safety vest dataset YAML config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the YOLO-formatted dataset will be generated.",
    )
    return parser.parse_args()


def load_yaml(yaml_path):
    with yaml_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_split_dir(dataset_root, split_value):
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path
    return dataset_root / split_path


def sanitize_class_name(name):
    return str(name).strip()


def build_class_map(class_names):
    return {sanitize_class_name(name): index for index, name in enumerate(class_names)}


def ensure_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def collect_image_map(split_dir):
    image_map = {}
    for image_path in split_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            image_map[image_path.name] = image_path
    return image_map


def convert_bbox_to_yolo(row, image_width, image_height):
    xmin = float(row["xmin"])
    ymin = float(row["ymin"])
    xmax = float(row["xmax"])
    ymax = float(row["ymax"])

    box_width = max(0.0, xmax - xmin)
    box_height = max(0.0, ymax - ymin)
    center_x = xmin + box_width / 2.0
    center_y = ymin + box_height / 2.0

    return (
        center_x / image_width,
        center_y / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def write_label_file(label_path, annotations):
    with label_path.open("w", encoding="utf-8") as handle:
        for class_id, bbox in annotations:
            handle.write(
                f"{class_id} "
                f"{bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
            )


def convert_split(split_name, split_dir, output_root, class_map):
    annotations_csv = split_dir / "_annotations.csv"
    if not annotations_csv.exists():
        raise FileNotFoundError(f"Missing annotations file for split '{split_name}': {annotations_csv}")

    image_map = collect_image_map(split_dir)
    if not image_map:
        raise FileNotFoundError(f"No images found in split '{split_name}': {split_dir}")

    output_images_dir = output_root / "images" / split_name
    output_labels_dir = output_root / "labels" / split_name
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    grouped_annotations = {}
    with annotations_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Annotations file {annotations_csv} is missing columns: {', '.join(sorted(missing_columns))}"
            )

        for row in reader:
            filename = row["filename"]
            image_path = image_map.get(filename)
            if image_path is None:
                continue

            class_name = sanitize_class_name(row["class"])
            if class_name not in class_map:
                raise ValueError(f"Unknown class '{class_name}' found in {annotations_csv}")

            image_width = float(row["width"])
            image_height = float(row["height"])
            if image_width <= 0 or image_height <= 0:
                continue

            bbox = convert_bbox_to_yolo(row, image_width, image_height)
            grouped_annotations.setdefault(filename, []).append((class_map[class_name], bbox))

    copied_images = 0
    written_labels = 0

    for filename, image_path in image_map.items():
        shutil.copy2(image_path, output_images_dir / filename)
        copied_images += 1

        annotations = grouped_annotations.get(filename, [])
        label_path = output_labels_dir / f"{Path(filename).stem}.txt"
        write_label_file(label_path, annotations)
        if annotations:
            written_labels += 1

    return {
        "split_name": split_name,
        "images_copied": copied_images,
        "images_with_labels": written_labels,
        "images_without_labels": copied_images - written_labels,
    }


def convert_dataset(source_yaml_path, output_root):
    source_data = load_yaml(source_yaml_path)
    dataset_root = Path(source_data["path"])
    class_names = source_data["names"]
    class_map = build_class_map(class_names)

    ensure_clean_dir(output_root)

    split_results = []
    split_key_to_output = {"train": "train", "val": "val", "test": "test"}
    for split_key, output_name in split_key_to_output.items():
        if split_key not in source_data:
            continue
        source_split_dir = resolve_split_dir(dataset_root, source_data[split_key])
        result = convert_split(output_name, source_split_dir, output_root, class_map)
        split_results.append(result)

    generated_yaml = {
        "path": str(output_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": int(source_data["nc"]),
        "names": class_names,
    }
    if any(result["split_name"] == "test" for result in split_results):
        generated_yaml["test"] = "images/test"

    generated_yaml_path = output_root / "dataset.yaml"
    with generated_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(generated_yaml, handle, sort_keys=False)

    return generated_yaml_path, split_results


def main():
    args = parse_args()
    generated_yaml_path, split_results = convert_dataset(args.data, args.output)

    print(f"Generated YOLO dataset config: {generated_yaml_path}")
    for result in split_results:
        print(
            f"{result['split_name']}: copied {result['images_copied']} image(s), "
            f"wrote labels for {result['images_with_labels']} image(s), "
            f"{result['images_without_labels']} image(s) have no objects."
        )


if __name__ == "__main__":
    main()
