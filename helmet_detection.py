from pathlib import Path

import torch


DEFAULT_HELMET_CLASS_NAMES = {"helmet", "hardhat", "hard hat", "safety helmet"}


def normalize_class_name(value):
    return str(value).strip().lower().replace("-", " ")


def load_yolo_model(repo_dir, weights_path=None, device="cpu", model_name=None):
    """Load a YOLOv5 model from the local repository using either built-in or custom weights."""
    repo_dir = Path(repo_dir)
    if weights_path:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights were not found: {weights_path}")
        model = torch.hub.load(str(repo_dir), "custom", path=str(weights_path), source="local")
    elif model_name:
        model = torch.hub.load(str(repo_dir), model_name, source="local")
    else:
        raise ValueError("Either weights_path or model_name must be provided to load_yolo_model().")

    model.to(device)
    return model


def get_intersection_area(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    if right <= left or bottom <= top:
        return 0
    return (right - left) * (bottom - top)


def find_class_ids(names, target_names):
    normalized_targets = {normalize_class_name(name) for name in target_names}
    matches = set()
    if isinstance(names, dict):
        iterable = names.items()
    else:
        iterable = enumerate(names)

    for class_id, class_name in iterable:
        if normalize_class_name(class_name) in normalized_targets:
            matches.add(int(class_id))
    return matches


def detect_helmet_boxes(model, frame, helmet_class_names=None, image_size=None):
    """Run a dedicated helmet model and return only helmet-class detections."""
    helmet_class_names = helmet_class_names or DEFAULT_HELMET_CLASS_NAMES
    with torch.inference_mode():
        results = model(frame, size=image_size) if image_size else model(frame)

    detections = results.xyxy[0].cpu().numpy()
    names = results.names
    helmet_class_ids = find_class_ids(names, helmet_class_names)

    helmets = []
    for x1, y1, x2, y2, conf, cls in detections:
        if int(cls) not in helmet_class_ids:
            continue
        helmets.append(
            {
                "box": tuple(map(int, (x1, y1, x2, y2))),
                "score": float(conf),
                "class_name": names[int(cls)],
            }
        )
    return helmets


def get_person_head_region(person_box):
    """Approximate the upper head area of a detected person box."""
    x1, y1, x2, y2 = person_box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    head_left = x1 + int(width * 0.12)
    head_right = x2 - int(width * 0.12)
    head_top = y1
    head_bottom = y1 + int(height * 0.38)
    return head_left, head_top, head_right, head_bottom


def match_helmet_to_person(person_box, helmet_detections):
    """Return the best helmet detection that overlaps the upper part of a person box."""
    x1, y1, x2, y2 = person_box
    head_region = get_person_head_region(person_box)
    best_match = None

    for helmet in helmet_detections:
        hx1, hy1, hx2, hy2 = helmet["box"]
        center_x = (hx1 + hx2) // 2
        center_y = (hy1 + hy2) // 2

        center_inside_person = x1 <= center_x <= x2 and y1 <= center_y <= y2
        center_in_upper_body = center_inside_person and center_y <= y1 + int((y2 - y1) * 0.45)
        head_overlap = get_intersection_area(head_region, helmet["box"])

        if not center_in_upper_body and head_overlap <= 0:
            continue

        if best_match is None or helmet["score"] > best_match["score"]:
            best_match = helmet

    return best_match


def append_helmet_status(label, helmet_match):
    suffix = "Helmet" if helmet_match else "No Helmet"
    return f"{label} | {suffix}"
