from helmet_detection import get_intersection_area


DEFAULT_SAFETY_VEST_CLASS_NAMES = {"safety vest", "vest", "safety-vest", "high visibility vest"}


def detect_safety_vest_boxes(model, frame, vest_class_names=None, image_size=None):
    """Run a dedicated safety vest model and return only safety-vest-class detections."""
    if model is None:
        return []

    from helmet_detection import detect_helmet_boxes

    return detect_helmet_boxes(
        model,
        frame,
        helmet_class_names=vest_class_names or DEFAULT_SAFETY_VEST_CLASS_NAMES,
        image_size=image_size,
    )


def get_person_torso_region(person_box):
    """Approximate the torso area of a detected person box."""
    x1, y1, x2, y2 = person_box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    torso_left = x1 + int(width * 0.15)
    torso_right = x2 - int(width * 0.15)
    torso_top = y1 + int(height * 0.22)
    torso_bottom = y1 + int(height * 0.72)
    return torso_left, torso_top, torso_right, torso_bottom


def match_safety_vest_to_person(person_box, vest_detections):
    """Return the best safety vest detection that overlaps the torso of a person box."""
    x1, y1, x2, y2 = person_box
    torso_region = get_person_torso_region(person_box)
    best_match = None

    for vest in vest_detections:
        vx1, vy1, vx2, vy2 = vest["box"]
        center_x = (vx1 + vx2) // 2
        center_y = (vy1 + vy2) // 2

        center_inside_person = x1 <= center_x <= x2 and y1 <= center_y <= y2
        center_in_torso = center_inside_person and (torso_region[1] <= center_y <= torso_region[3])
        torso_overlap = get_intersection_area(torso_region, vest["box"])

        if not center_in_torso and torso_overlap <= 0:
            continue

        if best_match is None or vest["score"] > best_match["score"]:
            best_match = vest

    return best_match


def append_safety_vest_status(label, vest_match):
    suffix = f"Vest {vest_match['score']:.2f}" if vest_match else "No Vest"
    return f"{label} | {suffix}"
