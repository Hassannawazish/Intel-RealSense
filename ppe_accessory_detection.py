from helmet_detection import (
    detect_helmet_boxes,
    get_intersection_area,
)


DEFAULT_GLOVE_CLASS_NAMES = {
    "glove",
    "gloves",
    "safety glove",
    "safety gloves",
    "protective glove",
    "protective gloves",
    "hand glove",
    "hand gloves",
    "work glove",
    "work gloves",
}

DEFAULT_GOGGLE_CLASS_NAMES = {
    "goggle",
    "goggles",
    "safety goggle",
    "safety goggles",
    "protective goggle",
    "protective goggles",
    "glass",
    "glasses",
    "safety glass",
    "safety glasses",
    "eyewear",
    "protective eyewear",
}


def detect_glove_boxes(model, frame, glove_class_names=None, image_size=None):
    """Run the shared PPE model and return only glove-class detections."""
    if model is None:
        return []

    return detect_helmet_boxes(
        model,
        frame,
        helmet_class_names=glove_class_names or DEFAULT_GLOVE_CLASS_NAMES,
        image_size=image_size,
    )


def detect_goggle_boxes(model, frame, goggle_class_names=None, image_size=None):
    """Run the shared PPE model and return only goggle/glasses-class detections."""
    if model is None:
        return []

    return detect_helmet_boxes(
        model,
        frame,
        helmet_class_names=goggle_class_names or DEFAULT_GOGGLE_CLASS_NAMES,
        image_size=image_size,
    )


def get_person_hand_regions(person_box):
    """Approximate left and right hand zones near the lower outer body."""
    x1, y1, x2, y2 = person_box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    hand_top = y1 + int(height * 0.38)
    hand_bottom = y1 + int(height * 0.95)
    left_hand = (x1, hand_top, x1 + int(width * 0.35), hand_bottom)
    right_hand = (x2 - int(width * 0.35), hand_top, x2, hand_bottom)
    return left_hand, right_hand


def get_person_eye_region(person_box):
    """Approximate the upper face region where goggles/glasses usually appear."""
    x1, y1, x2, y2 = person_box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    eye_left = x1 + int(width * 0.18)
    eye_right = x2 - int(width * 0.18)
    eye_top = y1 + int(height * 0.05)
    eye_bottom = y1 + int(height * 0.32)
    return eye_left, eye_top, eye_right, eye_bottom


def match_gloves_to_person(person_box, glove_detections):
    """Return the best glove detection that overlaps either hand region."""
    x1, y1, x2, y2 = person_box
    left_hand, right_hand = get_person_hand_regions(person_box)
    best_match = None

    for glove in glove_detections:
        gx1, gy1, gx2, gy2 = glove["box"]
        center_x = (gx1 + gx2) // 2
        center_y = (gy1 + gy2) // 2

        center_inside_person = x1 <= center_x <= x2 and y1 <= center_y <= y2
        in_left_hand = left_hand[0] <= center_x <= left_hand[2] and left_hand[1] <= center_y <= left_hand[3]
        in_right_hand = right_hand[0] <= center_x <= right_hand[2] and right_hand[1] <= center_y <= right_hand[3]
        overlap = max(
            get_intersection_area(left_hand, glove["box"]),
            get_intersection_area(right_hand, glove["box"]),
        )

        if not center_inside_person and overlap <= 0:
            continue
        if not in_left_hand and not in_right_hand and overlap <= 0:
            continue

        if best_match is None or glove["score"] > best_match["score"]:
            best_match = glove

    return best_match


def match_goggles_to_person(person_box, goggle_detections):
    """Return the best goggles/glasses detection that overlaps the face region."""
    x1, y1, x2, y2 = person_box
    eye_region = get_person_eye_region(person_box)
    best_match = None

    for goggle in goggle_detections:
        gx1, gy1, gx2, gy2 = goggle["box"]
        center_x = (gx1 + gx2) // 2
        center_y = (gy1 + gy2) // 2

        center_inside_person = x1 <= center_x <= x2 and y1 <= center_y <= y2
        center_in_face = center_inside_person and eye_region[1] <= center_y <= eye_region[3]
        overlap = get_intersection_area(eye_region, goggle["box"])

        if not center_in_face and overlap <= 0:
            continue

        if best_match is None or goggle["score"] > best_match["score"]:
            best_match = goggle

    return best_match


def append_gloves_status(label, glove_match):
    suffix = f"Gloves {glove_match['score']:.2f}" if glove_match else "No Gloves"
    return f"{label} | {suffix}"


def append_goggles_status(label, goggle_match):
    suffix = f"Goggles {goggle_match['score']:.2f}" if goggle_match else "No Goggles"
    return f"{label} | {suffix}"
