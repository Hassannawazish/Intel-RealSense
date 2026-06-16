import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(os.cpu_count() or 8))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["NUMEXPR_MAX_THREADS"])

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

from display_utils import build_person_ppe_lines, draw_label_block
from helmet_detection import (
    append_helmet_status,
    detect_helmet_boxes,
    get_intersection_area,
    load_yolo_model,
    match_helmet_to_person,
)
from ppe_accessory_detection import (
    append_gloves_status,
    append_goggles_status,
    detect_glove_boxes,
    detect_goggle_boxes,
    match_gloves_to_person,
    match_goggles_to_person,
)
from safety_vest_detection import (
    append_safety_vest_status,
    detect_safety_vest_boxes,
    match_safety_vest_to_person,
)

try:
    from facial_recognition import add_person, recognize_image, remove_face_database
except ModuleNotFoundError as exc:
    missing_package = exc.name or "required package"
    raise ModuleNotFoundError(
        f"Missing dependency: {missing_package}. Install the face recognition runtime with "
        "`python -m pip install onnxruntime facial_recognition` and run `facial_recognition setup`."
    ) from exc


ROOT = Path(__file__).resolve().parent
KNOWN_FACES_ROOT = ROOT / "known_faces"
FACE_MATCH_THRESHOLD = 0.3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SCREEN_DEVICE_CLASSES = {"cell phone", "tv", "laptop", "tablet", "monitor"}
YOLO_REPO_DIR = ROOT / "yolov5"
OBJECT_MODEL_NAME = "yolov5s"
HELMET_WEIGHTS = ROOT / "weights" / "helmet_best.pt"
HELMET_IMAGE_SIZE = 416
SAFETY_VEST_WEIGHTS = ROOT / "weights" / "safety_vest_best.pt"
SAFETY_VEST_IMAGE_SIZE = 416
ACCESSORY_WEIGHTS = ROOT / "external_models" / "epoch30.pt"
ACCESSORY_IMAGE_SIZE = 416
WINDOW_NAME = "YOLOv5 RealSense Recognition"
DISPLAY_SIZE = (1440, 960)


def format_person_name(folder_name):
    """Convert a folder name like 'rana' or 'abdul_rahman' into a display label."""
    return folder_name.replace("_", " ").replace("-", " ").title()


def prepare_known_faces(known_faces_root):
    """Reset the face database and add every known person found under known_faces/."""
    if not known_faces_root.exists():
        raise FileNotFoundError(
            f"Known faces root folder not found: {known_faces_root}\n"
            "Create person subfolders inside known_faces and place clear front-facing photos in each one."
        )

    person_dirs = sorted(path for path in known_faces_root.iterdir() if path.is_dir())
    if not person_dirs:
        raise FileNotFoundError(
            f"No person folders were found in {known_faces_root}.\n"
            "Add folders like known_faces/hassan, known_faces/rana, known_faces/khan, or known_faces/fernando."
        )

    remove_face_database()
    loaded_people = {}
    skipped_dirs = []

    for person_dir in person_dirs:
        image_paths = sorted(
            path for path in person_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            skipped_dirs.append(person_dir.name)
            continue

        person_name = format_person_name(person_dir.name)
        for image_path in image_paths:
            add_person(person_name, str(image_path))
        loaded_people[person_name] = len(image_paths)

    if not loaded_people:
        raise FileNotFoundError(
            f"No supported images were found in any subfolder of {known_faces_root}.\n"
            "Add JPG, JPEG, PNG, or BMP images for each person you want to recognize."
        )

    for person_name, image_count in loaded_people.items():
        print(f"Loaded {image_count} reference image(s) for {person_name}.")
    if skipped_dirs:
        print(f"Skipped empty known face folders: {', '.join(skipped_dirs)}")


def recognize_faces(frame):
    """Return recognized face metadata and locations for the current frame."""
    results = recognize_image(frame, save_output=False, threshold=FACE_MATCH_THRESHOLD)

    recognized_faces = []
    frame_height, frame_width = frame.shape[:2]
    for result in results:
        x, y, w, h = result["box"]
        left = int(x * frame_width)
        top = int(y * frame_height)
        right = int((x + w) * frame_width)
        bottom = int((y + h) * frame_height)

        name = result["name"]
        if name == "Unknown":
            name = "Unknown Person"
        score = float(result.get("score", 0.0))

        recognized_faces.append(
            {
                "name": name,
                "score": score,
                "box": (left, top, right, bottom),
                "is_spoof": False,
            }
        )

    return recognized_faces


def extract_screen_boxes(detections, names):
    """Collect boxes for phones and other screen-like devices."""
    screen_boxes = []
    for x1, y1, x2, y2, conf, cls in detections:
        class_name = names[int(cls)]
        if class_name in SCREEN_DEVICE_CLASSES:
            screen_boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return screen_boxes


def flag_spoof_faces(recognized_faces, screen_boxes):
    """Mark recognized faces as spoofed when they appear inside a device screen region."""
    for face in recognized_faces:
        face_box = face["box"]
        face_area = max(1, (face_box[2] - face_box[0]) * (face_box[3] - face_box[1]))
        center_x = (face_box[0] + face_box[2]) // 2
        center_y = (face_box[1] + face_box[3]) // 2

        for screen_box in screen_boxes:
            center_inside_screen = (
                screen_box[0] <= center_x <= screen_box[2] and screen_box[1] <= center_y <= screen_box[3]
            )
            overlap_ratio = get_intersection_area(face_box, screen_box) / face_area
            if center_inside_screen or overlap_ratio >= 0.35:
                face["is_spoof"] = True
                break


def get_person_match(person_box, recognized_faces):
    """Return the matched face metadata for a YOLO person box."""
    x1, y1, x2, y2 = person_box
    for face in recognized_faces:
        left, top, right, bottom = face["box"]
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            return face
    return None


def format_person_label(face):
    """Build the on-screen label for a recognized face match."""
    if face is None:
        return "Unknown Person"
    if face["is_spoof"]:
        return "Threat"
    if face["name"] == "Unknown Person":
        return face["name"]
    return f'{face["name"]}: {face["score"]:.2f}'


def screen_has_spoof_face(screen_box, recognized_faces):
    """Return True when a spoofed face overlaps a screen-like object box."""
    for face in recognized_faces:
        if face["is_spoof"] and get_intersection_area(screen_box, face["box"]) > 0:
            return True
    return False


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using torch device: {device}")

prepare_known_faces(KNOWN_FACES_ROOT)

# Load the local YOLOv5 model from the cloned repository.
model = load_yolo_model(YOLO_REPO_DIR, device=device, model_name=OBJECT_MODEL_NAME)
vest_model = None
if SAFETY_VEST_WEIGHTS.exists():
    vest_model = load_yolo_model(YOLO_REPO_DIR, weights_path=SAFETY_VEST_WEIGHTS, device=device)
    print(f"Loaded safety vest detector weights: {SAFETY_VEST_WEIGHTS}")
else:
    print(
        f"Safety vest detector weights not found at {SAFETY_VEST_WEIGHTS}. "
        "Train a safety vest model first if you want live safety vest detection."
    )

accessory_model = None
if ACCESSORY_WEIGHTS.exists():
    accessory_model = load_yolo_model(YOLO_REPO_DIR, weights_path=ACCESSORY_WEIGHTS, device=device)
    print(f"Loaded helmet/gloves/goggles detector weights: {ACCESSORY_WEIGHTS}")
else:
    print(
        f"Helmet/gloves/goggles detector weights not found at {ACCESSORY_WEIGHTS}. "
        "Place your trained shared PPE model there if you want live helmet, glove, and goggle detection."
    )

helmet_model = accessory_model
if helmet_model is not None:
    print(f"Using shared PPE model for helmet detection: {ACCESSORY_WEIGHTS}")
elif HELMET_WEIGHTS.exists():
    helmet_model = load_yolo_model(YOLO_REPO_DIR, weights_path=HELMET_WEIGHTS, device=device)
    print(f"Shared PPE model unavailable, fell back to helmet detector weights: {HELMET_WEIGHTS}")
else:
    print(
        f"No helmet-capable model found at {ACCESSORY_WEIGHTS} or {HELMET_WEIGHTS}. "
        "Place a trained model there if you want live helmet detection."
    )

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, *DISPLAY_SIZE)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        recognized_faces = recognize_faces(frame)

        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()
        names = results.names
        helmet_detections = detect_helmet_boxes(helmet_model, frame, image_size=HELMET_IMAGE_SIZE) if helmet_model else []
        vest_detections = (
            detect_safety_vest_boxes(vest_model, frame, image_size=SAFETY_VEST_IMAGE_SIZE) if vest_model else []
        )
        glove_detections = (
            detect_glove_boxes(accessory_model, frame, image_size=ACCESSORY_IMAGE_SIZE) if accessory_model else []
        )
        goggle_detections = (
            detect_goggle_boxes(accessory_model, frame, image_size=ACCESSORY_IMAGE_SIZE) if accessory_model else []
        )
        screen_boxes = extract_screen_boxes(detections, names)
        flag_spoof_faces(recognized_faces, screen_boxes)

        annotated_frame = frame.copy()

        for helmet in helmet_detections:
            hx1, hy1, hx2, hy2 = helmet["box"]
            helmet_label = f'{helmet["class_name"]} {helmet["score"]:.2f}'
            cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (255, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                helmet_label,
                (hx1, max(30, hy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        for vest in vest_detections:
            vx1, vy1, vx2, vy2 = vest["box"]
            vest_label = f'{vest["class_name"]} {vest["score"]:.2f}'
            cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (255, 0, 255), 2)
            cv2.putText(
                annotated_frame,
                vest_label,
                (vx1, max(30, vy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

        for glove in glove_detections:
            gx1, gy1, gx2, gy2 = glove["box"]
            glove_label = f'{glove["class_name"]} {glove["score"]:.2f}'
            cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
            cv2.putText(
                annotated_frame,
                glove_label,
                (gx1, max(30, gy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for goggle in goggle_detections:
            gx1, gy1, gx2, gy2 = goggle["box"]
            goggle_label = f'{goggle["class_name"]} {goggle["score"]:.2f}'
            cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (0, 128, 255), 2)
            cv2.putText(
                annotated_frame,
                goggle_label,
                (gx1, max(30, gy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 128, 255),
                2,
                cv2.LINE_AA,
            )

        for x1, y1, x2, y2, conf, cls in detections:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            class_name = names[int(cls)]
            current_box = (x1, y1, x2, y2)

            if class_name == "person":
                face_match = get_person_match(current_box, recognized_faces)
                base_label = format_person_label(face_match)
                helmet_match = match_helmet_to_person(current_box, helmet_detections)
                vest_match = match_safety_vest_to_person(current_box, vest_detections)
                glove_match = match_gloves_to_person(current_box, glove_detections)
                goggle_match = match_goggles_to_person(current_box, goggle_detections)
                label_lines = (
                    [base_label]
                    if base_label == "Threat"
                    else build_person_ppe_lines(base_label, helmet_match, vest_match, glove_match, goggle_match)
                )
                if base_label == "Threat":
                    label = base_label
                else:
                    label = append_goggles_status(
                        append_gloves_status(
                            append_safety_vest_status(append_helmet_status(base_label, helmet_match), vest_match),
                            glove_match,
                        ),
                        goggle_match,
                    )
                if base_label == "Threat":
                    color = (0, 0, 255)
                elif helmet_match and vest_match and glove_match and goggle_match:
                    color = (0, 200, 0)
                else:
                    color = (0, 165, 255)
            elif class_name in SCREEN_DEVICE_CLASSES and screen_has_spoof_face(current_box, recognized_faces):
                label = "Threat"
                color = (0, 0, 255)
            else:
                label = f"{class_name} {conf:.2f}"
                color = (255, 0, 0)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            if class_name == "person":
                draw_label_block(annotated_frame, label_lines, (x1, y1), color, font_scale=0.6, thickness=2)
            else:
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        display_frame = cv2.resize(annotated_frame, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(WINDOW_NAME, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
