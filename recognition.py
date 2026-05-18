import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(os.cpu_count() or 8))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["NUMEXPR_MAX_THREADS"])

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

try:
    from facial_recognition import add_person, recognize_image, remove_face_database
except ModuleNotFoundError as exc:
    missing_package = exc.name or "required package"
    raise ModuleNotFoundError(
        f"Missing dependency: {missing_package}. Install the face recognition runtime with "
        "`python -m pip install onnxruntime facial_recognition` and run `facial_recognition setup`."
    ) from exc


ROOT = Path(__file__).resolve().parent
KNOWN_FACE_DIR = ROOT / "known_faces" / "hassan"
KNOWN_FACE_NAME = "Hassan"
FACE_MATCH_THRESHOLD = 0.3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SCREEN_DEVICE_CLASSES = {"cell phone", "tv", "laptop", "tablet", "monitor"}


def prepare_known_faces(image_dir):
    """Reset the face database and add Hassan images from the configured folder."""
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Known face folder not found: {image_dir}\n"
            "Create the folder and place clear front-facing photos inside it so recognition can identify Hassan."
        )

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No supported images were found in {image_dir}.\n"
            "Add JPG, JPEG, PNG, or BMP images for Hassan."
        )

    remove_face_database()
    for image_path in image_paths:
        add_person(KNOWN_FACE_NAME, str(image_path))

    print(f"Loaded {len(image_paths)} reference image(s) for {KNOWN_FACE_NAME}.")


def recognize_faces(frame):
    """Return recognized face names and locations for the current frame."""
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

        recognized_faces.append(
            {
                "name": name,
                "box": (left, top, right, bottom),
                "is_spoof": False,
            }
        )

    return recognized_faces


def get_intersection_area(box_a, box_b):
    """Return the overlap area between two bounding boxes."""
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    if right <= left or bottom <= top:
        return 0
    return (right - left) * (bottom - top)


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


def get_person_label(person_box, recognized_faces):
    """Attach a face label to a YOLO person box, blocking screen-based spoof attempts."""
    x1, y1, x2, y2 = person_box
    for face in recognized_faces:
        left, top, right, bottom = face["box"]
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            if face["is_spoof"]:
                return "Threat"
            return face["name"]
    return "Unknown Person"


def screen_has_spoof_face(screen_box, recognized_faces):
    """Return True when a spoofed face overlaps a screen-like object box."""
    for face in recognized_faces:
        if face["is_spoof"] and get_intersection_area(screen_box, face["box"]) > 0:
            return True
    return False


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using torch device: {device}")

prepare_known_faces(KNOWN_FACE_DIR)

# Load the local YOLOv5 model from the cloned repository.
model = torch.hub.load("./yolov5", "yolov5s", source="local")
model.to(device)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

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
        screen_boxes = extract_screen_boxes(detections, names)
        flag_spoof_faces(recognized_faces, screen_boxes)

        annotated_frame = frame.copy()

        for x1, y1, x2, y2, conf, cls in detections:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            class_name = names[int(cls)]
            current_box = (x1, y1, x2, y2)

            if class_name == "person":
                label = get_person_label(current_box, recognized_faces)
                if label == KNOWN_FACE_NAME:
                    color = (0, 200, 0)
                elif label == "Threat":
                    color = (0, 0, 255)
                else:
                    color = (0, 165, 255)
            elif class_name in SCREEN_DEVICE_CLASSES and screen_has_spoof_face(current_box, recognized_faces):
                label = "Threat"
                color = (0, 0, 255)
            else:
                label = f"{class_name} {conf:.2f}"
                color = (255, 0, 0)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
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

        cv2.imshow("YOLOv5 RealSense Recognition", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
