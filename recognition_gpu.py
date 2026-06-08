import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(os.cpu_count() or 8))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["NUMEXPR_MAX_THREADS"])

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

from helmet_detection import (
    append_helmet_status,
    detect_helmet_boxes,
    get_intersection_area,
    load_yolo_model,
    match_helmet_to_person,
)
from safety_vest_detection import (
    append_safety_vest_status,
    detect_safety_vest_boxes,
    match_safety_vest_to_person,
)

try:
    import onnxruntime as ort
except ModuleNotFoundError:
    ort = None

try:
    from facial_recognition import add_person, recognize_image, remove_face_database
except ModuleNotFoundError as exc:
    missing_package = exc.name or "required package"
    raise ModuleNotFoundError(
        f"Missing dependency: {missing_package}. Install the GPU-friendly face recognition runtime with "
        "`python -m pip install onnxruntime-gpu facial_recognition` and run `facial_recognition setup`."
    ) from exc


ROOT = Path(__file__).resolve().parent
KNOWN_FACES_ROOT = ROOT / "known_faces"
FACE_MATCH_THRESHOLD = 0.3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SCREEN_DEVICE_CLASSES = {"cell phone", "tv", "laptop", "tablet", "monitor"}
YOLO_MODEL_NAME = "yolov5n"
YOLO_IMAGE_SIZE = 416
YOLO_REPO_DIR = ROOT / "yolov5"
HELMET_WEIGHTS = ROOT / "weights" / "helmet_best.pt"
HELMET_IMAGE_SIZE = 416
SAFETY_VEST_WEIGHTS = ROOT / "weights" / "safety_vest_best.pt"
SAFETY_VEST_IMAGE_SIZE = 416
FACE_RECOGNITION_EVERY_N_FRAMES = 5
FACE_RECOGNITION_SCALE = 0.5


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
    if FACE_RECOGNITION_SCALE != 1.0:
        face_frame = cv2.resize(frame, (0, 0), fx=FACE_RECOGNITION_SCALE, fy=FACE_RECOGNITION_SCALE)
    else:
        face_frame = frame

    results = recognize_image(face_frame, save_output=False, threshold=FACE_MATCH_THRESHOLD)

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


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available to PyTorch in this environment. Use a CUDA-enabled PyTorch build before running "
        "recognition_gpu.py."
    )

device = "cuda"
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

print(f"Using torch device: {device}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")

if ort is None:
    print("ONNX Runtime is not installed. Facial recognition will not run without it.")
else:
    available_providers = ort.get_available_providers()
    print(f"ONNX Runtime providers: {available_providers}")
    if "CUDAExecutionProvider" not in available_providers:
        print(
            "CUDAExecutionProvider is not available for ONNX Runtime. YOLO will run on GPU, "
            "but facial recognition may still run on CPU."
        )

prepare_known_faces(KNOWN_FACES_ROOT)

# Load the local YOLOv5 model from the cloned repository.
model = load_yolo_model(YOLO_REPO_DIR, device=device, model_name=YOLO_MODEL_NAME)
if hasattr(model, "half"):
    model.half()
if hasattr(model, "amp"):
    model.amp = True

helmet_model = None
if HELMET_WEIGHTS.exists():
    helmet_model = load_yolo_model(YOLO_REPO_DIR, weights_path=HELMET_WEIGHTS, device=device)
    if hasattr(helmet_model, "half"):
        helmet_model.half()
    if hasattr(helmet_model, "amp"):
        helmet_model.amp = True
    print(f"Loaded helmet detector weights: {HELMET_WEIGHTS}")
else:
    print(
        f"Helmet detector weights not found at {HELMET_WEIGHTS}. "
        "Train a helmet model first if you want live helmet detection."
    )

vest_model = None
if SAFETY_VEST_WEIGHTS.exists():
    vest_model = load_yolo_model(YOLO_REPO_DIR, weights_path=SAFETY_VEST_WEIGHTS, device=device)
    if hasattr(vest_model, "half"):
        vest_model.half()
    if hasattr(vest_model, "amp"):
        vest_model.amp = True
    print(f"Loaded safety vest detector weights: {SAFETY_VEST_WEIGHTS}")
else:
    print(
        f"Safety vest detector weights not found at {SAFETY_VEST_WEIGHTS}. "
        "Train a safety vest model first if you want live safety vest detection."
    )

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

frame_index = 0
recognized_faces = []

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        frame_index += 1
        if frame_index % FACE_RECOGNITION_EVERY_N_FRAMES == 0 or not recognized_faces:
            recognized_faces = recognize_faces(frame)

        with torch.inference_mode():
            results = model(frame, size=YOLO_IMAGE_SIZE)
        detections = results.xyxy[0].cpu().numpy()
        names = results.names
        helmet_detections = detect_helmet_boxes(helmet_model, frame, image_size=HELMET_IMAGE_SIZE) if helmet_model else []
        vest_detections = (
            detect_safety_vest_boxes(vest_model, frame, image_size=SAFETY_VEST_IMAGE_SIZE) if vest_model else []
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

        for x1, y1, x2, y2, conf, cls in detections:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            class_name = names[int(cls)]
            current_box = (x1, y1, x2, y2)

            if class_name == "person":
                face_match = get_person_match(current_box, recognized_faces)
                base_label = format_person_label(face_match)
                helmet_match = match_helmet_to_person(current_box, helmet_detections)
                vest_match = match_safety_vest_to_person(current_box, vest_detections)
                if base_label == "Threat":
                    label = base_label
                else:
                    label = append_safety_vest_status(append_helmet_status(base_label, helmet_match), vest_match)
                if base_label == "Threat":
                    color = (0, 0, 255)
                elif helmet_match and vest_match:
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

        cv2.imshow("YOLOv5 RealSense Recognition (GPU)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
