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
from helmet_detection import append_helmet_status, detect_helmet_boxes, load_yolo_model, match_helmet_to_person
from ppe_accessory_detection import (
    append_gloves_status,
    append_goggles_status,
    detect_glove_boxes,
    detect_goggle_boxes,
    match_gloves_to_person,
    match_goggles_to_person,
)
from safety_vest_detection import append_safety_vest_status, detect_safety_vest_boxes, match_safety_vest_to_person


ROOT = Path(__file__).resolve().parent
YOLO_REPO_DIR = ROOT / "yolov5"
OBJECT_MODEL_NAME = "yolov5s"
HELMET_WEIGHTS = ROOT / "weights" / "helmet_best.pt"
HELMET_IMAGE_SIZE = 416
SAFETY_VEST_WEIGHTS = ROOT / "weights" / "safety_vest_best.pt"
SAFETY_VEST_IMAGE_SIZE = 416
ACCESSORY_WEIGHTS = ROOT / "external_models" / "epoch30.pt"
ACCESSORY_IMAGE_SIZE = 416
WINDOW_NAME = "YOLOv5 RealSense Detection"
DISPLAY_SIZE = (1440, 960)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using torch device: {device}")

# Load the local YOLOv5 model from the cloned repository.
model = load_yolo_model(YOLO_REPO_DIR, device=device, model_name=OBJECT_MODEL_NAME)
helmet_model = None
if HELMET_WEIGHTS.exists():
    helmet_model = load_yolo_model(YOLO_REPO_DIR, weights_path=HELMET_WEIGHTS, device=device)
    print(f"Loaded helmet detector weights: {HELMET_WEIGHTS}")
else:
    print(
        f"Helmet detector weights not found at {HELMET_WEIGHTS}. "
        "Train a helmet model first if you want live helmet detection."
    )

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
    print(f"Loaded gloves/goggles detector weights: {ACCESSORY_WEIGHTS}")
else:
    print(
        f"Gloves/goggles detector weights not found at {ACCESSORY_WEIGHTS}. "
        "Place your trained accessory model there if you want live glove and goggle detection."
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
            label = f"{class_name} {conf:.2f}"
            color = (255, 0, 0)

            if class_name == "person":
                helmet_match = match_helmet_to_person((x1, y1, x2, y2), helmet_detections)
                vest_match = match_safety_vest_to_person((x1, y1, x2, y2), vest_detections)
                glove_match = match_gloves_to_person((x1, y1, x2, y2), glove_detections)
                goggle_match = match_goggles_to_person((x1, y1, x2, y2), goggle_detections)
                label_lines = build_person_ppe_lines("Person", helmet_match, vest_match, glove_match, goggle_match)
                label = append_goggles_status(
                    append_gloves_status(
                        append_safety_vest_status(append_helmet_status("person", helmet_match), vest_match),
                        glove_match,
                    ),
                    goggle_match,
                )
                all_ppe_present = helmet_match and vest_match and glove_match and goggle_match
                color = (0, 200, 0) if all_ppe_present else (0, 165, 255)

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
