import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(os.cpu_count() or 8))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["NUMEXPR_MAX_THREADS"])

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

from helmet_detection import append_helmet_status, detect_helmet_boxes, load_yolo_model, match_helmet_to_person


ROOT = Path(__file__).resolve().parent
YOLO_REPO_DIR = ROOT / "yolov5"
OBJECT_MODEL_NAME = "yolov5s"
HELMET_WEIGHTS = ROOT / "weights" / "helmet_best.pt"
HELMET_IMAGE_SIZE = 416

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
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()
        names = results.names
        helmet_detections = detect_helmet_boxes(helmet_model, frame, image_size=HELMET_IMAGE_SIZE) if helmet_model else []

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

        for x1, y1, x2, y2, conf, cls in detections:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            class_name = names[int(cls)]
            label = f"{class_name} {conf:.2f}"
            color = (255, 0, 0)

            if class_name == "person":
                helmet_match = match_helmet_to_person((x1, y1, x2, y2), helmet_detections)
                label = append_helmet_status("person", helmet_match)
                color = (0, 200, 0) if helmet_match else (0, 165, 255)

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

        cv2.imshow("YOLOv5 RealSense Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
