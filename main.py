import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(os.cpu_count() or 8))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["NUMEXPR_MAX_THREADS"])

import cv2
import numpy as np
import pyrealsense2 as rs
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using torch device: {device}")

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
        results = model(frame)
        annotated_frame = results.render()[0]

        cv2.imshow("YOLOv5 RealSense Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
