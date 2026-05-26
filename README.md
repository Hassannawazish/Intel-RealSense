# Camera Object Recognition

This project uses an Intel RealSense camera with YOLOv5 for live object detection, person recognition, and basic anti-spoofing against faces shown on device screens.

## Files

### `test_camera.py`

This script tests the Intel RealSense RGB stream only.

What it does:
- starts the RealSense pipeline
- opens the color stream at `640x480` and `30 FPS`
- displays the raw camera feed
- exits when you press `q`

Use it to confirm:
- the RealSense camera is connected
- `pyrealsense2` is installed correctly
- Python can read frames from the device

Run:

```powershell
python .\test_camera.py
```

### `main.py`

This is the basic YOLOv5 object detection script.

What it does:
- opens the RealSense RGB stream
- loads the local YOLOv5 model from `./yolov5`
- runs object detection on each frame
- draws bounding boxes and labels
- shows the annotated live video

Run:

```powershell
python .\main.py
```

### `recognition.py`

This is the CPU-oriented recognition pipeline.

What it does:
- opens the RealSense RGB stream
- runs YOLOv5 object detection
- runs face recognition using all person folders inside `known_faces`
- labels each matched person using their folder name
- labels other people as `Unknown Person`
- detects faces shown inside screen-like devices such as phones or laptops
- blocks those spoofed matches and labels them as `Threat`

Run:

```powershell
python .\recognition.py
```

### `recognition_gpu.py`

This is the GPU-oriented recognition pipeline.

What it does:
- requires CUDA-enabled PyTorch
- runs YOLOv5 on GPU
- uses a lighter YOLO model and smaller input size for better frame rate
- reduces face recognition frequency to improve live performance
- keeps the same multi-person naming behavior as `recognition.py`
- keeps the same spoof blocking logic and labels screen-based attacks as `Threat`

Run:

```powershell
python .\recognition_gpu.py
```

## Known Faces

The scripts now load every person folder inside `known_faces`.

Example structure:

```text
known_faces/
  hassan/
  rana/
  khan/
  fernando/
```

Each person folder can contain supported images:
- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`

Folder names become display names automatically:
- `hassan` -> `Hassan`
- `rana` -> `Rana`
- `khan` -> `Khan`
- `fernando` -> `Fernando`

Use several clear front-facing images per person for better recognition accuracy.

## Anti-Spoofing Behavior

Both `recognition.py` and `recognition_gpu.py` include a basic anti-spoofing rule.

If a recognized face appears inside a detected screen-like device, such as:
- `cell phone`
- `laptop`
- `tv`
- `monitor`
- `tablet`

then the match is blocked and labeled as:

`Threat`

This is a practical screen-spoof heuristic, not full liveness detection.

## Requirements

There are now two dedicated requirements files:

### CPU recognition

Use:

```powershell
python -m pip install -r .\requirements_recognition_cpu.txt
python -m pip install -r .\yolov5\requirements.txt
facial_recognition setup
```

### GPU recognition

Use:

```powershell
python -m pip install -r .\requirements_recognition_gpu.txt
python -m pip install -r .\yolov5\requirements.txt
facial_recognition setup
```

## Recommended Environments

### CPU / general recognition

A Python environment with:
- `torch`
- `onnxruntime`
- `facial_recognition`
- `mediapipe`
- `pyrealsense2`

### GPU recognition

A Python environment with:
- CUDA-enabled `torch`
- `onnxruntime-gpu` if you want the face-recognition backend to try GPU too
- the rest of the same packages as the CPU setup

If `recognition_gpu.py` says CUDA is unavailable, your PyTorch install is CPU-only and must be replaced with a CUDA-enabled build.

## Notes

- Close RealSense Viewer before running these scripts, otherwise the camera stream may already be occupied.
- `recognition_gpu.py` improves YOLO performance, but face recognition may still run on CPU if ONNX Runtime GPU support is not active.
- If `main.py` or `recognition.py` feels slow, that is expected on CPU-heavy environments.
- If you want better spoof resistance, the next step would be to use RealSense depth data for liveness checks instead of only 2D screen overlap.
