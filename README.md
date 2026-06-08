# Camera Object Recognition

This project uses an Intel RealSense camera with YOLOv5 for live object detection, person recognition, helmet detection, safety vest detection, and basic anti-spoofing against faces shown on device screens.

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
- optionally loads a second custom YOLOv5 helmet model from `./weights/helmet_best.pt`
- optionally loads a third custom YOLOv5 safety vest model from `./weights/safety_vest_best.pt`
- marks each detected person with both helmet and vest status
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
- optionally loads a custom helmet detector from `./weights/helmet_best.pt`
- optionally loads a custom safety vest detector from `./weights/safety_vest_best.pt`
- appends both helmet and vest status to each person label
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
- optionally runs a GPU helmet detector from `./weights/helmet_best.pt`
- optionally runs a GPU safety vest detector from `./weights/safety_vest_best.pt`
- appends both helmet and vest status to each person label
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

## PPE Detection

Helmet detection and safety vest detection are supported as dedicated YOLOv5 models in:
- `main.py`
- `recognition.py`
- `recognition_gpu.py`

How it works:
- the main YOLO model still detects people and general objects
- a custom helmet model detects helmets
- a custom safety vest model detects safety vests
- when a helmet box lands in the upper part of a detected person box, that person is labeled `Helmet`
- when a vest box lands in the torso region of a detected person box, that person is labeled `Vest`
- when a PPE item is not matched to that person, the label becomes `No Helmet` or `No Vest`

Expected live model paths:

```text
weights/helmet_best.pt
weights/safety_vest_best.pt
```

If either file does not exist, the scripts still run, but that PPE detector stays disabled.

## Train Your Helmet Model

If you already have a helmet dataset, the quickest path is:

1. Your repo now includes a ready-to-use dataset config at `configs/helmet_dataset.yaml`
2. It points to:

```text
C:\Users\hassa\Desktop\Intel-RealSense\dataset\new dataset
```

3. Train the model with:

```powershell
python .\train_helmet_model.py --epochs 80 --img 640 --batch 16
```

The script wraps the local `yolov5/train.py` entrypoint and stores outputs under:

```text
runs/train/helmet_detector/
```

After training, place the best weights here:

```text
weights/helmet_best.pt
```

Example:

```powershell
Copy-Item .\runs\train\helmet_detector\weights\best.pt .\weights\helmet_best.pt
```

### Expected Dataset Layout

Your configured dataset is expected to follow a standard YOLO layout, for example:

```text
your_dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Your current dataset config uses:
- `nc: 3`
- `names: [Helmet, No Helmet, Worker]`

The runtime currently uses helmet-like class names such as:
- `helmet`
- `hardhat`
- `hard hat`
- `safety helmet`

If your class name is different, rename it in your dataset YAML or in the trained model pipeline so the live matcher can recognize it cleanly.

## Train Your Safety Vest Model

Your repo now also includes a ready-to-use safety vest dataset config at `configs/safety_vest_dataset.yaml`.

It points to:

```text
C:\Users\hassa\Desktop\Intel-RealSense\safety_vest_dataset
```

The current vest dataset uses:
- `nc: 2`
- `names: [Safety Vest, NO-Safety Vest]`

This vest dataset is currently a CSV-style export with `_annotations.csv` files, not native YOLO label files.
The repo now handles that automatically:
- `train_safety_vest_model.py` first converts the dataset into YOLO image/label format
- the converted dataset is written under `generated_datasets/safety_vest_yolo/`
- YOLOv5 training then runs on that generated dataset

Train the vest detector with:

```powershell
python .\train_safety_vest_model.py --epochs 80 --img 640 --batch 16
```

The script stores outputs under:

```text
runs/train/safety_vest_detector/
```

The generated YOLO-format dataset will be stored under:

```text
generated_datasets/safety_vest_yolo/
```

After training, place the best weights here:

```text
weights/safety_vest_best.pt
```

Example:

```powershell
Copy-Item .\runs\train\safety_vest_detector\weights\best.pt .\weights\safety_vest_best.pt
```

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
- Running a second YOLO model for helmets adds some overhead, especially on CPU.
- If you want better spoof resistance, the next step would be to use RealSense depth data for liveness checks instead of only 2D screen overlap.
