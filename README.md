# Intel-RealSense

# Camera Object Recognition

This project uses an Intel RealSense camera and YOLOv5 for live object detection.

## Files

### `test_camera.py`

This is a simple camera test script for the Intel RealSense D415.

What it does:
- Starts the RealSense pipeline
- Opens the RGB color stream at `640x480` and `30 FPS`
- Reads frames from the camera
- Displays the raw color video in an OpenCV window
- Exits when you press `q`

Why it is useful:
- Confirms that `pyrealsense2` is installed correctly
- Confirms that the RealSense camera is connected and readable from Python
- Helps isolate camera problems before adding YOLO inference

Run it with:

```powershell
python .\test_camera.py
```

### `main.py`

This is the main live object detection script.

What it does:
- Sets a couple of environment variables to reduce OpenMP and NumExpr startup issues
- Detects whether PyTorch can use `cuda` or must run on `cpu`
- Loads the local YOLOv5 model from the `./yolov5` folder
- Starts the RealSense RGB color stream
- Reads each camera frame
- Runs YOLOv5 inference on each frame
- Draws detection boxes and labels on the frame
- Displays the annotated live video
- Exits when you press `q`

Run it with:

```powershell
python .\main.py
```

## Script Difference

`test_camera.py` only checks the camera stream.

`main.py` uses the same RealSense stream, but adds YOLOv5 object detection on top of it.

## Requirements

You need these main Python packages:

```powershell
python -m pip install opencv-python numpy torch pyrealsense2
```

You also need:
- an Intel RealSense camera connected
- the local `yolov5` folder present in this project
- the YOLOv5 weights file or the ability for YOLOv5 to load the default model

## Notes

- If `test_camera.py` works but `main.py` is slow, that usually means YOLO inference is running on CPU.
- If `main.py` prints `Using torch device: cpu`, then your current PyTorch install does not have CUDA support enabled.
- Close RealSense Viewer before running these scripts, because it can occupy the camera stream.
