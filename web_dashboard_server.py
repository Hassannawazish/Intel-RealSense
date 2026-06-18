import os
import threading
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(os.cpu_count() or 8))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["NUMEXPR_MAX_THREADS"])

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from flask import Flask, Response, abort, jsonify, request, send_file

try:
    import winsound
except ImportError:
    winsound = None

if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.benchmark = True

try:
    import serial
    from serial import SerialException
except ModuleNotFoundError:
    serial = None
    SerialException = Exception

from display_utils import build_person_ppe_lines, draw_label_block
from employee_face_sync import sync_employee_faces
from helmet_detection import (
    detect_helmet_boxes,
    get_person_head_region,
    get_intersection_area,
    load_yolo_model,
    match_helmet_to_person,
)
from ppe_accessory_detection import (
    detect_glove_boxes,
    detect_goggle_boxes,
    get_person_eye_region,
    get_person_hand_regions,
    match_gloves_to_person,
    match_goggles_to_person,
)
from safety_vest_detection import (
    detect_safety_vest_boxes,
    get_person_torso_region,
    match_safety_vest_to_person,
)

try:
    from facial_recognition import add_person, recognize_image, remove_face_database
except ModuleNotFoundError as exc:
    missing_package = exc.name or "required package"
    raise ModuleNotFoundError(
        f"Missing dependency: {missing_package}. Install the web dashboard face recognition runtime with "
        "`python -m pip install -r .\\requirements_web_dashboard.txt` and run `facial_recognition setup`."
    ) from exc


ROOT = Path(__file__).resolve().parent
YOLO_REPO_DIR = ROOT / "yolov5"
OBJECT_MODEL_NAME = "yolov5n"
OBJECT_IMAGE_SIZE = 352
HELMET_WEIGHTS = ROOT / "weights" / "helmet_best.pt"
HELMET_IMAGE_SIZE = 352
SAFETY_VEST_WEIGHTS = ROOT / "weights" / "safety_vest_best.pt"
SAFETY_VEST_IMAGE_SIZE = 352
ACCESSORY_WEIGHTS = ROOT / "external_models" / "epoch30.pt"
ACCESSORY_IMAGE_SIZE = 352
WINDOW_SIZE = (960, 540)
JPEG_QUALITY = 70
OBJECT_DETECTION_INTERVAL = 3
PPE_DETECTION_INTERVAL = 3
FACE_RECOGNITION_INTERVAL = 5
KNOWN_FACES_ROOT = ROOT / "known_faces"
FACE_MATCH_THRESHOLD = 0.3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SCREEN_DEVICE_CLASSES = {"cell phone", "tv", "laptop", "tablet", "monitor"}
LOGO_PATH = ROOT / "logo" / "scailogo.png"
AUTH_REQUIRED_FRAMES = 2
DOOR_OPEN_SECONDS = 5.0
KNOWN_FACES_SYNC_SECONDS = 15.0
KNOWN_FACES_SOURCE = os.environ.get("KNOWN_FACES_SOURCE", "api").strip().lower()
KNOWN_FACES_RUNTIME_SYNC_ENABLED = os.environ.get("KNOWN_FACES_RUNTIME_SYNC_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
EMPLOYEE_API_URL = os.environ.get(
    "EMPLOYEE_API_URL",
    "https://scai-erp.tech/api/ppe-detection/employees?companyId=CEC276A2-0939-4F32-A228-DA6EAA9FCA57",
).strip()
EMPLOYEE_API_TIMEOUT_SECONDS = float(os.environ.get("EMPLOYEE_API_TIMEOUT_SECONDS", "20.0"))
ARDUINO_ENABLED = os.environ.get("ARDUINO_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
ARDUINO_PORT = os.environ.get("ARDUINO_PORT", "COM3").strip()
ARDUINO_BAUDRATE = int(os.environ.get("ARDUINO_BAUDRATE", "9600"))
ARDUINO_TIMEOUT_SECONDS = float(os.environ.get("ARDUINO_TIMEOUT_SECONDS", "1.0"))
SUPPORTED_LANGUAGES = {"en", "fr", "es"}

BACKEND_TRANSLATIONS = {
    "en": {
        "unknown_person": "Unknown Person",
        "helmet": "Helmet",
        "vest": "Vest",
        "gloves": "Gloves",
        "goggles": "Goggles",
        "door_triggered": "[ACCESS] Door open triggered for {name}",
    },
    "fr": {
        "unknown_person": "Personne Inconnue",
        "helmet": "Casque",
        "vest": "Gilet",
        "gloves": "Gants",
        "goggles": "Lunettes",
        "door_triggered": "[ACCÈS] Ouverture de porte déclenchée pour {name}",
    },
    "es": {
        "unknown_person": "Persona Desconocida",
        "helmet": "Casco",
        "vest": "Chaleco",
        "gloves": "Guantes",
        "goggles": "Gafas",
        "door_triggered": "[ACCESO] Apertura de puerta activada para {name}",
    },
}

app = Flask(__name__)
DETECTED_BOX_COLOR = (0, 255, 255)
MISSING_BOX_COLOR = (0, 0, 255)
FACE_BOX_COLOR = (0, 255, 255)


def draw_detection_rectangle(frame, box, color, label=None, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if not label:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    text_thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
    text_top = max(0, y1 - text_height - 10)
    text_bottom = text_top + text_height + 8
    text_right = x1 + text_width + 10
    cv2.rectangle(frame, (x1, text_top), (text_right, text_bottom), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 5, text_bottom - 5),
        font,
        font_scale,
        (20, 20, 20),
        text_thickness,
        cv2.LINE_AA,
    )


def get_person_face_region(person_box):
    x1, y1, x2, y2 = person_box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return (
        x1 + int(width * 0.2),
        y1 + int(height * 0.04),
        x2 - int(width * 0.2),
        y1 + int(height * 0.3),
    )


def build_item_status(label, match):
    return {
        "label": label,
        "detected": bool(match),
        "score": round(float(match["score"]), 3) if match else None,
    }


def get_language():
    requested = request.args.get("lang", "fr").strip().lower()
    return requested if requested in SUPPORTED_LANGUAGES else "fr"


def translate(language, key):
    return BACKEND_TRANSLATIONS.get(language, BACKEND_TRANSLATIONS["en"])[key]


def format_person_name(folder_name):
    return folder_name.replace("_", " ").replace("-", " ").title()


def prepare_known_faces(known_faces_root):
    if not known_faces_root.exists():
        raise FileNotFoundError(
            f"Known faces root folder not found: {known_faces_root}. "
            "Create person subfolders inside known_faces and place front-facing photos in each one."
        )

    person_dirs = sorted(path for path in known_faces_root.iterdir() if path.is_dir())
    if not person_dirs:
        raise FileNotFoundError(
            f"No person folders were found in {known_faces_root}. "
            "Add folders like known_faces/hassan or known_faces/rana."
        )

    remove_face_database()
    loaded_people = 0

    for person_dir in person_dirs:
        image_paths = sorted(
            path for path in person_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            continue

        person_name = format_person_name(person_dir.name)
        for image_path in image_paths:
            add_person(person_name, str(image_path))
            loaded_people += 1

    if loaded_people == 0:
        raise FileNotFoundError(
            f"No supported images were found in any subfolder of {known_faces_root}. "
            "Add JPG, JPEG, PNG, or BMP images for each person you want to recognize."
        )


def recognize_faces(frame, language):
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
            name = translate(language, "unknown_person")
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
    screen_boxes = []
    for x1, y1, x2, y2, conf, cls in detections:
        class_name = names[int(cls)]
        if class_name in SCREEN_DEVICE_CLASSES:
            screen_boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return screen_boxes


def flag_spoof_faces(recognized_faces, screen_boxes):
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


def screen_has_spoof_face(screen_box, recognized_faces):
    for face in recognized_faces:
        if face["is_spoof"] and get_intersection_area(screen_box, face["box"]) > 0:
            return True
    return False


def get_person_match(person_box, recognized_faces):
    x1, y1, x2, y2 = person_box
    for face in recognized_faces:
        left, top, right, bottom = face["box"]
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            return face
    return None


def format_person_label(face, language):
    if face is None:
        return translate(language, "unknown_person")
    if face.get("is_spoof"):
        return "Threat"
    if face["name"] == translate(language, "unknown_person"):
        return face["name"]
    return f'{face["name"]}: {face["score"]:.2f}'


class PPECameraService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.object_model = None
        self.helmet_model = None
        self.vest_model = None
        self.accessory_model = None
        self.pipeline = None
        self.frame_lock = threading.Lock()
        self.latest_jpeg = None
        self.current_language = "fr"
        self.authorized_streak = 0
        self.door_open_until = 0.0
        self.last_opened_at = None
        self.last_authorized_name = None
        self.arduino = None
        self.arduino_switch_on = False
        self.last_known_faces_sync = 0.0
        self.frame_index = 0
        self.cached_faces = []
        self.cached_person_detections = np.empty((0, 6), dtype=np.float32)
        self.cached_detection_names = {}
        self.cached_screen_boxes = []
        self.cached_helmet_detections = []
        self.cached_vest_detections = []
        self.cached_glove_detections = []
        self.cached_goggle_detections = []
        self.latest_status = {
            "ready": False,
            "device": self.device,
            "person_count": 0,
            "primary_person": None,
            "threat_detected": False,
            "threat_count": 0,
            "models": {
                "helmet": False,
                "vest": False,
                "accessory": False,
            },
            "updated_at": None,
            "error": None,
            "authorization": {
                "required_frames": AUTH_REQUIRED_FRAMES,
                "consecutive_frames": 0,
                "eligible": False,
                "known_person": False,
                "ppe_complete": False,
                "switch_on": False,
                "door_open": False,
                "authorized_name": None,
                "last_opened_at": None,
            },
        }
        self._stop_event = threading.Event()
        self._thread = None

    def _prepare_model(self, model):
        if model is None:
            return None
        if self.device == "cuda":
            if hasattr(model, "half"):
                model.half()
            if hasattr(model, "amp"):
                model.amp = True
        return model

    def load_models(self):
        self._sync_known_faces_source(force=True, reload_faces=False)
        prepare_known_faces(KNOWN_FACES_ROOT)
        self.last_known_faces_sync = time.time()
        self.object_model = self._prepare_model(
            load_yolo_model(YOLO_REPO_DIR, device=self.device, model_name=OBJECT_MODEL_NAME)
        )

        if SAFETY_VEST_WEIGHTS.exists():
            self.vest_model = self._prepare_model(
                load_yolo_model(YOLO_REPO_DIR, weights_path=SAFETY_VEST_WEIGHTS, device=self.device)
            )

        if ACCESSORY_WEIGHTS.exists():
            self.accessory_model = self._prepare_model(
                load_yolo_model(YOLO_REPO_DIR, weights_path=ACCESSORY_WEIGHTS, device=self.device)
            )

        self.helmet_model = self.accessory_model
        if self.helmet_model is None and HELMET_WEIGHTS.exists():
            self.helmet_model = self._prepare_model(
                load_yolo_model(YOLO_REPO_DIR, weights_path=HELMET_WEIGHTS, device=self.device)
            )

        self.latest_status["models"] = {
            "helmet": self.helmet_model is not None,
            "vest": self.vest_model is not None,
            "accessory": self.accessory_model is not None,
        }

    def start(self):
        self.load_models()
        self._connect_arduino()
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self.pipeline:
            self.pipeline.stop()
        self._set_arduino_switch(False)
        self._close_arduino()

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                if KNOWN_FACES_RUNTIME_SYNC_ENABLED:
                    self._refresh_known_faces_if_needed()
                frames = self.pipeline.wait_for_frames()
                while True:
                    latest_frames = self.pipeline.poll_for_frames()
                    if not latest_frames:
                        break
                    frames = latest_frames
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())
                language = self.current_language
                annotated_frame, status = self._process_frame(frame, language)
                ok, encoded = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if not ok:
                    continue

                with self.frame_lock:
                    self.latest_jpeg = encoded.tobytes()
                    self.latest_status = status
            except Exception as exc:
                with self.frame_lock:
                    self.latest_status = {
                        **self.latest_status,
                        "ready": False,
                        "error": str(exc),
                        "updated_at": time.time(),
                    }
                time.sleep(0.1)

    def _refresh_known_faces_if_needed(self):
        now = time.time()
        if now - self.last_known_faces_sync < KNOWN_FACES_SYNC_SECONDS:
            return

        self.last_known_faces_sync = now
        self._sync_known_faces_source(force=False, reload_faces=True)

    def _sync_known_faces_source(self, force, reload_faces):
        if KNOWN_FACES_SOURCE != "api":
            return
        if not EMPLOYEE_API_URL:
            print("[FACES] Employee API URL is empty, skipping remote face sync.")
            return

        try:
            summary = sync_employee_faces(
                EMPLOYEE_API_URL,
                KNOWN_FACES_ROOT,
                timeout_seconds=EMPLOYEE_API_TIMEOUT_SECONDS,
            )
            if force:
                print(
                    f"[FACES] Synced {summary['employee_count']} employees and {summary['image_count']} images from employee API"
                )
            if not reload_faces or not summary["has_changes"]:
                return

            prepare_known_faces(KNOWN_FACES_ROOT)
            print(
                f"[FACES] Sync applied: +{summary['created_folders']} folders, +{summary['downloaded_images']} images, "
                f"-{summary['removed_folders']} folders, -{summary['removed_images']} images"
            )
        except Exception as exc:
            if force:
                raise
            print(f"[FACES] Remote sync skipped: {exc}")

    def _connect_arduino(self):
        if not ARDUINO_ENABLED:
            print("[ARDUINO] Disabled")
            return
        if serial is None:
            print("[ARDUINO] pyserial is not installed. Run `python -m pip install -r .\\requirements_web_dashboard.txt`.")
            return
        if not ARDUINO_PORT:
            print("[ARDUINO] No serial port configured.")
            return

        try:
            self.arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUDRATE, timeout=ARDUINO_TIMEOUT_SECONDS)
            time.sleep(2.0)
            print(f"[ARDUINO] Connected on {ARDUINO_PORT} at {ARDUINO_BAUDRATE} baud")
            self._write_arduino_command("OFF")
            self.arduino_switch_on = False
        except SerialException as exc:
            self.arduino = None
            print(f"[ARDUINO] Connection failed on {ARDUINO_PORT}: {exc}")

    def _close_arduino(self):
        if self.arduino is None:
            return
        try:
            if self.arduino.is_open:
                self.arduino.close()
        finally:
            self.arduino = None

    def _write_arduino_command(self, command):
        if self.arduino is None:
            return
        try:
            self.arduino.write(f"{command}\n".encode("utf-8"))
            self.arduino.flush()
        except SerialException as exc:
            print(f"[ARDUINO] Write failed: {exc}")
            self._close_arduino()

    def _set_arduino_switch(self, switch_on):
        switch_on = bool(switch_on)
        if switch_on == self.arduino_switch_on:
            return

        if self.arduino is None and ARDUINO_ENABLED:
            self._connect_arduino()

        self._write_arduino_command("ON" if switch_on else "OFF")
        self.arduino_switch_on = switch_on

    def _is_known_person(self, face_match, language):
        return bool(
            face_match
            and not face_match.get("is_spoof")
            and face_match["name"] != translate(language, "unknown_person")
        )

    def _update_authorization_state(self, primary_person, language):
        now = time.time()
        known_person = bool(primary_person and self._is_known_person(primary_person["face"], language))
        ppe_complete = bool(
            primary_person
            and primary_person["helmet"]
            and primary_person["vest"]
            and primary_person["gloves"]
            and primary_person["goggles"]
        )
        eligible = known_person and ppe_complete

        if eligible:
            self.authorized_streak += 1
            self.last_authorized_name = primary_person["face"]["name"]
        else:
            self.authorized_streak = 0

        if eligible and self.authorized_streak == AUTH_REQUIRED_FRAMES:
            self.door_open_until = now + DOOR_OPEN_SECONDS
            self.last_opened_at = now
            self._trigger_door_open(self.last_authorized_name, language)

        switch_on = now < self.door_open_until
        self._set_arduino_switch(switch_on)
        return {
            "required_frames": AUTH_REQUIRED_FRAMES,
            "consecutive_frames": self.authorized_streak,
            "eligible": eligible,
            "known_person": known_person,
            "ppe_complete": ppe_complete,
            "switch_on": switch_on,
            "door_open": switch_on,
            "authorized_name": self.last_authorized_name if eligible or switch_on else None,
            "last_opened_at": self.last_opened_at,
        }

    def _trigger_door_open(self, person_name, language):
        # Placeholder hook for real door hardware integration.
        if winsound is not None:
            try:
                winsound.Beep(1800, 350)
                winsound.Beep(2200, 350)
            except RuntimeError:
                pass
        print(BACKEND_TRANSLATIONS.get(language, BACKEND_TRANSLATIONS["en"])["door_triggered"].format(name=person_name))

    def _process_frame(self, frame, language):
        self.frame_index += 1

        if self.frame_index % FACE_RECOGNITION_INTERVAL == 1 or not self.cached_faces:
            self.cached_faces = recognize_faces(frame, language)
        recognized_faces = [dict(face) for face in self.cached_faces]

        if self.frame_index % OBJECT_DETECTION_INTERVAL == 1 or len(self.cached_person_detections) == 0:
            with torch.inference_mode():
                results = self.object_model(frame, size=OBJECT_IMAGE_SIZE)
            detections = results.xyxy[0].cpu().numpy()
            names = results.names
            self.cached_detection_names = names
            self.cached_person_detections = detections
            self.cached_screen_boxes = extract_screen_boxes(detections, names)
        else:
            detections = self.cached_person_detections
            names = self.cached_detection_names

        screen_boxes = self.cached_screen_boxes
        flag_spoof_faces(recognized_faces, screen_boxes)

        if self.frame_index % PPE_DETECTION_INTERVAL == 1 or (
            self.cached_helmet_detections is None and self.cached_vest_detections is None
        ):
            self.cached_helmet_detections = (
                detect_helmet_boxes(self.helmet_model, frame, image_size=HELMET_IMAGE_SIZE) if self.helmet_model else []
            )
            self.cached_vest_detections = (
                detect_safety_vest_boxes(self.vest_model, frame, image_size=SAFETY_VEST_IMAGE_SIZE)
                if self.vest_model
                else []
            )
            self.cached_glove_detections = (
                detect_glove_boxes(self.accessory_model, frame, image_size=ACCESSORY_IMAGE_SIZE)
                if self.accessory_model
                else []
            )
            self.cached_goggle_detections = (
                detect_goggle_boxes(self.accessory_model, frame, image_size=ACCESSORY_IMAGE_SIZE)
                if self.accessory_model
                else []
            )

        helmet_detections = self.cached_helmet_detections
        vest_detections = self.cached_vest_detections
        glove_detections = self.cached_glove_detections
        goggle_detections = self.cached_goggle_detections

        annotated = frame.copy()
        persons = []

        for x1, y1, x2, y2, conf, cls in detections:
            class_name = names[int(cls)]
            if class_name != "person":
                continue

            person_box = tuple(map(int, (x1, y1, x2, y2)))
            face_match = get_person_match(person_box, recognized_faces)
            helmet_match = match_helmet_to_person(person_box, helmet_detections)
            vest_match = match_safety_vest_to_person(person_box, vest_detections)
            glove_match = match_gloves_to_person(person_box, glove_detections)
            goggle_match = match_goggles_to_person(person_box, goggle_detections)

            persons.append(
                {
                    "box": person_box,
                    "confidence": float(conf),
                    "face": face_match,
                    "helmet": helmet_match,
                    "vest": vest_match,
                    "gloves": glove_match,
                    "goggles": goggle_match,
                }
            )

        primary = None
        if persons:
            primary = max(persons, key=lambda item: (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1]))

        authorization = self._update_authorization_state(primary, language)

        for person in persons:
            x1, y1, x2, y2 = person["box"]
            person_name = format_person_label(person["face"], language)
            is_threat = person_name == "Threat"
            all_ppe = person["helmet"] and person["vest"] and person["gloves"] and person["goggles"]
            if is_threat:
                color = (0, 0, 255)
            else:
                color = (0, 200, 0) if all_ppe else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            face_region = get_person_face_region(person["box"])
            head_region = get_person_head_region(person["box"])
            torso_region = get_person_torso_region(person["box"])
            left_hand_region, right_hand_region = get_person_hand_regions(person["box"])
            eye_region = get_person_eye_region(person["box"])

            if person["face"]:
                draw_detection_rectangle(
                    annotated,
                    person["face"]["box"],
                    FACE_BOX_COLOR,
                    f"Visage {person['face']['score']:.2f}",
                )
            else:
                draw_detection_rectangle(annotated, face_region, MISSING_BOX_COLOR, "Visage")

            if person["helmet"]:
                draw_detection_rectangle(
                    annotated,
                    person["helmet"]["box"],
                    DETECTED_BOX_COLOR,
                    f"Casque {person['helmet']['score']:.2f}",
                )
            else:
                draw_detection_rectangle(annotated, head_region, MISSING_BOX_COLOR, "Pas de Casque")

            if person["vest"]:
                draw_detection_rectangle(
                    annotated,
                    person["vest"]["box"],
                    DETECTED_BOX_COLOR,
                    f"Gilet {person['vest']['score']:.2f}",
                )
            else:
                draw_detection_rectangle(annotated, torso_region, MISSING_BOX_COLOR, "Pas de Gilet")

            if person["gloves"]:
                draw_detection_rectangle(
                    annotated,
                    person["gloves"]["box"],
                    DETECTED_BOX_COLOR,
                    f"Gants {person['gloves']['score']:.2f}",
                )
            else:
                draw_detection_rectangle(annotated, left_hand_region, MISSING_BOX_COLOR, "Main")
                draw_detection_rectangle(annotated, right_hand_region, MISSING_BOX_COLOR, "Main")

            if person["goggles"]:
                draw_detection_rectangle(
                    annotated,
                    person["goggles"]["box"],
                    DETECTED_BOX_COLOR,
                    f"Lunettes {person['goggles']['score']:.2f}",
                )
            else:
                draw_detection_rectangle(annotated, eye_region, MISSING_BOX_COLOR, "Pas de Lunettes")

            label_lines = (
                ["Threat"]
                if is_threat
                else build_person_ppe_lines(
                    person_name,
                    person["helmet"],
                    person["vest"],
                    person["gloves"],
                    person["goggles"],
                    language=language,
                )
            )
            draw_label_block(annotated, label_lines, (x1, y1), color, font_scale=0.6, thickness=2)

        for x1, y1, x2, y2, conf, cls in detections:
            class_name = names[int(cls)]
            current_box = (int(x1), int(y1), int(x2), int(y2))
            if class_name in SCREEN_DEVICE_CLASSES and screen_has_spoof_face(current_box, recognized_faces):
                draw_detection_rectangle(annotated, current_box, (0, 0, 255), "Threat")

        status = {
            "ready": True,
            "device": self.device,
            "person_count": len(persons),
            "threat_detected": any(face.get("is_spoof") for face in recognized_faces),
            "threat_count": sum(1 for face in recognized_faces if face.get("is_spoof")),
            "updated_at": time.time(),
            "models": self.latest_status["models"],
            "primary_person": None,
            "error": None,
            "authorization": authorization,
        }

        if primary:
            primary_name = format_person_label(primary["face"], language)
            status["primary_person"] = {
                "name": primary_name,
                "name_confidence": (
                    round(float(primary["face"]["score"]), 3)
                    if primary["face"] and not primary["face"].get("is_spoof")
                    else None
                ),
                "confidence": round(primary["confidence"], 3),
                "ppe": {
                    "helmet": build_item_status(translate(language, "helmet"), primary["helmet"]),
                    "vest": build_item_status(translate(language, "vest"), primary["vest"]),
                    "gloves": build_item_status(translate(language, "gloves"), primary["gloves"]),
                    "goggles": build_item_status(translate(language, "goggles"), primary["goggles"]),
                },
            }

        resized = cv2.resize(annotated, WINDOW_SIZE, interpolation=cv2.INTER_LINEAR)
        return resized, status

    def get_frame(self):
        with self.frame_lock:
            return self.latest_jpeg

    def get_status(self):
        with self.frame_lock:
            return dict(self.latest_status)

    def set_language(self, language):
        self.current_language = language


service = PPECameraService()


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "device": service.device})


@app.route("/api/status")
def status():
    service.set_language(get_language())
    return jsonify(service.get_status())


@app.route("/api/branding/logo")
def branding_logo():
    if not LOGO_PATH.exists():
        abort(404, description=f"Logo not found at {LOGO_PATH}")
    return send_file(LOGO_PATH, mimetype="image/png", max_age=0)


@app.route("/api/frame")
def frame():
    service.set_language(get_language())
    latest_frame = service.get_frame()
    if latest_frame is None:
        abort(503, description="Camera frame is not ready yet.")
    return Response(latest_frame, mimetype="image/jpeg")


@app.route("/video_feed")
def video_feed():
    service.set_language(get_language())
    def generate():
        while True:
            frame = service.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    response = Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Accel-Buffering"] = "no"
    return response


if __name__ == "__main__":
    service.start()
    try:
        app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
    finally:
        service.stop()
