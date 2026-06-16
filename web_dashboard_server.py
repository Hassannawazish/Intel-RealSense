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

from display_utils import build_person_ppe_lines, draw_label_block
from helmet_detection import (
    detect_helmet_boxes,
    get_person_head_region,
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
OBJECT_IMAGE_SIZE = 416
HELMET_WEIGHTS = ROOT / "weights" / "helmet_best.pt"
HELMET_IMAGE_SIZE = 416
SAFETY_VEST_WEIGHTS = ROOT / "weights" / "safety_vest_best.pt"
SAFETY_VEST_IMAGE_SIZE = 416
ACCESSORY_WEIGHTS = ROOT / "external_models" / "epoch30.pt"
ACCESSORY_IMAGE_SIZE = 416
WINDOW_SIZE = (1280, 720)
KNOWN_FACES_ROOT = ROOT / "known_faces"
FACE_MATCH_THRESHOLD = 0.3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
LOGO_PATH = ROOT / "logo" / "scailogo.png"
AUTH_REQUIRED_FRAMES = 2
DOOR_OPEN_SECONDS = 5.0
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
            }
        )

    return recognized_faces


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
        self.latest_status = {
            "ready": False,
            "device": self.device,
            "person_count": 0,
            "primary_person": None,
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
        prepare_known_faces(KNOWN_FACES_ROOT)
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

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())
                language = self.current_language
                annotated_frame, status = self._process_frame(frame, language)
                ok, encoded = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
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

    def _is_known_person(self, face_match, language):
        return bool(face_match and face_match["name"] != translate(language, "unknown_person"))

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
        print(BACKEND_TRANSLATIONS.get(language, BACKEND_TRANSLATIONS["en"])["door_triggered"].format(name=person_name))

    def _process_frame(self, frame, language):
        recognized_faces = recognize_faces(frame, language)
        with torch.inference_mode():
            results = self.object_model(frame, size=OBJECT_IMAGE_SIZE)
        detections = results.xyxy[0].cpu().numpy()
        names = results.names

        helmet_detections = detect_helmet_boxes(self.helmet_model, frame, image_size=HELMET_IMAGE_SIZE) if self.helmet_model else []
        vest_detections = (
            detect_safety_vest_boxes(self.vest_model, frame, image_size=SAFETY_VEST_IMAGE_SIZE) if self.vest_model else []
        )
        glove_detections = (
            detect_glove_boxes(self.accessory_model, frame, image_size=ACCESSORY_IMAGE_SIZE) if self.accessory_model else []
        )
        goggle_detections = (
            detect_goggle_boxes(self.accessory_model, frame, image_size=ACCESSORY_IMAGE_SIZE) if self.accessory_model else []
        )

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
            all_ppe = person["helmet"] and person["vest"] and person["gloves"] and person["goggles"]
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

            person_name = format_person_label(person["face"], language)
            label_lines = build_person_ppe_lines(
                person_name,
                person["helmet"],
                person["vest"],
                person["gloves"],
                person["goggles"],
                language=language,
            )
            draw_label_block(annotated, label_lines, (x1, y1), color, font_scale=0.6, thickness=2)

        status = {
            "ready": True,
            "device": self.device,
            "person_count": len(persons),
            "updated_at": time.time(),
            "models": self.latest_status["models"],
            "primary_person": None,
            "error": None,
            "authorization": authorization,
        }

        if primary:
            status["primary_person"] = {
                "name": primary["face"]["name"] if primary["face"] else translate(language, "unknown_person"),
                "name_confidence": round(float(primary["face"]["score"]), 3) if primary["face"] else None,
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
