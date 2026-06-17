import json
import re
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency: requests. Install it with `python -m pip install -r .\\requirements_web_dashboard.txt`."
    ) from exc


RAW_PAYLOAD_FILENAME = "raw.json"
MANIFEST_FILENAME = ".employee_sync_manifest.json"
INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*]+')


def sanitize_folder_name(value, fallback):
    cleaned = INVALID_PATH_CHARS.sub("_", str(value or "").strip())
    cleaned = cleaned.strip(" .")
    return cleaned or fallback


def infer_extension(source_name, content_type=None):
    suffix = Path(urlparse(source_name).path).suffix.lower()
    if suffix:
        return suffix

    if content_type:
        mapping = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/bmp": ".bmp",
            "image/webp": ".webp",
        }
        return mapping.get(content_type.lower(), ".jpg")

    return ".jpg"


def load_manifest(target_root):
    manifest_path = Path(target_root) / MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}

    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_manifest(target_root, manifest):
    manifest_path = Path(target_root) / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_employee_entries(payload):
    employees = payload.get("data", [])
    if not isinstance(employees, list):
        raise ValueError("Expected `data` to be a list of employees in the employee API response.")

    entries = []
    for employee in employees:
        if not isinstance(employee, dict):
            continue
        if employee.get("isActive") is False:
            continue

        employee_id = str(employee.get("id") or "").strip()
        last_name = str(employee.get("lastName") or "").strip()
        if not employee_id or not last_name:
            continue

        gallery_images = employee.get("employeeGalleryImages") or []
        gallery_images = sorted(
            (item for item in gallery_images if isinstance(item, dict) and item.get("fileUrl")),
            key=lambda item: (item.get("sortOrder") or 0, str(item.get("employeeGalleryImageId") or "")),
        )

        image_sources = []
        for index, item in enumerate(gallery_images, start=1):
            image_id = str(item.get("employeeGalleryImageId") or f"gallery_{index}")
            image_sources.append(
                {
                    "id": image_id,
                    "url": str(item["fileUrl"]).strip(),
                    "content_type": item.get("contentType"),
                    "original_name": item.get("originalFileName"),
                }
            )

        # Fallback for employees that only have a single profile image.
        if not image_sources and employee.get("imageUrl"):
            image_sources.append(
                {
                    "id": "profile",
                    "url": str(employee["imageUrl"]).strip(),
                    "content_type": None,
                    "original_name": None,
                }
            )

        if not image_sources:
            continue

        folder_name = sanitize_folder_name(last_name, employee_id)
        entries.append(
            {
                "employee_id": employee_id,
                "folder_name": folder_name,
                "last_name": last_name,
                "images": image_sources,
            }
        )

    return entries


def download_file(url, destination, timeout_seconds):
    response = requests.get(url, stream=True, timeout=timeout_seconds)
    response.raise_for_status()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file_handle:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file_handle.write(chunk)


def sync_employee_faces(api_url, target_root, timeout_seconds=20.0):
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    response = requests.get(api_url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()

    raw_path = target_root / RAW_PAYLOAD_FILENAME
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    entries = build_employee_entries(payload)
    previous_manifest = load_manifest(target_root)
    previous_folders = set(previous_manifest.get("managed_folders", []))
    current_folders = set()
    total_images = 0
    downloaded_images = 0
    created_folders = 0
    removed_images = 0
    removed_folders = 0

    for entry in entries:
        folder_path = target_root / entry["folder_name"]
        if not folder_path.exists():
            created_folders += 1
        folder_path.mkdir(parents=True, exist_ok=True)
        current_folders.add(entry["folder_name"])

        expected_files = set()
        for index, image in enumerate(entry["images"], start=1):
            total_images += 1
            extension = infer_extension(image.get("original_name") or image["url"], image.get("content_type"))
            file_name = f"{index:02d}_{sanitize_folder_name(image['id'], f'image_{index}')}{extension}"
            expected_files.add(file_name)
            destination = folder_path / file_name
            if not destination.exists():
                download_file(image["url"], destination, timeout_seconds)
                downloaded_images += 1

        for existing_file in folder_path.iterdir():
            if not existing_file.is_file():
                continue
            if existing_file.name in {RAW_PAYLOAD_FILENAME, MANIFEST_FILENAME}:
                continue
            if existing_file.name not in expected_files:
                existing_file.unlink()
                removed_images += 1

    stale_folders = previous_folders - current_folders
    for folder_name in stale_folders:
        stale_folder = target_root / folder_name
        if not stale_folder.exists() or not stale_folder.is_dir():
            continue

        for child in stale_folder.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                for nested in child.rglob("*"):
                    if nested.is_file():
                        nested.unlink()
                for nested_dir in sorted((p for p in child.rglob("*") if p.is_dir()), reverse=True):
                    nested_dir.rmdir()
                child.rmdir()
        stale_folder.rmdir()
        removed_folders += 1

    save_manifest(
        target_root,
        {
            "api_url": api_url,
            "managed_folders": sorted(current_folders),
            "employee_count": len(entries),
            "image_count": total_images,
        },
    )

    return {
        "employee_count": len(entries),
        "image_count": total_images,
        "created_folders": created_folders,
        "downloaded_images": downloaded_images,
        "removed_images": removed_images,
        "removed_folders": removed_folders,
        "has_changes": bool(created_folders or downloaded_images or removed_images or removed_folders),
    }
