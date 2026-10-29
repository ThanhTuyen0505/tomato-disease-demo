import os
import subprocess
import zipfile
from pathlib import Path

import streamlit as st
from ultralytics import YOLO

DATA_DIR = Path("/tmp/tomato-detection-models")
DATASET_ID = "thanhtuanh/tomato-detection-models"

MODEL_FILES = {
    "YOLOv8n": "tomato_results/YOLOv8n/selected/selected_model.pt",
    "YOLO11n": "tomato_results/YOLO11n/selected/selected_model.pt",
    "YOLO11s": "tomato_results/YOLO11s/selected/selected_model.pt",
    "RT-DETR-L": "tomato_results/RT-DETR-L/selected/selected_model.pt",
}


def download_one_model(model_name, relative_path):
    model_dir = DATA_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    target = model_dir / "selected_model.pt"

    if target.is_file() and target.stat().st_size > 0:
        return target

    token = st.secrets.get("KAGGLE_API_TOKEN", "")
    if not token:
        raise RuntimeError(
            "Thiếu KAGGLE_API_TOKEN trong Streamlit Secrets."
        )

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = str(token)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d", DATASET_ID,
        "-f", relative_path,
        "-p", str(model_dir),
        "--force",
    ]

    result = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "Không có chi tiết")[-2000:]
        raise RuntimeError(
            f"Không tải được {model_name} từ Kaggle.\n"
            f"Đường dẫn: {relative_path}\n"
            f"Chi tiết: {details}"
        )

    # Kaggle thường trả về ZIP; chỉ giải nén checkpoint cần thiết.
    zip_files = list(model_dir.glob("*.zip"))

    if zip_files:
        extracted = None

        for zip_path in zip_files:
            try:
                with zipfile.ZipFile(zip_path, "r") as archive:
                    matches = [
                        name for name in archive.namelist()
                        if name.endswith("selected_model.pt")
                    ]

                    if matches:
                        # Chỉ lấy đúng file checkpoint, không giải nén toàn bộ.
                        with archive.open(matches[0]) as src:
                            with target.open("wb") as dst:
                                while True:
                                    chunk = src.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    dst.write(chunk)
                        extracted = target
                        break
            except zipfile.BadZipFile:
                continue

        for zip_path in zip_files:
            zip_path.unlink(missing_ok=True)

        if extracted is None:
            raise FileNotFoundError(
                f"Không tìm thấy selected_model.pt trong file ZIP "
                f"của {model_name}."
            )

    # Một số phiên bản CLI có thể tạo file checkpoint trực tiếp.
    if not target.is_file():
        candidates = list(model_dir.rglob("selected_model.pt"))
        if candidates:
            candidate = candidates[0]
            if candidate.resolve() != target.resolve():
                target.write_bytes(candidate.read_bytes())

    if not target.is_file() or target.stat().st_size == 0:
        raise FileNotFoundError(
            f"Checkpoint {model_name} không tồn tại hoặc bị rỗng."
        )

    return target


@st.cache_resource
def load_model(model_name):
    if model_name not in MODEL_FILES:
        raise ValueError(f"Mô hình không hợp lệ: {model_name}")

    model_path = download_one_model(
        model_name,
        MODEL_FILES[model_name],
    )

    return YOLO(str(model_path))
