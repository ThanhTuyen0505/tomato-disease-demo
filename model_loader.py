import os
import subprocess
from pathlib import Path

import streamlit as st
from ultralytics import YOLO

DATA_DIR = Path("/tmp/tomato_models")
KAGGLE_DATASET_ID = "thanhtuanh/tomato-detection-models"

MODEL_FILES = {
    "YOLOv8n": "yolov8n.pt",
    "YOLO11n": "yolo11n.pt",
    "YOLO11s": "yolo11s.pt",
    "RT-DETR-L": "rtdetr-l.pt",
}


def download_models():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if all((DATA_DIR / f).is_file() for f in MODEL_FILES.values()):
        return

    token = st.secrets.get("KAGGLE_API_TOKEN", "")

    if not token:
        raise RuntimeError(
            "Chưa cấu hình KAGGLE_API_TOKEN trong Streamlit Secrets."
        )

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token

    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET_ID,
            "-p",
            str(DATA_DIR),
            "--unzip",
            "--force",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "Không tải được Dataset Kaggle:\n" + detail[-2000:]
        )

    missing = [
        f for f in MODEL_FILES.values()
        if not (DATA_DIR / f).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Không tìm thấy file ở thư mục gốc Dataset: "
            + ", ".join(missing)
            + ". Có thể các file đang nằm trong thư mục con."
        )


@st.cache_resource
def load_model(model_name):
    if model_name not in MODEL_FILES:
        raise ValueError(f"Tên mô hình không hợp lệ: {model_name}")

    download_models()

    model_path = DATA_DIR / MODEL_FILES[model_name]
    return YOLO(str(model_path))
