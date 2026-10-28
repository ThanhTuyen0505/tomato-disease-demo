
import os
import subprocess
import zipfile
from pathlib import Path

import streamlit as st
from ultralytics import YOLO

DATA_DIR = Path("/tmp/yolodemo")

MODEL_RELATIVE_PATHS = {
    "YOLOv8n": Path("tomato_results_archive/YOLOv8n/selected/selected_model.pt"),
    "YOLO11n": Path("tomato_results_archive/YOLO11n/selected/selected_model.pt"),
    "YOLO11s": Path("tomato_results_archive/YOLO11s/selected/selected_model.pt"),
}


def download_models():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Không tải lại nếu đã có đủ 3 model
    if all((DATA_DIR / p).is_file() for p in MODEL_RELATIVE_PATHS.values()):
        return

    os.environ["KAGGLE_API_TOKEN"] = st.secrets["KAGGLE_API_TOKEN"]

    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "thanhtuanh/yolodemo",
            "-p", str(DATA_DIR),
            "--unzip",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Không tải được Kaggle Dataset.\n" + result.stderr[-3000:]
        )

    # Một số Dataset có thể có thêm thư mục bao ngoài.
    # Tìm file thực tế theo cấu trúc thư mục.
    for model_name, relative_path in MODEL_RELATIVE_PATHS.items():
        expected = DATA_DIR / relative_path
        if expected.is_file():
            continue

        matches = list(DATA_DIR.rglob("selected_model.pt"))
        matching = [
            p for p in matches
            if p.parts[-4:-1] == (
                MODEL_RELATIVE_PATHS[model_name].parts[-4:-1]
            )
        ]

        if matching:
            expected.parent.mkdir(parents=True, exist_ok=True)
            if matching[0] != expected:
                import shutil
                shutil.copy2(matching[0], expected)

    missing = [
        name for name, relative_path in MODEL_RELATIVE_PATHS.items()
        if not (DATA_DIR / relative_path).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Không tìm thấy checkpoint cho: " + ", ".join(missing)
            + ". Hãy kiểm tra cấu trúc file trong Kaggle Dataset."
        )


@st.cache_resource
def load_model(model_name):
    download_models()
    model_path = DATA_DIR / MODEL_RELATIVE_PATHS[model_name]
    return YOLO(str(model_path))
