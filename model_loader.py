
import os
import shutil
import subprocess
from pathlib import Path

import streamlit as st
from ultralytics import YOLO

DATA_DIR = Path("/tmp/yolodemo")

# Dataset ID trên Kaggle
KAGGLE_DATASET_ID = "thanhtuanh/tomato-detection-models"

# Đường dẫn tương đối của 4 mô hình trong Dataset
MODEL_RELATIVE_PATHS = {
    "YOLOv8n": Path(
        "tomato_results_archive/YOLOv8n/selected/selected_model.pt"
    ),
    "YOLO11n": Path(
        "tomato_results_archive/YOLO11n/selected/selected_model.pt"
    ),
    "YOLO11s": Path(
        "tomato_results_archive/YOLO11s/selected/selected_model.pt"
    ),
    "RT-DETR-L": Path(
        "tomato_results_archive/RT-DETR-L/selected/selected_model.pt"
    ),
}


def download_models():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Không tải lại nếu đã có đủ 4 mô hình
    if all(
        (DATA_DIR / path).is_file()
        for path in MODEL_RELATIVE_PATHS.values()
    ):
        return

    token = st.secrets.get("KAGGLE_API_TOKEN")

    if not token:
        raise RuntimeError(
            "Chưa cấu hình KAGGLE_API_TOKEN trong "
            "Streamlit Secrets."
        )

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token

    # Tải và giải nén Dataset từ Kaggle
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
            "--quiet",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Tải Kaggle Dataset thất bại:\n"
            + (result.stderr or result.stdout)[-2000:]
        )

    # Kiểm tra các mô hình còn thiếu
    missing = [
        name
        for name, relative_path in MODEL_RELATIVE_PATHS.items()
        if not (DATA_DIR / relative_path).is_file()
    ]

    if missing:
        # Tìm các file selected_model.pt trong Dataset
        found = list(DATA_DIR.rglob("selected_model.pt"))

        for name in missing:
            matches = [
                p for p in found
                if p.parent.name == "selected"
                and p.parent.parent.name == name
            ]

            if matches:
                target = DATA_DIR / MODEL_RELATIVE_PATHS[name]
                target.parent.mkdir(parents=True, exist_ok=True)

                if matches[0] != target:
                    shutil.copy2(matches[0], target)

    # Kiểm tra lại sau khi tìm kiếm
    missing = [
        name
        for name, relative_path in MODEL_RELATIVE_PATHS.items()
        if not (DATA_DIR / relative_path).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Không tìm thấy các mô hình: "
            + ", ".join(missing)
            + ". Hãy kiểm tra đường dẫn file trong Dataset."
        )


@st.cache_resource
def load_model(model_name):
    download_models()

    if model_name not in MODEL_RELATIVE_PATHS:
        raise ValueError(f"Mô hình không hợp lệ: {model_name}")

    model_path = DATA_DIR / MODEL_RELATIVE_PATHS[model_name]

    return YOLO(str(model_path))
