
import os
import shutil
import subprocess
from pathlib import Path

import streamlit as st
from ultralytics import YOLO

DATA_DIR = Path("/tmp/yolodemo")

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
}


def download_models():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Đã có đủ model thì không tải lại
    if all(
        (DATA_DIR / path).is_file()
        for path in MODEL_RELATIVE_PATHS.values()
    ):
        return

    token = st.secrets.get("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Chưa cấu hình KAGGLE_API_TOKEN trong Streamlit Secrets."
        )

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token

    # Xóa ZIP cũ nếu lần tải trước bị lỗi
    for zip_file in DATA_DIR.glob("*.zip"):
        zip_file.unlink(missing_ok=True)

    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "thanhtuanh/yolodemo",
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
            "Kaggle tải Dataset thất bại.\n"
            + (result.stderr or result.stdout)[-2000:]
        )

    missing = [
        name
        for name, relative_path in MODEL_RELATIVE_PATHS.items()
        if not (DATA_DIR / relative_path).is_file()
    ]

    if missing:
        # Tìm các model nếu Dataset có thư mục bao ngoài
        found = list(DATA_DIR.rglob("selected_model.pt"))

        for name in missing[:]:
            expected = MODEL_RELATIVE_PATHS[name]
            matches = [
                p for p in found
                if p.parent.name == "selected"
                and p.parent.parent.name == name
            ]

            if matches:
                target = DATA_DIR / expected
                target.parent.mkdir(parents=True, exist_ok=True)
                if matches[0] != target:
                    shutil.copy2(matches[0], target)

        missing = [
            name
            for name, relative_path in MODEL_RELATIVE_PATHS.items()
            if not (DATA_DIR / relative_path).is_file()
        ]

    if missing:
        raise FileNotFoundError(
            "Không tìm thấy model: " + ", ".join(missing)
        )


@st.cache_resource
def load_model(model_name):
    download_models()
    model_path = DATA_DIR / MODEL_RELATIVE_PATHS[model_name]
    return YOLO(str(model_path))
