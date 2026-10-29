```python
import os
import subprocess
from pathlib import Path

import streamlit as st
from ultralytics import YOLO

# Thư mục lưu mô hình sau khi tải từ Kaggle
DATA_DIR = Path("/tmp/tomato_models")

# ID Dataset Kaggle
KAGGLE_DATASET_ID = "thanhtuanh/tomato-detection-models"

# Tên mô hình và tên file trọng số tương ứng trong Dataset.
# Điều chỉnh đường dẫn theo cấu trúc file thực tế trên Kaggle.
MODEL_FILES = {
    "YOLOv8n": "yolov8n.pt",
    "YOLO11n": "yolo11n.pt",
    "YOLO11s": "yolo11s.pt",
    "RT-DETR-L": "rtdetr-l.pt",
}


def download_models():
    """Tải Dataset Kaggle và kiểm tra các file trọng số."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Nếu cả bốn file đã có, không tải lại
    if all((DATA_DIR / filename).is_file()
           for filename in MODEL_FILES.values()):
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
            "Không tải được Dataset Kaggle.\n"
            f"Chi tiết: {detail[-2000:]}"
        )

    missing = [
        filename
        for filename in MODEL_FILES.values()
        if not (DATA_DIR / filename).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Dataset đã tải nhưng không tìm thấy các file: "
            + ", ".join(missing)
            + ". Hãy kiểm tra cấu trúc file trên Kaggle."
        )


@st.cache_resource
def load_model(model_name):
    """Tải và lưu mô hình trong bộ nhớ để tái sử dụng."""

    if model_name not in MODEL_FILES:
        raise ValueError(f"Tên mô hình không hợp lệ: {model_name}")

    download_models()

    model_path = DATA_DIR / MODEL_FILES[model_name]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file mô hình: {model_path}"
        )

    return YOLO(str(model_path))
```
