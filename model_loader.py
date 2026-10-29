import os
import subprocess
import zipfile
from pathlib import Path

import streamlit as st
from ultralytics import YOLO

DATA_DIR = Path("/tmp/tomato_models")
KAGGLE_DATASET_ID = "thanhtuanh/tomato-detection-models"

# Tên mô hình và tên checkpoint dự kiến trong Dataset
MODEL_FILES = {
    "YOLOv8n": "yolov8n.pt",
    "YOLO11n": "yolo11n.pt",
    "YOLO11s": "yolo11s.pt",
    "RT-DETR-L": "rtdetr-l.pt",
}

# Các thư mục kết quả huấn luyện có thể chứa checkpoint đã chọn.
MODEL_SEARCH_HINTS = {
    "YOLOv8n": ["yolov8n", "yolov8"],
    "YOLO11n": ["yolo11n", "yolo11"],
    "YOLO11s": ["yolo11s", "yolo11"],
    "RT-DETR-L": ["rtdetr-l", "rtdetr"],
}


def run_kaggle_command(args, env):
    """Chạy lệnh Kaggle và trả về thông báo lỗi nếu thất bại."""
    result = subprocess.run(
        args,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"Lệnh Kaggle thất bại: {' '.join(args)}\n"
            f"Chi tiết: {detail[-2500:]}"
        )

    return result.stdout


def download_dataset():
    """Tải và giải nén Dataset nếu chưa có dữ liệu."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    token = st.secrets.get("KAGGLE_API_TOKEN", "")
    if not token:
        raise RuntimeError(
            "Thiếu KAGGLE_API_TOKEN trong Streamlit Cloud → "
            "App settings → Secrets."
        )

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token

    # Kiểm tra quyền truy cập và mã Dataset trước khi tải.
    run_kaggle_command(
        [
            "kaggle",
            "datasets",
            "files",
            KAGGLE_DATASET_ID,
            "--page-size",
            "100",
        ],
        env,
    )

    # Tải ZIP vào thư mục riêng.
    zip_path = DATA_DIR / "dataset.zip"

    run_kaggle_command(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET_ID,
            "-p",
            str(DATA_DIR),
            "-f",
            "tomato_results",
        ],
        env,
    )

    # Nếu Kaggle tải một ZIP có tên khác, tìm ZIP mới nhất.
    zip_files = list(DATA_DIR.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(
            "Không tìm thấy file ZIP sau khi tải Dataset."
        )

    # Giải nén tất cả ZIP trong Dataset; bỏ qua ZIP đã giải nén.
    for archive in zip_files:
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(DATA_DIR)
        except zipfile.BadZipFile:
            continue


def find_checkpoint(model_name):
    """Tìm checkpoint trong Dataset và các thư mục con."""
    expected = MODEL_FILES[model_name]

    # Ưu tiên file có tên chính xác, ở bất kỳ thư mục con nào.
    matches = list(DATA_DIR.rglob(expected))
    if matches:
        return matches[0]

    # Nếu tên file khác, chỉ tìm trong thư mục có tên phù hợp.
    hints = MODEL_SEARCH_HINTS[model_name]
    candidates = []

    for path in DATA_DIR.rglob("*.pt"):
        normalized = str(path).lower()
        if any(hint in normalized for hint in hints):
            candidates.append(path)

    # Ưu tiên checkpoint đã chọn hoặc best.pt.
    candidates.sort(
        key=lambda p: (
            "selected_model" not in p.name.lower(),
            p.name.lower() != "best.pt",
            len(str(p)),
        )
    )

    if candidates:
        return candidates[0]

    available = [
        str(p.relative_to(DATA_DIR))
        for p in DATA_DIR.rglob("*.pt")
    ]

    raise FileNotFoundError(
        f"Không tìm thấy checkpoint cho {model_name}.\n"
        f"Các file .pt tìm thấy: {available[:50] or 'Không có'}"
    )


@st.cache_resource
def load_model(model_name):
    if model_name not in MODEL_FILES:
        raise ValueError(f"Tên mô hình không hợp lệ: {model_name}")

    # Chỉ tải khi chưa có file dữ liệu.
    if not any(DATA_DIR.rglob("*.pt")):
        download_dataset()

    checkpoint = find_checkpoint(model_name)
    return YOLO(str(checkpoint))
