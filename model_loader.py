import os
import subprocess
from pathlib import Path

import streamlit as st
from ultralytics import YOLO

DATA_DIR = Path("/tmp/tomato_models")
DATASET_ID = "thanhtuanh/tomato-detection-models"

MODEL_FILES = {
    "YOLOv8n": "yolov8n.pt",
    "YOLO11n": "yolo11n.pt",
    "YOLO11s": "yolo11s.pt",
    "RT-DETR-L": "rtdetr-l.pt",
}


def kaggle_env():
    token = st.secrets.get("KAGGLE_API_TOKEN", "")

    if not token:
        raise RuntimeError(
            "Thiếu KAGGLE_API_TOKEN trong Streamlit Secrets. "
            "Hãy kiểm tra token của tài khoản Kaggle có quyền "
            "truy cập Dataset."
        )

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    return env


def run_kaggle(args, env):
    result = subprocess.run(
        args,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()

        if "404" in detail:
            raise RuntimeError(
                "Kaggle API trả về 404 khi truy cập Dataset.\n\n"
                "Mã Dataset: " + DATASET_ID + "\n\n"
                "Hãy kiểm tra:\n"
                "1. Dataset có tồn tại và tài khoản API có quyền truy cập.\n"
                "2. KAGGLE_API_TOKEN thuộc đúng tài khoản Kaggle.\n"
                "3. Token được tạo tại Kaggle Settings > API.\n\n"
                "Chi tiết API: " + detail[-1500:]
            )

        raise RuntimeError(
            "Lệnh Kaggle thất bại:\n" + detail[-2000:]
        )

    return result.stdout


def download_models():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Không tải lại nếu đã có đủ bốn file ở thư mục gốc.
    if all((DATA_DIR / f).is_file() for f in MODEL_FILES.values()):
        return

    env = kaggle_env()

    # Kiểm tra Dataset trước khi tải để tránh tải dữ liệu lớn vô ích.
    listing = run_kaggle(
        [
            "kaggle",
            "datasets",
            "files",
            DATASET_ID,
            "--page-size",
            "100",
        ],
        env,
    )

    # Chỉ tải các checkpoint có tên ở thư mục gốc Dataset.
    # Không tải cả thư mục tomato_results 18 GB.
    for filename in MODEL_FILES.values():
        target = DATA_DIR / filename

        if target.is_file():
            continue

        run_kaggle(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                DATASET_ID,
                "-f",
                filename,
                "-p",
                str(DATA_DIR),
                "--force",
            ],
            env,
        )

        # Kaggle trả về file ZIP; giải nén file cần thiết.
        import zipfile

        zip_files = list(DATA_DIR.glob("*.zip"))
        extracted = False

        for archive in zip_files:
            try:
                with zipfile.ZipFile(archive) as zf:
                    members = zf.namelist()

                    for member in members:
                        if Path(member).name == filename:
                            destination = DATA_DIR / filename
                            with zf.open(member) as source:
                                destination.write_bytes(source.read())
                            extracted = True
                            break

                if extracted:
                    break

            except zipfile.BadZipFile:
                continue

        if not extracted and not target.is_file():
            raise FileNotFoundError(
                f"Không tải được {filename}.\n"
                "Danh sách file Dataset:\n" + listing[:3000]
            )


@st.cache_resource
def load_model(model_name):
    if model_name not in MODEL_FILES:
        raise ValueError(f"Tên mô hình không hợp lệ: {model_name}")

    download_models()

    model_path = DATA_DIR / MODEL_FILES[model_name]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file mô hình: {model_path}"
        )

    return YOLO(str(model_path))
