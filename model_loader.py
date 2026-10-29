
def download_models():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if all(
        (DATA_DIR / path).is_file()
        for path in MODEL_RELATIVE_PATHS.values()
    ):
        return

    token = st.secrets.get("KAGGLE_API_TOKEN", "")

    if not token:
        raise RuntimeError(
            "Thiếu KAGGLE_API_TOKEN trong Streamlit Secrets."
        )

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token

    # Bước 1: kiểm tra quyền truy cập và ID Dataset
    check = subprocess.run(
        [
            "kaggle",
            "datasets",
            "files",
            KAGGLE_DATASET_ID,
            "--page-size",
            "200",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if check.returncode != 0:
        detail = (check.stderr or check.stdout).strip()

        raise RuntimeError(
            f"Không truy cập được Dataset {KAGGLE_DATASET_ID}.\n"
            f"Chi tiết từ Kaggle:\n{detail[-2500:]}"
        )

    # Bước 2: tải Dataset
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
            f"Kaggle tìm thấy Dataset nhưng tải thất bại.\n"
            f"Chi tiết:\n{detail[-2500:]}"
        )

    # Bước 3: kiểm tra file trọng số
    missing = [
        name
        for name, relative_path in MODEL_RELATIVE_PATHS.items()
        if not (DATA_DIR / relative_path).is_file()
    ]

    if missing:
        found = list(DATA_DIR.rglob("selected_model.pt"))

        for name in missing:
            matches = [
                path for path in found
                if path.parent.name == "selected"
                and path.parent.parent.name == name
            ]

            if matches:
                target = DATA_DIR / MODEL_RELATIVE_PATHS[name]
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
            "Dataset đã tải nhưng thiếu mô hình: "
            + ", ".join(missing)
            + ". Hãy kiểm tra đường dẫn file bên trong Dataset."
        )
