import time

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from model_loader import load_model


# ============================================================
# 1. CẤU HÌNH TRANG
# ============================================================

st.set_page_config(
    page_title="Tomato AI | Phát hiện bệnh",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAMES = [
    "YOLOv8n",
    "YOLO11n",
    "YOLO11s",
    "RT-DETR-L",
]

CLASS_NAMES_VI = {
    "Early Blight": "Bệnh đốm vòng",
    "Healthy": "Lá khỏe mạnh",
    "Late Blight": "Bệnh mốc sương",
    "Leaf Miner": "Sâu vẽ bùa",
    "Leaf Mold": "Bệnh mốc lá",
    "Tomato Mosaic Virus": "Virus khảm cà chua",
    "Septoria Leaf Spot": "Bệnh đốm lá Septoria",
    "Red Spider Mite": "Nhện đỏ",
    "TYLCV": "Virus xoăn vàng lá cà chua",
}


# ============================================================
# 2. GIAO DIỆN
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap'
    );

    .stApp {
        background: #f5f8f4;
        font-family: 'Be Vietnam Pro', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(245, 248, 244, 0.9);
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(120deg, #174f32, #27834d);
        padding: 30px 32px;
        border-radius: 22px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(31, 91, 52, 0.12);
    }

    .hero h1 {
        color: white;
        font-size: clamp(27px, 4vw, 39px);
        font-weight: 800;
        margin: 0 0 10px 0;
    }

    .hero p {
        color: #e1f2e5;
        font-size: 15px;
        margin: 0;
        line-height: 1.7;
    }

    .section-title {
        color: #204d32;
        font-size: 19px;
        font-weight: 700;
        margin: 8px 0 14px 0;
    }

    .upload-hint {
        background: white;
        border: 1.5px dashed #8bb99a;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 12px;
        text-align: center;
    }

    .upload-hint .upload-icon {
        font-size: 35px;
        margin-bottom: 6px;
    }

    .upload-hint h3 {
        color: #245d38;
        font-size: 18px;
        margin: 0 0 7px 0;
    }

    .upload-hint p {
        color: #728075;
        font-size: 13px;
        margin: 0;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border: 1px solid #e0e9df;
        border-radius: 14px;
        padding: 12px;
    }

    div[data-testid="stFileUploaderDropzone"] {
        border: none;
        background: #fbfdfb;
    }

    .panel {
        background: white;
        border: 1px solid #e4ebe2;
        border-radius: 18px;
        padding: 18px;
        min-height: 100px;
    }

    .panel-title {
        color: #244e33;
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 14px;
    }

    div.stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        background: #247744;
        color: white;
        border: 1px solid #247744;
        font-weight: 700;
        font-size: 15px;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: #185c34;
        color: white;
        border-color: #185c34;
        transform: translateY(-1px);
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e4ebe2;
        padding: 16px;
        border-radius: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #617365;
    }

    [data-testid="stMetricValue"] {
        color: #215d37;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e4ebe2;
    }

    [data-testid="stSidebar"] h2 {
        color: #205b35;
    }

    .footer {
        color: #7a887d;
        font-size: 12px;
        text-align: center;
        line-height: 1.8;
        padding: 12px 0;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. TIÊU ĐỀ
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🍅 Tomato AI</h1>
        <p>
            Hệ thống phát hiện bệnh trên lá cà chua bằng trí tuệ nhân tạo.
            <br>
            Thử nghiệm và so sánh YOLOv8n, YOLO11n, YOLO11s và RT-DETR-L.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. THANH CẤU HÌNH
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Cấu hình nhận diện")
    st.caption("Tùy chỉnh mô hình trước khi phân tích ảnh.")

    model_name = st.selectbox(
        "🤖 Mô hình AI",
        MODEL_NAMES,
        index=1,
    )

    confidence = st.slider(
        "🎯 Ngưỡng confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help=(
            "Chỉ giữ các dự đoán có confidence "
            "đạt ngưỡng đã chọn."
        ),
    )

    image_size = st.selectbox(
        "🖼️ Kích thước ảnh suy luận",
        [320, 416, 512, 640],
        index=3,
    )

    st.divider()

    st.markdown("### 📌 Thông tin")
    st.caption("YOLOv8n · YOLO11n · YOLO11s · RT-DETR-L")
    st.caption("Đầu vào: JPG, JPEG, PNG, WEBP")
    st.caption("Đầu ra: Bounding box, lớp bệnh và confidence")


# ============================================================
# 5. TẢI ẢNH
# ============================================================

st.markdown(
    '<div class="section-title">📤 Tải ảnh cần kiểm tra</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="upload-hint">
        <div class="upload-icon">🌿</div>
        <h3>Chọn ảnh lá cà chua của bạn</h3>
        <p>
            Kéo thả ảnh vào khu vực bên dưới hoặc nhấn
            Browse files để chọn từ thiết bị.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Chọn ảnh",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
    key="tomato_image",
)


# ============================================================
# 6. HIỂN THỊ ẢNH VÀ NHẬN DIỆN
# ============================================================

if uploaded_file is not None:
    try:
        # Chuẩn hóa hướng ảnh theo EXIF và chuyển sang RGB.
        image = ImageOps.exif_transpose(
            Image.open(uploaded_file)
        ).convert("RGB")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(
                '<div class="panel-title">🖼️ Ảnh đầu vào</div>',
                unsafe_allow_html=True,
            )

            st.image(
                image,
                caption=f"Ảnh gốc · {image.width} × {image.height} px",
                use_container_width=True,
            )

        with col2:
            st.markdown(
                '<div class="panel-title">🔬 Kết quả nhận diện</div>',
                unsafe_allow_html=True,
            )

            result_placeholder = st.empty()

            # Chỉ hiển thị kết quả trước đó nếu toàn bộ cấu hình
            # vẫn giống lần nhận diện trước.
            previous = st.session_state.get("last_result")

            if (
                previous
                and previous.get("filename") == uploaded_file.name
                and previous.get("model_name") == model_name
                and previous.get("confidence") == confidence
                and previous.get("image_size") == image_size
                and previous.get("image_bytes") == uploaded_file.getvalue()
            ):
                result_placeholder.image(
                    previous["image"],
                    caption=f"Kết quả nhận diện · {previous['model_name']}",
                    use_container_width=True,
                )
            else:
                result_placeholder.info(
                    "Ảnh kết quả sẽ xuất hiện tại đây sau khi nhận diện."
                )

        st.markdown("")

        if st.button(
            "🔍 Bắt đầu nhận diện",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    f"Đang tải {model_name} và phân tích ảnh..."
                ):
                    # Thời gian tải model và thời gian suy luận
                    # được đo riêng để dễ theo dõi.
                    load_start = time.perf_counter()
                    model = load_model(model_name)
                    load_elapsed = time.perf_counter() - load_start

                    inference_start = time.perf_counter()

                    result = model.predict(
                        source=image,
                        conf=confidence,
                        imgsz=image_size,
                        verbose=False,
                    )[0]

                    inference_elapsed = (
                        time.perf_counter() - inference_start
                    ) * 1000

                # Ultralytics trả ảnh BGR; chuyển sang RGB để hiển thị.
                annotated_image = result.plot()[..., ::-1].copy()

                result_placeholder.image(
                    annotated_image,
                    caption=f"Kết quả nhận diện · {model_name}",
                    use_container_width=True,
                )

                st.session_state["last_result"] = {
                    "filename": uploaded_file.name,
                    "image_bytes": uploaded_file.getvalue(),
                    "model_name": model_name,
                    "confidence": confidence,
                    "image_size": image_size,
                    "image": annotated_image,
                }

                boxes = result.boxes
                count = len(boxes) if boxes is not None else 0

                max_conf = (
                    float(boxes.conf.max().item())
                    if count > 0
                    else 0.0
                )

                st.markdown("")
                st.markdown(
                    '<div class="section-title">📊 Thống kê kết quả</div>',
                    unsafe_allow_html=True,
                )

                m1, m2, m3 = st.columns(3, gap="medium")

                m1.metric(
                    "Đối tượng phát hiện",
                    count,
                )

                m2.metric(
                    "Confidence cao nhất",
                    f"{max_conf:.1%}",
                )

                m3.metric(
                    "Thời gian suy luận",
                    f"{inference_elapsed:.1f} ms",
                )

                st.caption(
                    f"Mô hình: {model_name} · "
                    f"Ngưỡng confidence: {confidence:.2f} · "
                    f"Image size: {image_size} px"
                )

                with st.expander("⏱️ Thông tin thời gian xử lý"):
                    st.write(
                        f"Thời gian tải/khởi tạo mô hình: "
                        f"{load_elapsed:.2f} giây"
                    )
                    st.write(
                        f"Thời gian suy luận: "
                        f"{inference_elapsed:.1f} ms"
                    )

                if count > 0:
                    names = result.names
                    rows = []

                    for box in boxes:
                        class_id = int(box.cls.item())
                        score = float(box.conf.item())

                        class_name = (
                            names.get(class_id, str(class_id))
                            if isinstance(names, dict)
                            else names[class_id]
                        )

                        class_name_vi = CLASS_NAMES_VI.get(
                            class_name,
                            class_name,
                        )

                        # Tọa độ chuẩn hóa YOLO: tâm x, tâm y,
                        # chiều rộng và chiều cao, trong khoảng 0–1.
                        xywhn = box.xywhn[0].tolist()

                        # Tọa độ pixel của bounding box.
                        xyxy = box.xyxy[0].tolist()

                        rows.append(
                            {
                                "Lớp bệnh": class_name_vi,
                                "Tên lớp": class_name,
                                "Confidence (%)": round(score * 100, 2),
                                "x_center": round(xywhn[0], 5),
                                "y_center": round(xywhn[1], 5),
                                "width": round(xywhn[2], 5),
                                "height": round(xywhn[3], 5),
                                "X1 (px)": round(xyxy[0], 1),
                                "Y1 (px)": round(xyxy[1], 1),
                                "X2 (px)": round(xyxy[2], 1),
                                "Y2 (px)": round(xyxy[3], 1),
                            }
                        )

                    df = pd.DataFrame(rows)

                    st.markdown("")
                    st.markdown(
                        '<div class="section-title">📋 Chi tiết phát hiện</div>',
                        unsafe_allow_html=True,
                    )

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "⬇️ Tải bảng kết quả CSV",
                        data=df.to_csv(
                            index=False
                        ).encode("utf-8-sig"),
                        file_name="tomato_detection_results.csv",
                        mime="text/csv",
                    )

                else:
                    st.warning(
                        "Không phát hiện được đối tượng ở ngưỡng hiện tại. "
                        "Hãy thử giảm confidence xuống 0.10–0.20, "
                        "kiểm tra ảnh có lá cà chua rõ nét và xác nhận "
                        "ứng dụng đang sử dụng đúng checkpoint đã huấn luyện."
                    )

            except Exception as exc:
                st.error("Đã xảy ra lỗi khi nhận diện.")
                st.code(str(exc))

                st.info(
                    "Nếu lỗi liên quan đến Kaggle, hãy kiểm tra token, "
                    "quyền truy cập dataset và đường dẫn checkpoint."
                )

    except Exception as exc:
        st.error(f"Không thể đọc ảnh này: {exc}")

else:
    st.info(
        "👆 Tải ảnh lá cà chua lên để xem ảnh và bắt đầu nhận diện."
    )


# ============================================================
# 7. CHÂN TRANG
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🍅 Tomato AI · Demo phát hiện bệnh trên lá cà chua
        <br>
        Kết quả là dự đoán của mô hình, không thay thế
        việc xác nhận bệnh bởi chuyên gia.
    </div>
    """,
    unsafe_allow_html=True,
)
