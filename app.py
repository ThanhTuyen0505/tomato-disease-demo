
import time

import pandas as pd
import streamlit as st
from PIL import Image

from model_loader import load_model

st.set_page_config(
    page_title="Tomato AI | Phát hiện bệnh",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== GIAO DIỆN ====================

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

# ==================== TIÊU ĐỀ ====================

st.markdown(
    """
    <div class="hero">
        <h1>🍅 Tomato AI</h1>
        <p>
            Hệ thống phát hiện bệnh trên lá cà chua bằng trí tuệ nhân tạo.
            <br>
            Thử nghiệm và so sánh YOLOv8n, YOLO11n và YOLO11s.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================== CẤU HÌNH ====================

with st.sidebar:
    st.markdown("## ⚙️ Cấu hình nhận diện")
    st.caption("Tùy chỉnh mô hình trước khi phân tích ảnh.")

    model_name = st.selectbox(
        "🤖 Mô hình AI",
        ["YOLOv8n", "YOLO11n", "YOLO11s"],
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
    st.caption("YOLOv8n · YOLO11n · YOLO11s")
    st.caption("Đầu vào: JPG, JPEG, PNG, WEBP")
    st.caption("Đầu ra: Bounding box và confidence")

# ==================== TẢI ẢNH ====================

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

# ==================== HIỂN THỊ ẢNH ====================

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("")
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

            # Khu vực kết quả sẽ được cập nhật sau khi dự đoán.
            result_placeholder = st.empty()

            if "last_result" in st.session_state:
                previous = st.session_state["last_result"]
                if previous.get("filename") == uploaded_file.name:
                    result_placeholder.image(
                        previous["image"],
                        caption=(
                            f"Kết quả {previous['model_name']}"
                        ),
                        use_container_width=True,
                    )
                else:
                    result_placeholder.info(
                        "Nhấn nút bên dưới để phân tích ảnh."
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
                    model = load_model(model_name)

                    start = time.perf_counter()
                    result = model.predict(
                        source=image,
                        conf=confidence,
                        imgsz=image_size,
                        verbose=False,
                    )[0]
                    elapsed = (
                        time.perf_counter() - start
                    ) * 1000

                # Ultralytics trả ảnh BGR, đổi về RGB.
                annotated_image = result.plot()[..., ::-1].copy()

                result_placeholder.image(
                    annotated_image,
                    caption=f"Kết quả nhận diện · {model_name}",
                    use_container_width=True,
                )

                st.session_state["last_result"] = {
                    "filename": uploaded_file.name,
                    "model_name": model_name,
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
                m1.metric("Đối tượng phát hiện", count)
                m2.metric(
                    "Confidence cao nhất",
                    f"{max_conf:.1%}",
                )
                m3.metric(
                    "Thời gian suy luận",
                    f"{elapsed:.1f} ms",
                )

                st.caption(
                    f"Mô hình: {model_name} · "
                    f"Ngưỡng confidence: {confidence:.2f} · "
                    f"Image size: {image_size}"
                )

                if count > 0:
                    rows = []

                    for box in boxes:
                        class_id = int(box.cls.item())
                        xywhn = box.xywhn[0].tolist()

                        rows.append(
                            {
                                "Lớp dự đoán": result.names[class_id],
                                "Confidence": round(
                                    float(box.conf.item()), 4
                                ),
                                "x_center": round(xywhn[0], 4),
                                "y_center": round(xywhn[1], 4),
                                "width": round(xywhn[2], 4),
                                "height": round(xywhn[3], 4),
                            }
                        )

                    st.markdown("")
                    st.markdown(
                        '<div class="section-title">📋 Chi tiết phát hiện</div>',
                        unsafe_allow_html=True,
                    )

                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                    csv_data = pd.DataFrame(rows).to_csv(
                        index=False
                    ).encode("utf-8-sig")

                    st.download_button(
                        "⬇️ Tải bảng kết quả CSV",
                        data=csv_data,
                        file_name="tomato_detection_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning(
                        "Không phát hiện đối tượng ở ngưỡng hiện tại. "
                        "Bạn có thể giảm confidence và thử lại."
                    )

            except Exception as exc:
                st.error(
                    f"Đã xảy ra lỗi khi nhận diện: {exc}"
                )
                st.info(
                    "Nếu lỗi liên quan đến Kaggle, hãy kiểm tra "
                    "Secrets và log của ứng dụng Streamlit Cloud."
                )

    except Exception as exc:
        st.error(f"Không thể đọc ảnh này: {exc}")

else:
    st.markdown("")
    st.info(
        "👆 Tải ảnh lá cà chua lên để xem ảnh và bắt đầu nhận diện."
    )

# ==================== CHÂN TRANG ====================

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
