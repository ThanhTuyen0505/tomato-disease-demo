
import time

import pandas as pd
import streamlit as st
from PIL import Image

from model_loader import load_model

st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="wide",
)

st.title("🍅 Phát hiện bệnh trên lá cà chua")
st.caption("So sánh mô hình YOLOv8n, YOLO11n và YOLO11s")

with st.sidebar:
    st.header("Cấu hình")
    model_name = st.selectbox(
        "Chọn mô hình",
        ["YOLOv8n", "YOLO11n", "YOLO11s"],
    )
    confidence = st.slider(
        "Ngưỡng confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )
    image_size = st.selectbox(
        "Kích thước ảnh",
        [320, 416, 512, 640],
        index=3,
    )

uploaded_file = st.file_uploader(
    "Tải ảnh lá cà chua lên",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ảnh đầu vào")
        st.image(image, use_container_width=True)

    if st.button("🔍 Bắt đầu nhận diện", type="primary"):
        try:
            with st.spinner(f"Đang chạy {model_name}..."):
                model = load_model(model_name)

                start = time.perf_counter()
                result = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=image_size,
                    verbose=False,
                )[0]
                elapsed = (time.perf_counter() - start) * 1000

            # Ultralytics trả ảnh BGR, Streamlit hiển thị RGB
            annotated_image = result.plot()[..., ::-1]

            with col2:
                st.subheader(f"Kết quả — {model_name}")
                st.image(annotated_image, use_container_width=True)

            boxes = result.boxes
            count = len(boxes) if boxes is not None else 0
            max_conf = (
                float(boxes.conf.max().item())
                if count > 0 else 0.0
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Số đối tượng", count)
            m2.metric("Confidence cao nhất", f"{max_conf:.1%}")
            m3.metric("Thời gian suy luận", f"{elapsed:.1f} ms")

            if count > 0:
                rows = []
                for box in boxes:
                    class_id = int(box.cls.item())
                    xywhn = box.xywhn[0].tolist()

                    rows.append({
                        "Lớp dự đoán": result.names[class_id],
                        "Confidence": round(float(box.conf.item()), 4),
                        "x_center": round(xywhn[0], 4),
                        "y_center": round(xywhn[1], 4),
                        "width": round(xywhn[2], 4),
                        "height": round(xywhn[3], 4),
                    })

                st.subheader("Chi tiết phát hiện")
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning(
                    "Không phát hiện đối tượng. "
                    "Bạn có thể thử giảm ngưỡng confidence."
                )

        except Exception as exc:
            st.error(f"Đã xảy ra lỗi khi nhận diện: {exc}")
            st.info(
                "Hãy kiểm tra Kaggle Secrets, cấu trúc Dataset "
                "và log của ứng dụng nếu lỗi vẫn tiếp diễn."
            )
else:
    st.info("Tải một ảnh lá cà chua để bắt đầu nhận diện.")

st.divider()
st.caption(
    "Lưu ý: thời gian suy luận phụ thuộc cấu hình máy chủ. "
    "Các kết quả nhận diện là dự đoán của mô hình, không thay thế "
    "việc xác nhận bệnh bởi chuyên gia."
)
