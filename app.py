import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

st.set_page_config(
    page_title="Deteksi Karies Gigi",
    page_icon="🦷",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        st.error("❌ File best.pt tidak ditemukan!")
        st.stop()
    return YOLO(model_path)

model = load_model()

# ── Override nama kelas di sini ──────────────────────────────────
# Sesuaikan jumlahnya dengan kelas yang ada di model Anda
CLASS_NAMES = {
    0: "Karies",
    # Kalau ada lebih dari 1 kelas, tambahkan di sini:
    # 1: "Karies Awal",
    # 2: "Karies Lanjut",
}

def get_class_name(cls_id):
    """Ambil nama kelas dari mapping, fallback ke nama model jika tidak ada."""
    return CLASS_NAMES.get(int(cls_id), model.names[int(cls_id)])
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Pengaturan Deteksi")
    st.markdown("---")

    conf_threshold = st.slider(
        label="Confidence Threshold",
        min_value=0.01,
        max_value=1.0,
        value=0.25,
        step=0.01,
        help="Semakin rendah = lebih banyak deteksi"
    )

    iou_threshold = st.slider(
        label="IoU Threshold",
        min_value=0.01,
        max_value=1.0,
        value=0.45,
        step=0.01,
        help="Mengontrol overlap antar bounding box"
    )

    st.markdown("---")
    st.markdown("### 📋 Info Model")
    st.write("**Kelas yang dikenali:**")
    for idx in model.names:
        st.write(f"- `{idx}`: {get_class_name(idx)}")

    st.markdown("---")

st.title("Deteksi Karies Gigi")
st.write("Upload foto gigi untuk mendeteksi karies menggunakan YOLOv8")

uploaded_file = st.file_uploader(
    "Pilih gambar gigi",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)

    with st.spinner("🔍 Mendeteksi karies..."):
        results = model.predict(
            source=np.array(image),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

    result = results[0]

    # ── Gambar hasil dengan label yang sudah di-rename ──
    with col2:
        st.subheader("Hasil Deteksi")

        # Gambar manual agar nama kelas bisa di-override
        img_draw = np.array(image).copy()
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)

        COLORS = [
            (226, 75, 74),   # merah
            (29, 158, 117),  # hijau
            (55, 138, 221),  # biru
            (239, 159, 39),  # amber
        ]

        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            label = f"{get_class_name(cls)} {conf:.0%}"
            color = COLORS[cls % len(COLORS)]

            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_draw, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(img_draw, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        result_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, use_container_width=True)

    # ── Detail deteksi ──
    st.subheader("Detail Deteksi")
    boxes = result.boxes

    if len(boxes) > 0:
        st.success(f"✅ Ditemukan **{len(boxes)}** karies terdeteksi")

        col_h1, col_h2, col_h3, col_h4 = st.columns([2, 1, 1, 2])
        col_h1.markdown("**Kelas**")
        col_h2.markdown("**Confidence**")
        col_h3.markdown("**No.**")
        col_h4.markdown("**Koordinat (x1,y1,x2,y2)**")
        st.markdown("---")

        for i, box in enumerate(boxes):
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = get_class_name(cls)   # ← pakai nama yang sudah di-override
            x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]

            col1d, col2d, col3d, col4d = st.columns([2, 1, 1, 2])
            col1d.write(f"🔴 {label}")
            col2d.write(f"{conf:.2%}")
            col3d.write(f"#{i+1}")
            col4d.write(f"{x1}, {y1}, {x2}, {y2}")
    else:
        st.warning(
            f"⚠️ Tidak ada karies terdeteksi dengan conf={conf_threshold:.2f}. "
            f"Coba turunkan **Confidence Threshold** di sidebar."
        )

    with st.expander("🛠️ Debug Info"):
        st.write(f"Conf threshold: `{conf_threshold}`")
        st.write(f"IoU threshold: `{iou_threshold}`")
        st.write(f"Jumlah deteksi: `{len(result.boxes)}`")
        st.write(f"Nama kelas asli model: `{model.names}`")
        st.write(f"Nama kelas override: `{CLASS_NAMES}`")