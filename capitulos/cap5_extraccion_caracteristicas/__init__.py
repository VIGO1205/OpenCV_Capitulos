import streamlit as st
import cv2
import numpy as np
from utils.common import load_image, download_button

def run():

    if "cap5_img" not in st.session_state:
        st.session_state.cap5_img = None

    # === ESTILO DEL UPLOADER ===
    st.markdown("""
        <style>
        .custom-uploader div[data-testid="stFileUploader"] {
            background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
            border: 2px dashed #8b7ce6;
            padding: 1rem;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .custom-uploader div[data-testid="stFileUploader"]:hover {
            background: linear-gradient(135deg, #667eea30 0%, #764ba230 100%);
            box-shadow: 0 0 10px rgba(102,126,234,0.4);
            border-color: #667eea;
        }
        .custom-uploader label {
            font-weight: 600 !important;
            color: #333 !important;
            font-size: 1.05rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # === CARGA DE IMAGEN ===
    st.markdown("### 📤 Carga de Imagen")
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown('<div class="custom-uploader">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Selecciona una imagen:",
            type=["jpg", "jpeg", "png"],
            key="cap5_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is None and st.session_state.cap5_img is not None:
            st.session_state.cap5_img = None
        elif uploaded_file is not None:
            st.session_state.cap5_img = load_image(uploaded_file)

    with col2:
        if st.session_state.cap5_img is not None:
            st.image(st.session_state.cap5_img, caption="🖼️ Imagen Original", use_column_width=True)
            st.success("✅ Imagen cargada correctamente")
        else:
            st.info("👆 Carga una imagen para comenzar")

    if st.session_state.cap5_img is None:
        return

    # === DETECCIÓN ORB ===
    st.markdown("### 🔑 Extracción de Características con ORB")

    col1, col2 = st.columns([1, 1.6])
    with col1:
        num_features = st.slider("Número de características (keypoints)", 100, 2000, 500, 100)
        st.info("💡 ORB combina FAST y BRIEF para detectar puntos robustos y rápidos")

    with col2:
        img = st.session_state.cap5_img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(nfeatures=num_features)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        img_with_keypoints = cv2.drawKeypoints(
            img,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        st.image(
            img_with_keypoints,
            caption=f"🔍 {len(keypoints)} puntos clave detectados",
            use_column_width=True
        )
        download_button(img_with_keypoints, "orb_keypoints.png")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Número de Keypoints", len(keypoints))
        with col_b:
            if descriptors is not None:
                st.metric("Tamaño Descriptor", descriptors.shape[1])