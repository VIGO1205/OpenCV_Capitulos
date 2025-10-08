import streamlit as st
import cv2
import numpy as np
from utils.common import load_image, download_button

def run():

    if "cap2_img" not in st.session_state:
        st.session_state.cap2_img = None

    # === ESTILO PERSONALIZADO PARA UPLOADER ===
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
            key="cap2_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Limpieza automática si se borra la imagen
        if uploaded_file is None and st.session_state.cap2_img is not None:
            st.session_state.cap2_img = None
        elif uploaded_file is not None:
            st.session_state.cap2_img = load_image(uploaded_file)

    with col2:
        if st.session_state.cap2_img is not None:
            st.image(
                st.session_state.cap2_img,
                caption="🖼️ Imagen Original",
                use_container_width=True
            )
            st.success("✅ Imagen cargada correctamente")
        else:
            st.info("👆 Carga una imagen para comenzar")

    # Si no hay imagen cargada, detener
    if st.session_state.cap2_img is None:
        return

    # === PROCESAMIENTO CON CANNY ===
    st.markdown("### ✨ Detección de Bordes (Canny)")

    col1, col2 = st.columns([1, 1.6])
    with col1:
        t1 = st.slider("Umbral Inferior", 0, 255, 50)
        t2 = st.slider("Umbral Superior", 0, 255, 150)
        aperture = st.select_slider("Tamaño del Kernel Sobel", [3, 5, 7], value=3)
        st.info("💡 Ajusta los umbrales para detectar más o menos bordes")

    with col2:
        img = st.session_state.cap2_img
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, t1, t2, apertureSize=aperture)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        st.image(
            result,
            caption=f"Bordes detectados (T1={t1}, T2={t2})",
            use_container_width=True
        )
        download_button(result, "canny_edges.png")
