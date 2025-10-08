import streamlit as st
import cv2
import numpy as np
from utils.common import load_image, download_button

def run():

    if "cap3_img" not in st.session_state:
        st.session_state.cap3_img = None

    # === ESTILO PARA UPLOADER ===
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
            key="cap3_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is None and st.session_state.cap3_img is not None:
            st.session_state.cap3_img = None
        elif uploaded_file is not None:
            st.session_state.cap3_img = load_image(uploaded_file)

    with col2:
        if st.session_state.cap3_img is not None:
            st.image(st.session_state.cap3_img, caption="🖼️ Imagen Original", use_container_width=True)
            st.success("✅ Imagen cargada correctamente")
        else:
            st.info("👆 Carga una imagen para comenzar")

    if st.session_state.cap3_img is None:
        return

    # === CARTOONIZACIÓN BÁSICA ===
    st.markdown("### 🖍️ Filtro de Cartoonización")

    col1, col2 = st.columns([1, 1.6])
    with col1:
        num_bilateral = st.slider("Iteraciones Bilateral", 1, 10, 5)
        d = st.slider("Diámetro Bilateral", 3, 15, 9)
        sigma_color = st.slider("Sigma Color", 10, 200, 75)
        sigma_space = st.slider("Sigma Espacio", 10, 200, 75)
        st.info("💡 Ajusta los parámetros para suavizar colores y resaltar bordes")

    with col2:
        img = st.session_state.cap3_img

        # Aplicar filtro bilateral repetidas veces
        img_color = img.copy()
        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(img_color, d, sigma_color, sigma_space)

        # Detectar bordes en escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=9,
            C=2
        )

        # Combinar color suavizado con bordes binarios
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        result = cv2.bitwise_and(img_color, edges_colored)

        st.image(result, caption="🎨 Imagen Cartoonizada", use_container_width=True)
        download_button(result, "cartoonized.png")