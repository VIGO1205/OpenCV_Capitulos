import streamlit as st
import cv2
import numpy as np
from utils.common import load_image, download_button

def run():

    if "cap4_img" not in st.session_state:
        st.session_state.cap4_img = None

    # === CARGA DEL CLASIFICADOR ===
    @st.cache_resource
    def load_face_cascade():
        return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    face_cascade = load_face_cascade()

    # === ESTILO PERSONALIZADO ===
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
            key="cap4_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is None and st.session_state.cap4_img is not None:
            st.session_state.cap4_img = None
        elif uploaded_file is not None:
            st.session_state.cap4_img = load_image(uploaded_file)

    with col2:
        if st.session_state.cap4_img is not None:
            st.image(st.session_state.cap4_img, caption="🖼️ Imagen Original", use_container_width=True)
            st.success("✅ Imagen cargada correctamente")
        else:
            st.info("👆 Carga una imagen para comenzar")

    if st.session_state.cap4_img is None:
        return

    # === DETECCIÓN DE ROSTROS ===
    st.markdown("### 🔍 Detección de Rostros")

    col1, col2 = st.columns([1, 1.6])
    with col1:
        scale_factor = st.slider("Factor de escala", 1.1, 2.0, 1.3, 0.1)
        min_neighbors = st.slider("Vecinos mínimos", 1, 10, 5)
        st.info("💡 Ajusta los parámetros para mejorar la detección según la imagen")

    with col2:
        img = st.session_state.cap4_img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img, "Rostro", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        st.image(
            img,
            caption=f"👤 {len(faces)} rostro(s) detectado(s)",
            use_container_width=True
        )
        download_button(img, "rostros_detectados.png")

        if len(faces) == 0:
            st.warning("😐 No se detectaron rostros. Prueba otra imagen o ajusta los parámetros.")
        else:
            st.success(f"✅ Se detectaron {len(faces)} rostro(s)")