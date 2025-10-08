import streamlit as st
import cv2
import numpy as np
from utils.common import load_image, download_button

def run():

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
            key="cap6_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file:
            st.session_state['cap6_img'] = load_image(uploaded_file)

    with col2:
        if 'cap6_img' in st.session_state:
            img = st.session_state['cap6_img']
            st.image(img, caption="🖼️ Imagen Original", use_container_width=True)
            st.success("✅ Imagen cargada correctamente")
        else:
            st.info("👆 Carga una imagen para comenzar")

    if 'cap6_img' not in st.session_state:
        return

    # === CÁLCULO DEL MAPA DE ENERGÍA ===
    st.markdown("### 🔥 Mapa de Energía (Gradiente Sobel)")
    st.write("El mapa de energía resalta las regiones con más detalles o bordes importantes en la imagen.")

    img = st.session_state['cap6_img']
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Calcular gradientes con Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.abs(sobelx) + np.abs(sobely)

    # Normalizar para visualizar
    energy_normalized = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX)
    energy_colored = cv2.applyColorMap(energy_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    energy_colored = cv2.cvtColor(energy_colored, cv2.COLOR_BGR2RGB)

    st.image(energy_colored, caption="🔥 Mapa de Energía (Regiones rojas = más importantes)", use_container_width=True)

    st.info("""
    💡 **Explicación:**
    - Las zonas **rojas** tienen cambios bruscos de intensidad → bordes, detalles, rostros.
    - Las zonas **azules** son áreas planas o de poco interés.
    - El algoritmo *Seam Carving* usa este mapa para decidir qué píxeles eliminar o conservar.
    """)

    download_button(energy_colored, "mapa_energia.png")