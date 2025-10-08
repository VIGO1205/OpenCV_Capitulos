import streamlit as st
import cv2
import numpy as np
from utils.common import load_image, download_button

def run():

    # === ESTILO DEL UPLOADER ===
    st.markdown("""
        <style>
        .custom-uploader div[data-testid="stFileUploader"] {
            background: linear-gradient(135deg, #f6d36520 0%, #fda08520 100%);
            border: 2px dashed #f6a56f;
            padding: 1rem;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .custom-uploader div[data-testid="stFileUploader"]:hover {
            background: linear-gradient(135deg, #f6d36530 0%, #fda08530 100%);
            box-shadow: 0 0 10px rgba(246,163,100,0.4);
            border-color: #fda085;
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
            key="cap7_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file:
            st.session_state['cap7_img'] = load_image(uploaded_file)

    with col2:
        if 'cap7_img' in st.session_state:
            img = st.session_state['cap7_img']
            st.image(img, caption="🖼️ Imagen Original", use_column_width=True)
            st.success("✅ Imagen cargada correctamente")
        else:
            st.info("👆 Carga una imagen para comenzar")

    if 'cap7_img' not in st.session_state:
        return

    # === SEGMENTACIÓN POR COLOR ===
    st.markdown("### 🎯 Segmentación por Color (Espacio HSV)")

    img = st.session_state['cap7_img'].copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    color_preset = st.selectbox(
        "🎨 Selecciona un color:",
        ["Personalizado", "Rojo", "Verde", "Azul", "Amarillo", "Cian", "Magenta"]
    )

    presets = {
        "Rojo": ([0, 100, 100], [10, 255, 255]),
        "Verde": ([40, 40, 40], [80, 255, 255]),
        "Azul": ([100, 100, 100], [130, 255, 255]),
        "Amarillo": ([20, 100, 100], [30, 255, 255]),
        "Cian": ([80, 100, 100], [100, 255, 255]),
        "Magenta": ([140, 100, 100], [170, 255, 255])
    }

    if color_preset == "Personalizado":
        st.markdown("Ajusta los rangos de color HSV:")
        col1, col2 = st.columns(2)
        with col1:
            h_min = st.slider("Hue Min", 0, 179, 0)
            s_min = st.slider("Sat Min", 0, 255, 100)
            v_min = st.slider("Val Min", 0, 255, 100)
        with col2:
            h_max = st.slider("Hue Max", 0, 179, 10)
            s_max = st.slider("Sat Max", 0, 255, 255)
            v_max = st.slider("Val Max", 0, 255, 255)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
    else:
        lower, upper = np.array(presets[color_preset][0]), np.array(presets[color_preset][1])

    mask = cv2.inRange(hsv, lower, upper)

    # Aplicar operaciones morfológicas para limpiar la máscara
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

    result = cv2.bitwise_and(img, img, mask=mask_cleaned)

    st.info(f"💡 Segmentando color: **{color_preset}**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img, caption="Original", use_column_width=True)
    with col2:
        st.image(mask_cleaned, caption="Máscara de Color", use_column_width=True)
    with col3:
        st.image(result, caption="Resultado Segmentado", use_column_width=True)

    # Métrica de porcentaje
    pixels_segmented = np.count_nonzero(mask_cleaned)
    total_pixels = mask_cleaned.shape[0] * mask_cleaned.shape[1]
    percentage = (pixels_segmented / total_pixels) * 100
    st.metric("Cobertura del color detectado", f"{percentage:.2f}%")

    download_button(result, "color_segmentation.png")