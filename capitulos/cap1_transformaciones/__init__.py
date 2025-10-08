import streamlit as st
import cv2
import numpy as np
from utils.common import load_image, download_button

def run():

    # Estado inicial
    if "current_image" not in st.session_state:
        st.session_state.current_image = None

    # === Estilo personalizado para el uploader ===
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
            color: #444;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # === Carga de imagen ===
    st.markdown("### 📤 Carga de Imagen")
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown('<div class="custom-uploader">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Selecciona una imagen:",
            type=["jpg", "jpeg", "png", "bmp"],
            key="uploader_cap1"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Si se borra la imagen (clic en ❌)
        if uploaded_file is None and st.session_state.current_image is not None:
            st.session_state.current_image = None

        # Si se carga una nueva
        elif uploaded_file is not None:
            st.session_state.current_image = load_image(uploaded_file)

    with col2:
        if st.session_state.current_image is not None:
            st.image(
                st.session_state.current_image,
                caption="🖼️ Imagen Original",
                use_container_width=True,
                output_format="auto"
            )
            st.success("✅ Imagen cargada correctamente")
        else:
            st.info("👆 Carga una imagen para comenzar")

    # Si no hay imagen, no mostrar tabs
    if st.session_state.current_image is None:
        return

    # === Tabs ===
    tabs = st.tabs(["↔️ Traslación", "🔄 Rotación", "📏 Escalado", "🎭 Perspectiva"])

    # === TRASLACIÓN ===
    with tabs[0]:
        col1, col2 = st.columns([1, 1.6])
        with col1:
            tx = st.slider("Desplazamiento X", -200, 200, 0)
            ty = st.slider("Desplazamiento Y", -200, 200, 0)

        with col2:
            img = st.session_state.current_image
            rows, cols = img.shape[:2]
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            result = cv2.warpAffine(img, M, (cols, rows))
            st.image(result, caption=f"Traslación ({tx}, {ty})", use_container_width=True)
            download_button(result, "traslacion.png")

    # === ROTACIÓN ===
    with tabs[1]:
        col1, col2 = st.columns([1, 1.6])
        with col1:
            angle = st.slider("Ángulo (°)", -180, 180, 0)
            scale = st.slider("Escala", 0.1, 2.0, 1.0, 0.1)

        with col2:
            img = st.session_state.current_image
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
            result = cv2.warpAffine(img, M, (cols, rows))
            st.image(result, caption=f"Rotada {angle}° (escala {scale})", use_container_width=True)
            download_button(result, "rotacion.png")

    # === ESCALADO ===
    with tabs[2]:
        col1, col2 = st.columns([1, 1.6])
        with col1:
            scale_factor = st.slider("Factor de Escala", 0.1, 3.0, 1.0, 0.1)
            interpolation = st.selectbox(
                "Método de Interpolación:",
                ["Linear", "Cubic", "Nearest", "Area"]
            )

        interp_map = {
            "Linear": cv2.INTER_LINEAR,
            "Cubic": cv2.INTER_CUBIC,
            "Nearest": cv2.INTER_NEAREST,
            "Area": cv2.INTER_AREA
        }

        with col2:
            img = st.session_state.current_image
            result = cv2.resize(
                img, None,
                fx=scale_factor, fy=scale_factor,
                interpolation=interp_map[interpolation]
            )
            st.image(result, caption=f"Escalada {scale_factor}x ({interpolation})", use_container_width=True)
            download_button(result, "escalado.png")

    # === PERSPECTIVA ===
    with tabs[3]:
        col1, col2 = st.columns([1, 1.6])
        with col1:
            st.markdown("Ajusta los puntos para deformar la imagen:")
            tl_x = st.slider("Superior Izq X", 0, 200, 0)
            tl_y = st.slider("Superior Izq Y", 0, 200, 0)
            br_x = st.slider("Inferior Der X", 0, 200, 0)
            br_y = st.slider("Inferior Der Y", 0, 200, 0)

        with col2:
            img = st.session_state.current_image
            rows, cols = img.shape[:2]
            src_points = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
            dst_points = np.float32([
                [tl_x, tl_y],
                [cols - br_x, tl_y],
                [tl_x, rows - br_y],
                [cols - br_x, rows - br_y]
            ])
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            result = cv2.warpPerspective(img, M, (cols, rows))
            st.image(result, caption="Transformación Perspectiva", use_container_width=True)
            download_button(result, "perspectiva.png")