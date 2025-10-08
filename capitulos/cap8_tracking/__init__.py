import streamlit as st
import cv2
import numpy as np

def run():

    st.markdown("""
    <style>
    .selector-box select {
        background-color: #f5f5f5;
        border: 2px solid #d68b40;
        border-radius: 8px;
        padding: 8px 10px;
        font-size: 1rem;
        font-weight: 500;
        color: #333;
    }
    .selector-box select:hover {
        background-color: #fff5ec;
        border-color: #ff9c42;
    }
    .run-btn {
        background: linear-gradient(90deg, #f6d365, #fda085);
        color: #222 !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Elegir método
    st.markdown("### ⚙️ Selecciona el método de sustracción de fondo")
    with st.container():
        bs_method = st.selectbox(
            "Método de Sustracción:",
            ["MOG2", "KNN"],
            key="cap8_selector",
            help="Usa MOG2 para mejor detección de sombras o KNN para ambientes con movimiento suave."
        )

    if st.button("🎬 Iniciar Detección", use_container_width=True):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ No se pudo acceder a la cámara.")
            return
        
        # Crear sustractor
        if bs_method == "MOG2":
            backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        else:
            backSub = cv2.createBackgroundSubtractorKNN(detectShadows=True)

        stframe1 = st.empty()
        stframe2 = st.empty()

        st.info("Presiona **Q** para detener el rastreo.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Aplicar sustracción
            fgMask = backSub.apply(frame)

            # Morfología para limpieza
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

            # Detectar contornos (movimiento)
            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frame_copy = frame.copy()

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Filtrar ruido
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame_copy, "Movimiento", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Mostrar video y máscara
            col1, col2 = st.columns(2)
            with col1:
                stframe1.image(frame_copy, channels="BGR",
                               caption="🎥 Detección de Movimiento", use_column_width=True)
            with col2:
                stframe2.image(fgMask, caption="🩶 Máscara de Primer Plano", use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        st.success("✅ Rastreo detenido correctamente")

    else:
        st.info("Pulsa el botón **Iniciar Detección** para comenzar.")