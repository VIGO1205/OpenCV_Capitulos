import streamlit as st
import cv2
import numpy as np
import io
from utils.common import load_image

def _to_png_bytes(img_bgr):
    is_success, buffer = cv2.imencode(".png", img_bgr)
    return io.BytesIO(buffer.tobytes()) if is_success else None

def run():
    st.markdown("Genera un marcador ArUco y detecta su posición automáticamente.")

    # === CONTROLES ===
    col_gen, col_opts = st.columns([1.4, 1])
    with col_gen:
        marker_id = st.number_input("ID del marcador (0..49)", min_value=0, max_value=49, value=0, step=1)
        marker_size_px = st.slider("Tamaño (px)", 100, 800, 400, step=50)
        generar = st.button("🚀 Generar y Detectar Marcador (auto)")

    with col_opts:
        use_uploaded = st.checkbox("Probar con imagen subida", value=False)
        uploaded_file = st.file_uploader("Selecciona una imagen con marcador ArUco", type=["jpg", "jpeg", "png"], key="cap10_upload") if use_uploaded else None

    if not generar:
        st.info("Presiona **Generar y Detectar Marcador (auto)** para comenzar.")
        return

    # === GENERAR MARCADOR ===
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, int(marker_id), int(marker_size_px))
    except Exception as e:
        st.error(f"Error generando marcador: {e}")
        return

    st.image(marker_img, caption="🆕 Marcador generado", width=300)

    # === DETECCIÓN ===
    images_to_test = []
    if use_uploaded and uploaded_file:
        try:
            img_uploaded = load_image(uploaded_file)
            img_bgr = cv2.cvtColor(img_uploaded, cv2.COLOR_RGB2BGR)
            images_to_test.append(("Subida", img_bgr))
        except Exception as e:
            st.warning(f"No se pudo leer la imagen subida: {e}")

    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    images_to_test.insert(0, ("Generada", marker_bgr))

    params = None
    try:
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            params = cv2.aruco.DetectorParameters_create()
        else:
            params = cv2.aruco.DetectorParameters()
    except Exception:
        pass

    detected = False
    for label, img_bgr in images_to_test:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        corners, ids = None, None
        try:
            if hasattr(cv2.aruco, "ArucoDetector"):
                detector = cv2.aruco.ArucoDetector(aruco_dict, params)
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        except Exception:
            pass

        if ids is None or len(ids) == 0:
            st.warning(f"❌ No se detectaron marcadores en la imagen **{label}**.")
            continue

        detected = True
        st.success(f"✅ Detectado marcador ID(s): {ids.flatten().tolist()} en {label}")
        output = img_bgr.copy()
        cv2.aruco.drawDetectedMarkers(output, corners, ids)

        # Dibujar una caja 3D simple
        for corner in corners:
            pts = corner[0].astype(int)
            offset = -int(marker_size_px * 0.15)
            top_pts = pts + np.array([[0, offset]] * 4)
            cv2.polylines(output, [pts], True, (0, 255, 0), 2)
            cv2.polylines(output, [top_pts], True, (255, 0, 0), 2)
            for p1, p2 in zip(pts, top_pts):
                cv2.line(output, tuple(p1), tuple(p2), (0, 0, 255), 2)

        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption=f"Resultado ({label})", use_container_width=True)
        png = _to_png_bytes(output)
        st.download_button(f"💾 Descargar resultado ({label})", png, f"aruco_{label}.png", mime="image/png")

    if not detected:
        st.error("No se detectó ningún marcador. Prueba con la imagen generada sin modificar.")