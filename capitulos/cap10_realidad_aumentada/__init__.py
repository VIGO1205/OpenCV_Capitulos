import streamlit as st
import cv2
import numpy as np
import io
from utils.common import load_image

def _to_png_bytes(img_bgr):
    """Convierte imagen BGR a bytes PNG para descarga"""
    is_success, buffer = cv2.imencode(".png", img_bgr)
    return io.BytesIO(buffer.tobytes()) if is_success else None

def run():
    st.markdown("### 🌟 Generación y Detección de Marcadores ArUco")
    st.markdown("Genera un marcador ArUco y detecta su posición automáticamente.")

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
        </style>
    """, unsafe_allow_html=True)

    # === CONTROLES ===
    col_gen, col_opts = st.columns([1.4, 1])
    with col_gen:
        marker_id = st.number_input("ID del marcador (0..49)", min_value=0, max_value=49, value=0, step=1)
        marker_size_px = st.slider("Tamaño (px)", 100, 800, 400, step=50)
        generar = st.button("🚀 Generar y Detectar Marcador", use_container_width=True)

    with col_opts:
        use_uploaded = st.checkbox("Probar con imagen subida", value=False)
        if use_uploaded:
            st.markdown('<div class="custom-uploader">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Selecciona una imagen con marcador ArUco", 
                type=["jpg", "jpeg", "png"], 
                key="cap10_upload"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            uploaded_file = None

    if not generar:
        st.info("👆 Presiona **Generar y Detectar Marcador** para comenzar.")
        return

    # === GENERAR MARCADOR ===
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, int(marker_id), int(marker_size_px))
        
        st.markdown("#### 🆕 Marcador Generado")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(marker_img, caption=f"Marcador ID: {marker_id}", width=300)
        with col2:
            st.info(f"""
            **Información del marcador:**
            - **ID**: {marker_id}
            - **Tamaño**: {marker_size_px}x{marker_size_px} px
            - **Diccionario**: DICT_4X4_50
            - **Total IDs disponibles**: 0-49
            """)
            
            # Botón de descarga del marcador
            marker_png = _to_png_bytes(marker_img)
            if marker_png:
                st.download_button(
                    "📥 Descargar Marcador",
                    marker_png,
                    f"aruco_marker_{marker_id}.png",
                    mime="image/png"
                )
    except Exception as e:
        st.error(f"❌ Error generando marcador: {e}")
        return

    # === PREPARAR IMÁGENES PARA DETECCIÓN ===
    images_to_test = []
    
    # Agregar imagen subida si existe
    if use_uploaded and uploaded_file:
        try:
            img_uploaded = load_image(uploaded_file)
            img_bgr = cv2.cvtColor(img_uploaded, cv2.COLOR_RGB2BGR)
            images_to_test.append(("Imagen Subida", img_bgr))
        except Exception as e:
            st.warning(f"⚠️ No se pudo leer la imagen subida: {e}")

    # Agregar imagen generada (siempre se prueba)
    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    images_to_test.insert(0, ("Marcador Generado", marker_bgr))

    # === CONFIGURAR DETECTOR ===
    try:
        if hasattr(cv2.aruco, "DetectorParameters"):
            params = cv2.aruco.DetectorParameters()
        elif hasattr(cv2.aruco, "DetectorParameters_create"):
            params = cv2.aruco.DetectorParameters_create()
        else:
            params = None
    except Exception:
        params = None

    # === DETECCIÓN DE MARCADORES ===
    st.markdown("---")
    st.markdown("#### 🔍 Resultados de Detección")
    
    detected_count = 0
    for label, img_bgr in images_to_test:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = None, None, None
        try:
            # OpenCV 4.7+ usa ArucoDetector
            if hasattr(cv2.aruco, "ArucoDetector"):
                detector = cv2.aruco.ArucoDetector(aruco_dict, params)
                corners, ids, rejected = detector.detectMarkers(gray)
            # OpenCV 4.0-4.6 usa detectMarkers
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        except Exception as e:
            st.error(f"❌ Error en detección de {label}: {e}")
            continue

        # Verificar si se detectó algo
        if ids is None or len(ids) == 0:
            st.warning(f"⚠️ No se detectaron marcadores en: **{label}**")
            if rejected is not None and len(rejected) > 0:
                st.info(f"Se encontraron {len(rejected)} candidatos rechazados. Intenta mejorar la iluminación o el enfoque.")
            continue

        # Se detectó al menos un marcador
        detected_count += 1
        detected_ids = ids.flatten().tolist()
        st.success(f"✅ **{label}**: Detectados {len(detected_ids)} marcador(es) con ID(s): {detected_ids}")
        
        # Dibujar marcadores detectados
        output = img_bgr.copy()
        cv2.aruco.drawDetectedMarkers(output, corners, ids)

        # Dibujar caja 3D simple sobre cada marcador
        for i, corner in enumerate(corners):
            pts = corner[0].astype(int)
            
            # Calcular offset para la "altura" de la caja
            box_height = int(np.linalg.norm(pts[0] - pts[1]) * 0.3)
            offset = np.array([[0, -box_height]] * 4)
            top_pts = pts + offset
            
            # Base de la caja (verde)
            cv2.polylines(output, [pts], True, (0, 255, 0), 3)
            # Techo de la caja (azul)
            cv2.polylines(output, [top_pts], True, (255, 0, 0), 3)
            # Columnas verticales (rojo)
            for p1, p2 in zip(pts, top_pts):
                cv2.line(output, tuple(p1), tuple(p2), (0, 0, 255), 2)
            
            # Etiqueta con el ID
            center = tuple(pts.mean(axis=0).astype(int))
            cv2.putText(output, f"ID:{detected_ids[i]}", 
                       (center[0]-30, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Mostrar resultado
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        st.image(output_rgb, caption=f"🎯 Detección en: {label}", use_column_width=True)
        
        # Botón de descarga
        png = _to_png_bytes(output)
        if png:
            st.download_button(
                f"💾 Descargar resultado ({label})",
                png,
                f"aruco_detectado_{label.replace(' ', '_').lower()}.png",
                mime="image/png",
                key=f"download_{label}"
            )

    # === RESUMEN FINAL ===
    if detected_count == 0:
        st.error("❌ No se detectó ningún marcador en ninguna imagen.")
        st.info("""
        **Sugerencias:**
        - Asegúrate de que el marcador esté bien iluminado
        - Evita reflejos o sombras fuertes
        - El marcador debe estar completamente visible
        - Prueba con la imagen generada primero
        """)
    else:
        st.success(f"🎉 Se detectaron marcadores en {detected_count} imagen(es)")
        
    # === INFORMACIÓN ADICIONAL ===
    with st.expander("ℹ️ Información sobre Marcadores ArUco"):
        st.markdown("""
        ### ¿Qué son los marcadores ArUco?
        
        Los marcadores ArUco son patrones cuadrados en blanco y negro utilizados para:
        - **Realidad Aumentada**: Superponer objetos 3D
        - **Calibración de cámaras**: Estimación de pose
        - **Robótica**: Navegación y localización
        - **Tracking**: Seguimiento de objetos en tiempo real
        
        ### Características:
        - Detección rápida y robusta
        - Identificación única mediante ID
        - Estimación de pose 3D
        - Resistente a rotaciones y perspectivas
        
        ### Diccionario DICT_4X4_50:
        - **50 marcadores únicos** (IDs del 0 al 49)
        - Cuadrícula de **4×4 bits** internos
        - Ideal para aplicaciones con pocos marcadores
        """)