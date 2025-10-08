import streamlit as st
import cv2
import numpy as np
import tempfile

class AR3DOverlay:
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=1000)
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        self.matcher = cv2.FlannBasedMatcher(index_params, dict(checks=50))
        self.min_matches = 10
    
    def get_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detectAndCompute(gray, None)
    
    def find_homography(self, ref_kp, ref_desc, frame):
        kp, desc = self.get_features(frame)
        if desc is None or len(kp) < self.min_matches:
            return None, 0
        
        try:
            matches = self.matcher.knnMatch(ref_desc, desc, k=2)
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]
            
            if len(good) < self.min_matches:
                return None, 0
            
            src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H, len(good)
        except:
            return None, 0
    
    def project_3d_to_2d(self, points_3d, H, ref_shape):
        """Proyecta puntos 3D a 2D usando homografía"""
        h, w = ref_shape[:2]
        points_2d = []
        
        for pt in points_3d:
            pt_2d = np.float32([[pt[0], pt[1]]]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(pt_2d, H)
            
            x, y = transformed[0][0]
            z_offset = pt[2] * h * 0.01
            points_2d.append([int(x), int(y - z_offset)])
        
        return np.array(points_2d, dtype=np.int32)
    
    def draw_pyramid_3d(self, frame, H, ref_shape):
        """Dibuja una pirámide 3D sobre el objeto detectado"""
        if H is None:
            return frame
        
        h, w = ref_shape[:2]
        
        pyramid_3d = np.float32([
            [0, 0, 0],
            [w, 0, 0],
            [w, h, 0],
            [0, h, 0],
            [w/2, h/2, 80]
        ])
        
        points_2d = self.project_3d_to_2d(pyramid_3d, H, ref_shape)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points_2d[:4]], (0, 255, 0))
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        cv2.polylines(frame, [points_2d[:4]], True, (0, 255, 0), 2)
        
        for i in range(4):
            cv2.line(frame, tuple(points_2d[i]), tuple(points_2d[4]), (0, 0, 255), 2)
        
        return frame
    
    def draw_cube_3d(self, frame, H, ref_shape):
        """Dibuja un cubo 3D sobre el objeto detectado"""
        if H is None:
            return frame
        
        h, w = ref_shape[:2]
        
        cube_3d = np.float32([
            [0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],
            [0, 0, 60], [w, 0, 60], [w, h, 60], [0, h, 60]
        ])
        
        points_2d = self.project_3d_to_2d(cube_3d, H, ref_shape)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points_2d[[0,1,5,4]]], (200, 100, 0))
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points_2d[[1,2,6,5]]], (0, 200, 100))
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points_2d[[4,5,6,7]]], (0, 100, 200))
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        cv2.polylines(frame, [points_2d[:4]], True, (0, 255, 0), 2)
        cv2.polylines(frame, [points_2d[4:]], True, (0, 255, 0), 2)
        for i in range(4):
            cv2.line(frame, tuple(points_2d[i]), tuple(points_2d[i+4]), (0, 255, 0), 2)
        
        return frame
    
    def draw_axes_3d(self, frame, H, ref_shape):
        """Dibuja ejes 3D (X, Y, Z) sobre el objeto"""
        if H is None:
            return frame
        
        h, w = ref_shape[:2]
        axis_length = min(w, h) * 0.7
        
        axes_3d = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length * 0.8]
        ])
        
        points_2d = self.project_3d_to_2d(axes_3d, H, ref_shape)
        
        origin = tuple(points_2d[0])
        cv2.line(frame, origin, tuple(points_2d[1]), (0, 0, 255), 3)
        cv2.line(frame, origin, tuple(points_2d[2]), (0, 255, 0), 3)
        cv2.line(frame, origin, tuple(points_2d[3]), (255, 0, 0), 3)
        
        cv2.putText(frame, 'X', tuple(points_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, 'Y', tuple(points_2d[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, 'Z', tuple(points_2d[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return frame


def run():
    st.markdown("### 🎨 Overlay de Objetos 3D sobre Video")
    st.markdown("Proyecta pirámides, cubos o ejes 3D sobre objetos en tiempo real")
    
    st.markdown("""<style>
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; padding: 0.5rem 2rem;
        border-radius: 8px; font-weight: 600;
    }
    .info-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1rem; border-radius: 10px; border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    </style>""", unsafe_allow_html=True)
    
    if 'ar_system' not in st.session_state:
        st.session_state.ar_system = AR3DOverlay()
        st.session_state.ref_data = None
    
    # Configuración
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ref_image = st.file_uploader("📸 Imagen de referencia", type=['jpg', 'jpeg', 'png'], key="ref_img")
    
    with col2:
        object_type = st.selectbox("Objeto 3D", ["Pirámide", "Cubo", "Ejes XYZ"])
    
    # Procesar imagen de referencia
    if ref_image:
        img = cv2.imdecode(np.frombuffer(ref_image.read(), np.uint8), cv2.IMREAD_COLOR)
        
        max_dim = 300
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        kp, desc = st.session_state.ar_system.get_features(img)
        
        if kp and desc is not None and len(kp) >= 10:
            st.session_state.ref_data = {'img': img, 'kp': kp, 'desc': desc}
            
            img_kp = cv2.drawKeypoints(img, kp, None, (0, 255, 0), 
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            col_a, col_b = st.columns(2)
            col_a.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", width=280)
            col_b.image(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB), 
                       caption=f"✅ {len(kp)} características", width=280)
            
            st.markdown(f"""<div class="info-card">
                ✅ <b>Referencia lista</b> • {len(kp)} puntos detectados
            </div>""", unsafe_allow_html=True)
        else:
            st.error("❌ Pocas características. Usa imagen con más detalles")
    
    st.markdown("---")
    
    if st.session_state.ref_data is None:
        st.info("👆 Sube una imagen de referencia para comenzar")
        return
    
    # Opciones de entrada
    st.markdown("### 🎥 Fuente de video")
    video_source = st.radio(
        "Selecciona la fuente:",
        ["📹 Subir video", "📷 Cámara web (experimental)"],
        horizontal=True
    )
    
    if video_source == "📹 Subir video":
        video_file = st.file_uploader("Sube un video", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if video_file:
            # Guardar video temporalmente
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            video_path = tfile.name
            
            if st.button("▶️ Procesar video"):
                process_video(video_path, object_type)
    
    else:  # Cámara web
        st.warning("⚠️ La cámara web puede no funcionar en todos los navegadores/sistemas")
        
        col_btn1, col_btn2 = st.columns(2)
        start = col_btn1.button("▶️ Iniciar cámara")
        reset = col_btn2.button("🔄 Reiniciar")
        
        if reset:
            st.session_state.ref_data = None
            st.rerun()
        
        if start:
            process_webcam(object_type)


def process_video(video_path, object_type):
    """Procesa un video subido"""
    video_placeholder = st.empty()
    info_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ref = st.session_state.ref_data
    frame_count = 0
    detections = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar cada 2 frames para velocidad
            if frame_count % 2 == 0:
                frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
                
                H, matches = st.session_state.ar_system.find_homography(
                    ref['kp'], ref['desc'], frame
                )
                
                if H is not None:
                    if object_type == "Pirámide":
                        frame = st.session_state.ar_system.draw_pyramid_3d(frame, H, ref['img'].shape)
                    elif object_type == "Cubo":
                        frame = st.session_state.ar_system.draw_cube_3d(frame, H, ref['img'].shape)
                    else:
                        frame = st.session_state.ar_system.draw_axes_3d(frame, H, ref['img'].shape)
                    
                    detections += 1
                    status = f"✅ Tracking | {matches} matches"
                    color = (0, 255, 0)
                else:
                    status = "🔍 Buscando..."
                    color = (255, 165, 0)
                
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if frame_count % 30 == 0:
                    accuracy = (detections / max(frame_count, 1)) * 100
                    info_placeholder.metric("Precisión", f"{accuracy:.0f}%", f"{detections}/{frame_count}")
            
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            frame_count += 1
            
    finally:
        cap.release()
        st.success(f"✅ Procesado • {detections} detecciones en {frame_count} frames")


def process_webcam(object_type):
    """Procesa video de cámara web"""
    video_placeholder = st.empty()
    info_placeholder = st.empty()
    stop_button = st.button("⏹️ Detener")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ No se pudo abrir la cámara")
        st.info("""
        **Posibles soluciones:**
        - Verifica que ninguna otra aplicación esté usando la cámara
        - Intenta reiniciar el navegador
        - Usa la opción de "Subir video" en su lugar
        - Si estás en servidor remoto, la cámara no funcionará
        """)
        return
    
    ref = st.session_state.ref_data
    frame_count = 0
    detections = 0
    
    try:
        while frame_count < 300 and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
            
            H, matches = st.session_state.ar_system.find_homography(
                ref['kp'], ref['desc'], frame
            )
            
            if H is not None:
                if object_type == "Pirámide":
                    frame = st.session_state.ar_system.draw_pyramid_3d(frame, H, ref['img'].shape)
                elif object_type == "Cubo":
                    frame = st.session_state.ar_system.draw_cube_3d(frame, H, ref['img'].shape)
                else:
                    frame = st.session_state.ar_system.draw_axes_3d(frame, H, ref['img'].shape)
                
                detections += 1
                status = f"✅ Tracking | {matches} matches"
                color = (0, 255, 0)
            else:
                status = "🔍 Buscando..."
                color = (255, 165, 0)
            
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if frame_count % 30 == 0:
                accuracy = (detections / max(frame_count, 1)) * 100
                info_placeholder.metric("Precisión", f"{accuracy:.0f}%", f"{detections}/{frame_count}")
            
            frame_count += 1
            
    finally:
        cap.release()
        st.success(f"✅ Finalizado • {detections} detecciones")


if __name__ == "__main__":
    run()