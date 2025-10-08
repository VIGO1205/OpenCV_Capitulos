import streamlit as st
import cv2
import numpy as np

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
            # Tomar solo x,y para la transformación
            pt_2d = np.float32([[pt[0], pt[1]]]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(pt_2d, H)
            
            # Ajustar por la altura z
            x, y = transformed[0][0]
            z_offset = pt[2] * h * 0.01  # Escalar el offset Z
            points_2d.append([int(x), int(y - z_offset)])
        
        return np.array(points_2d, dtype=np.int32)
    
    def draw_pyramid_3d(self, frame, H, ref_shape):
        """Dibuja una pirámide 3D sobre el objeto detectado"""
        if H is None:
            return frame
        
        h, w = ref_shape[:2]
        
        # Definir vértices 3D de la pirámide
        # Base: 4 esquinas en z=0
        # Punta: centro en z=altura
        pyramid_3d = np.float32([
            [0, 0, 0],      # 0: esquina inferior izquierda
            [w, 0, 0],      # 1: esquina inferior derecha
            [w, h, 0],      # 2: esquina superior derecha
            [0, h, 0],      # 3: esquina superior izquierda
            [w/2, h/2, 80]  # 4: punta de la pirámide (altura 80)
        ])
        
        # Proyectar a 2D
        points_2d = self.project_3d_to_2d(pyramid_3d, H, ref_shape)
        
        # Dibujar la base de la pirámide (verde semitransparente)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points_2d[:4]], (0, 255, 0))
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Dibujar bordes de la base
        cv2.polylines(frame, [points_2d[:4]], True, (0, 255, 0), 2)
        
        # Dibujar aristas desde las esquinas hasta la punta
        for i in range(4):
            cv2.line(frame, tuple(points_2d[i]), tuple(points_2d[4]), (0, 0, 255), 2)
        
        return frame
    
    def draw_cube_3d(self, frame, H, ref_shape):
        """Dibuja un cubo 3D sobre el objeto detectado"""
        if H is None:
            return frame
        
        h, w = ref_shape[:2]
        
        # Definir vértices 3D del cubo
        # Base inferior (z=0) y superior (z=altura)
        cube_3d = np.float32([
            # Base inferior
            [0, 0, 0],      # 0
            [w, 0, 0],      # 1
            [w, h, 0],      # 2
            [0, h, 0],      # 3
            # Base superior
            [0, 0, 60],     # 4
            [w, 0, 60],     # 5
            [w, h, 60],     # 6
            [0, h, 60]      # 7
        ])
        
        # Proyectar a 2D
        points_2d = self.project_3d_to_2d(cube_3d, H, ref_shape)
        
        # Dibujar caras del cubo con colores diferentes
        # Cara frontal (azul)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points_2d[[0,1,5,4]]], (200, 100, 0))
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Cara derecha (verde)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points_2d[[1,2,6,5]]], (0, 200, 100))
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Cara superior (rojo)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points_2d[[4,5,6,7]]], (0, 100, 200))
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Dibujar bordes del cubo
        # Base inferior
        cv2.polylines(frame, [points_2d[:4]], True, (0, 255, 0), 2)
        # Base superior
        cv2.polylines(frame, [points_2d[4:]], True, (0, 255, 0), 2)
        # Aristas verticales
        for i in range(4):
            cv2.line(frame, tuple(points_2d[i]), tuple(points_2d[i+4]), (0, 255, 0), 2)
        
        return frame
    
    def draw_axes_3d(self, frame, H, ref_shape):
        """Dibuja ejes 3D (X, Y, Z) sobre el objeto"""
        if H is None:
            return frame
        
        h, w = ref_shape[:2]
        axis_length = min(w, h) * 0.7
        
        # Definir puntos de los ejes
        axes_3d = np.float32([
            [0, 0, 0],                    # Origen
            [axis_length, 0, 0],          # Eje X (rojo)
            [0, axis_length, 0],          # Eje Y (verde)
            [0, 0, axis_length * 0.8]     # Eje Z (azul)
        ])
        
        # Proyectar a 2D
        points_2d = self.project_3d_to_2d(axes_3d, H, ref_shape)
        
        # Dibujar ejes
        origin = tuple(points_2d[0])
        cv2.line(frame, origin, tuple(points_2d[1]), (0, 0, 255), 3)  # X: Rojo
        cv2.line(frame, origin, tuple(points_2d[2]), (0, 255, 0), 3)  # Y: Verde
        cv2.line(frame, origin, tuple(points_2d[3]), (255, 0, 0), 3)  # Z: Azul
        
        # Etiquetas
        cv2.putText(frame, 'X', tuple(points_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, 'Y', tuple(points_2d[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, 'Z', tuple(points_2d[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return frame


def run():
    st.markdown("### 🎨 Overlay de Objetos 3D sobre Video")
    st.markdown("Proyecta pirámides, cubos o ejes 3D sobre objetos en tiempo real")
    
    # Estilos
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
    
    # Inicializar
    if 'ar_system' not in st.session_state:
        st.session_state.ar_system = AR3DOverlay()
        st.session_state.ref_data = None
    
    # Configuración
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded = st.file_uploader("📸 Sube imagen de referencia", type=['jpg', 'jpeg', 'png'])
    
    with col2:
        object_type = st.selectbox("Objeto 3D", ["Pirámide", "Cubo", "Ejes XYZ"])
    
    # Procesar imagen de referencia
    if uploaded:
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Redimensionar si es muy grande
        max_dim = 300
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        kp, desc = st.session_state.ar_system.get_features(img)
        
        if kp and desc is not None and len(kp) >= 10:
            st.session_state.ref_data = {'img': img, 'kp': kp, 'desc': desc}
            
            # Mostrar imagen con keypoints
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
            st.error("❌ Pocas características. Usa imagen con más detalles (texto, logos, patrones)")
    
    st.markdown("---")
    
    # Verificar si hay referencia
    if st.session_state.ref_data is None:
        st.info("👆 Sube una imagen de referencia para comenzar")
        return
    
    # Controles
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    start = col_btn1.button("▶️ Iniciar", key="start")
    stop = col_btn2.button("⏹️ Detener", key="stop")
    reset = col_btn3.button("🔄 Reiniciar", key="reset")
    
    if reset:
        st.session_state.ref_data = None
        st.rerun()
    
    if start:
        video_placeholder = st.empty()
        info_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ No se pudo abrir la cámara")
            return
        
        ref = st.session_state.ref_data
        frame_count = 0
        detections = 0
        
        try:
            while frame_count < 500 and not stop:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
                
                # Encontrar homografía
                H, matches = st.session_state.ar_system.find_homography(
                    ref['kp'], ref['desc'], frame
                )
                
                # Dibujar objeto 3D según selección
                if H is not None:
                    if object_type == "Pirámide":
                        frame = st.session_state.ar_system.draw_pyramid_3d(
                            frame, H, ref['img'].shape
                        )
                    elif object_type == "Cubo":
                        frame = st.session_state.ar_system.draw_cube_3d(
                            frame, H, ref['img'].shape
                        )
                    else:  # Ejes XYZ
                        frame = st.session_state.ar_system.draw_axes_3d(
                            frame, H, ref['img'].shape
                        )
                    
                    detections += 1
                    status = f"✅ Tracking | {matches} matches"
                    color = (0, 255, 0)
                else:
                    status = "🔍 Buscando objeto..."
                    color = (255, 165, 0)
                
                # Info en frame
                cv2.putText(frame, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Mostrar
                video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                # Actualizar métricas cada 30 frames
                if frame_count % 30 == 0:
                    accuracy = (detections / max(frame_count, 1)) * 100
                    info_placeholder.metric("Precisión", f"{accuracy:.0f}%", 
                                          f"{detections}/{frame_count} frames")
                
                frame_count += 1
                
        finally:
            cap.release()
            st.success(f"✅ Tracking finalizado • {detections} detecciones en {frame_count} frames")
    
    # Información adicional
    with st.expander("ℹ️ Cómo funciona"):
        st.markdown("""
        ### Proyección 3D a 2D
        
        **Proceso:**
        1. **Detectar características** en la imagen de referencia
        2. **Encontrar matches** en cada frame del video
        3. **Calcular homografía** (transformación geométrica)
        4. **Proyectar puntos 3D** a coordenadas 2D del frame
        5. **Dibujar objeto 3D** con perspectiva correcta
        
        **Objetos disponibles:**
        - **Pirámide**: Base cuadrada + punta elevada
        - **Cubo**: 6 caras con diferentes colores
        - **Ejes XYZ**: Sistema de coordenadas 3D
        
        **Consejos:**
        - Imagen con buen contraste y detalles
        - Iluminación uniforme
        - Mantén el objeto visible y estable
        """)

if __name__ == "__main__":
    run()