import streamlit as st
import cv2
import numpy as np

class ARTracker:
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=500)
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        self.matcher = cv2.FlannBasedMatcher(index_params, dict(checks=50))
    
    def get_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detectAndCompute(gray, None)
    
    def track(self, ref_kp, ref_desc, frame, ref_shape):
        kp, desc = self.get_features(frame)
        if desc is None or len(kp) < 10:
            return frame
        
        try:
            matches = self.matcher.knnMatch(ref_desc, desc, k=2)
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]
            
            if len(good) < 10:
                return frame
            
            src = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            
            if H is None:
                return frame
            
            # Dibujar pirámide
            h, w = ref_shape[:2]
            base = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1, 1, 2)
            corners = cv2.perspectiveTransform(base, H)
            corners_int = np.int32(corners).reshape(-1, 2)
            
            # Calcular punta
            center = corners.mean(axis=0)[0]
            tip = (int(center[0]), int(center[1] - h * 0.6))
            
            # Base semitransparente
            overlay = frame.copy()
            cv2.fillPoly(overlay, [corners_int], (50, 200, 50))
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            
            # Aristas
            cv2.polylines(frame, [corners_int], True, (0, 255, 0), 3)
            for corner in corners_int:
                cv2.line(frame, tuple(corner), tip, (0, 100, 255), 2)
            
            cv2.putText(frame, "Tracking OK", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except:
            pass
        
        return frame


def run():
    st.markdown("### 🎯 Realidad Aumentada - Tracking 3D")
    
    st.markdown("""<style>
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; padding: 0.5rem 2rem;
        border-radius: 8px; font-weight: 600;
    }
    </style>""", unsafe_allow_html=True)
    
    if 'tracker' not in st.session_state:
        st.session_state.tracker = ARTracker()
        st.session_state.ref = None
    
    # Upload
    uploaded = st.file_uploader("📸 Imagen de referencia (logos, texto, patrones)", 
                               type=['jpg', 'jpeg', 'png'])
    
    if uploaded:
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        
        kp, desc = st.session_state.tracker.get_features(img)
        
        if kp and desc is not None:
            st.session_state.ref = {'img': img, 'kp': kp, 'desc': desc}
            
            img_kp = cv2.drawKeypoints(img, kp, None, (0,255,0), 
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            col1, col2 = st.columns(2)
            col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=300)
            col2.image(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB), 
                      caption=f"✅ {len(kp)} características", width=300)
        else:
            st.error("❌ Sin características detectadas")
    
    st.markdown("---")
    
    if st.session_state.ref is None:
        st.info("👆 Sube una imagen primero")
        return
    
    col1, col2 = st.columns(2)
    start = col1.button("▶️ Iniciar", key="start_btn")
    reset = col2.button("🔄 Reiniciar", key="reset_btn")
    
    if reset:
        st.session_state.ref = None
        st.rerun()
    
    if start:
        placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cámara no disponible")
            return
        
        ref = st.session_state.ref
        count = 0
        
        try:
            while count < 300:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
                frame = st.session_state.tracker.track(
                    ref['kp'], ref['desc'], frame, ref['img'].shape
                )
                
                placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                channels="RGB")
                count += 1
        finally:
            cap.release()
            st.success("✅ Finalizado")

if __name__ == "__main__":
    run()