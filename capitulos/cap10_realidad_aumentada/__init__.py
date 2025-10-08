import cv2
import numpy as np

class ARTracker:
    def __init__(self):
        self.reference_image = None
        self.reference_kp = None
        self.reference_desc = None
        
        # Detector ORB
        self.detector = cv2.ORB_create(nfeatures=1000)
        
        # Matcher FLANN
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=6,
                           key_size=12,
                           multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        self.min_matches = 10
    
    def set_reference(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        self.reference_image = image.copy()
        self.reference_kp, self.reference_desc = self.detector.detectAndCompute(gray, None)
        
        if self.reference_desc is None:
            return False
        return True
    
    def track(self, frame):
        if self.reference_desc is None:
            return None, None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        kp, desc = self.detector.detectAndCompute(gray, None)
        
        if desc is None or len(kp) < self.min_matches:
            return None, None
        
        try:
            matches = self.matcher.knnMatch(self.reference_desc, desc, k=2)
        except:
            return None, None
        
        # Filtrar buenos matches
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good.append(m)
        
        if len(good) < self.min_matches:
            return None, None
        
        # Puntos para homografía
        src_pts = np.float32([self.reference_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return None, None
        
        # Calcular esquinas del objeto
        h, w = self.reference_image.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H)
        
        return transformed, H
    
    def overlay_pyramid(self, frame, corners, H):
        if corners is None or H is None:
            return frame
        
        # Obtener dimensiones de la referencia
        h, w = self.reference_image.shape[:2]
        
        # Vértices del cubo 3D (en coordenadas de la imagen de referencia)
        # Base: 4 esquinas en z=0
        # Top: 1 punto central en z=altura
        pyramid_3d = np.float32([
            [0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],  # Base
            [w/2, h/2, -h*0.8]  # Punta (altura negativa para que se vea arriba)
        ]).reshape(-1, 1, 3)
        
        # Proyectar puntos 3D a 2D usando la homografía
        # Aproximación simple: ignorar z para la transformación
        pyramid_2d = []
        for pt in pyramid_3d:
            pt_2d = np.float32([[pt[0][0], pt[0][1]]]).reshape(-1, 1, 2)
            # Para el punto elevado, ajustar manualmente
            if pt[0][2] != 0:  # Es la punta
                transformed = cv2.perspectiveTransform(pt_2d, H)
                # Mover hacia arriba según la altura
                offset_y = int(h * 0.5)
                transformed[0][0][1] -= offset_y
                pyramid_2d.append(transformed[0][0])
            else:
                transformed = cv2.perspectiveTransform(pt_2d, H)
                pyramid_2d.append(transformed[0][0])
        
        pyramid_2d = np.array(pyramid_2d, dtype=np.int32)
        
        # Dibujar la pirámide
        # Base (semitransparente)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pyramid_2d[:4]], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Bordes de la base
        cv2.polylines(frame, [pyramid_2d[:4]], True, (0, 255, 0), 2)
        
        # Líneas desde las esquinas a la punta
        for i in range(4):
            cv2.line(frame, tuple(pyramid_2d[i]), tuple(pyramid_2d[4]), (0, 0, 255), 2)
        
        return frame


def main():
    print("=== Realidad Aumentada - Tracking de Objetos ===\n")
    print("Controles:")
    print("  'r' - Capturar objeto de referencia")
    print("  'c' - Limpiar tracking")
    print("  ESC - Salir\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return
    
    tracker = ARTracker()
    selecting = False
    start_pt = None
    end_pt = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, start_pt, end_pt
        
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            start_pt = (x, y)
            end_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            end_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            end_pt = (x, y)
    
    cv2.namedWindow('AR Tracker')
    cv2.setMouseCallback('AR Tracker', mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
        display = frame.copy()
        
        # Tracking activo
        if tracker.reference_image is not None:
            corners, H = tracker.track(frame)
            
            if corners is not None:
                display = tracker.overlay_pyramid(display, corners, H)
                cv2.putText(display, "Tracking OK", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Buscando...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar selección
        if selecting and start_pt and end_pt:
            cv2.rectangle(display, start_pt, end_pt, (255, 255, 0), 2)
        
        cv2.imshow('AR Tracker', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            if start_pt and end_pt and not selecting:
                x1 = min(start_pt[0], end_pt[0])
                y1 = min(start_pt[1], end_pt[1])
                x2 = max(start_pt[0], end_pt[0])
                y2 = max(start_pt[1], end_pt[1])
                
                if x2 - x1 > 50 and y2 - y1 > 50:
                    roi = frame[y1:y2, x1:x2]
                    if tracker.set_reference(roi):
                        print("Objeto capturado")
                    else:
                        print("Error: No se detectaron características")
        elif key == ord('c'):
            tracker.reference_image = None
            tracker.reference_kp = None
            tracker.reference_desc = None
            print("Tracking reiniciado")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()