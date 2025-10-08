import streamlit as st
from pathlib import Path
import sys
import importlib

# === CONFIGURACIÓN DE RUTAS ===
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

# === CONFIGURACIÓN DE PÁGINA ===
st.set_page_config(
    page_title="OpenCV Capítulos",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === ESTILO PERSONALIZADO ===
def load_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem 3rem;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 10px 25px rgba(0,0,0,0.25);
            margin-bottom: 2rem;
        }
        .header-text h1 { color: white; font-size: 2.4rem; font-weight: 700; margin-bottom: 0.5rem; }
        .header-text p { color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-bottom: 0; }
        .header-logo { width: 110px; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3)); }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; padding: 0.6rem 2rem;
            border-radius: 8px; font-weight: 600; transition: all 0.3s ease;
        }
        .stButton>button:hover { transform: scale(1.05); box-shadow: 0 8px 20px rgba(102,126,234,0.35); }
        </style>
    """, unsafe_allow_html=True)

# === INFORMACIÓN DE CAPÍTULOS ===
CHAPTERS = {
    1: {"title": "Transformaciones Geométricas", "desc": "Traslaciones, rotaciones, escalado y transformaciones proyectivas", "icon": "🔄", "module": "capitulos.cap1_transformaciones", "state_keys": ["current_image"]},
    2: {"title": "Detección de Bordes y Filtros", "desc": "Filtros: blur, sharpen, Canny y Sobel", "icon": "🎭", "module": "capitulos.cap2_filtros", "state_keys": ["cap2_img"]},
    3: {"title": "Cartoonización de Imágenes", "desc": "Dibujos animados con filtros bilaterales y contornos", "icon": "🎨", "module": "capitulos.cap3_cartoonizacion", "state_keys": ["cap3_img"]},
    4: {"title": "Detección Facial y Corporal", "desc": "Rostros, ojos y sonrisas con Haar Cascades", "icon": "👤", "module": "capitulos.cap4_deteccion_facial", "state_keys": ["cap4_img"]},
    5: {"title": "Extracción de Características", "desc": "Keypoints con ORB, SIFT y SURF", "icon": "🔍", "module": "capitulos.cap5_extraccion_caracteristicas", "state_keys": ["cap5_img"]},
    6: {"title": "Seam Carving", "desc": "Redimensionamiento inteligente preservando contenido", "icon": "✂️", "module": "capitulos.cap6_seam_carving", "state_keys": []},
    7: {"title": "Segmentación de Imágenes", "desc": "GrabCut, Watershed y contornos avanzados", "icon": "🧩", "module": "capitulos.cap7_segmentacion", "state_keys": []},
    8: {"title": "Rastreo de Objetos", "desc": "Tracking con CAMShift y optical flow", "icon": "🎯", "module": "capitulos.cap8_tracking", "state_keys": []},
    9: {"title": "Reconocimiento de Objetos", "desc": "SVM, Bag of Words y descriptores", "icon": "🤖", "module": "capitulos.cap9_reconocimiento", "state_keys": []},
    10: {"title": "Realidad Aumentada", "desc": "Overlay 3D con pose estimation", "icon": "🌟", "module": "capitulos.cap10_realidad_aumentada", "state_keys": []},
    11: {"title": "Redes Neuronales (ANN-MLP)", "desc": "Clasificación con redes neuronales multicapa", "icon": "🧠", "module": "capitulos.cap11_redes_neuronales", "state_keys": []},
}

# === FUNCIÓN PRINCIPAL ===
def main():
    load_custom_css()

    # Sidebar
    with st.sidebar:
        st.markdown("### 📚 Navegación")
        selected_chapter = st.selectbox(
            "Selecciona un capítulo:",
            options=[0] + list(CHAPTERS.keys()),
            format_func=lambda x: "🏠 Inicio" if x == 0 else f"{CHAPTERS[x]['icon']} Cap. {x}: {CHAPTERS[x]['title']}"
        )
        st.markdown("---")
        st.markdown("### ℹ️ Información")
        st.info("💡 Usa imágenes claras para mejores resultados")

    # Limpieza selectiva del capítulo anterior
    prev_chap = st.session_state.get("last_chapter")
    if prev_chap != selected_chapter:
        if prev_chap in CHAPTERS:
            for key in CHAPTERS[prev_chap]["state_keys"]:
                if key in st.session_state:
                    del st.session_state[key]
        st.session_state.last_chapter = selected_chapter

    # Mostrar Home o Capítulo
    if selected_chapter == 0:
        show_home()
    else:
        load_chapter(selected_chapter)

# === HOME ===
def show_home():
    st.markdown("""
        <div class="main-header">
            <div class="header-text">
                <h1>🎨 Proyecto OpenCV con Streamlit</h1>
                <p>Operaciones más destacadas del libro <b>“OpenCV 3.x with Python By Example”</b></p>
            </div>
            <img src="https://upload.wikimedia.org/wikipedia/commons/3/32/OpenCV_Logo_with_text_svg_version.svg" 
                 alt="OpenCV Logo" class="header-logo">
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚀 El Sistema nos permitirá:")
    st.markdown("""
    - Aplicar **transformaciones y filtros** sobre imágenes reales  
    - Explorar **detección facial y extracción de características**  
    - Experimentar con **segmentación, tracking y redes neuronales**  
    """)

# === CARGA DE CAPÍTULOS ===
def load_chapter(chapter_num):
    chapter_info = CHAPTERS[chapter_num]

    # Header dentro de contenedor con key único
    with st.container(key=f"header_{chapter_num}"):
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 1.2rem; border-radius: 12px; margin-bottom: 2rem;">
                <h2 style="color: white; margin: 0;">
                    {chapter_info['icon']} Capítulo {chapter_num}: {chapter_info['title']}
                </h2>
                <p style="color: rgba(255,255,255,0.9); margin-top: 0.3rem;">
                    {chapter_info['desc']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Importar y ejecutar capítulo
    try:
        module = importlib.import_module(chapter_info['module'])
        importlib.reload(module)
        if hasattr(module, "run"):
            module.run()
        else:
            st.warning(f"⚠️ El módulo del Capítulo {chapter_num} aún no está implementado.")
    except Exception as e:
        st.error(f"❌ Error al cargar el Capítulo {chapter_num}: {e}")

# === EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    main()
