import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from utils.common import load_image
import matplotlib.pyplot as plt
import io
import pandas as pd

def run():

    # === ESTILO DEL UPLOADER ===
    st.markdown("""
        <style>
        .custom-uploader div[data-testid="stFileUploader"] {
            background: linear-gradient(135deg, #a0e9ff20 0%, #67b6ff20 100%);
            border: 2px dashed #3da9fc;
            padding: 1rem;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .custom-uploader div[data-testid="stFileUploader"]:hover {
            background: linear-gradient(135deg, #a0e9ff30 0%, #67b6ff30 100%);
            box-shadow: 0 0 10px rgba(61,169,252,0.4);
            border-color: #3da9fc;
        }
        .custom-uploader label {
            font-weight: 600 !important;
            color: #222 !important;
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
            key="cap9_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file:
            st.session_state['cap9_img'] = load_image(uploaded_file)

    with col2:
        if 'cap9_img' in st.session_state:
            img = st.session_state['cap9_img']
            st.image(img, caption="🖼️ Imagen Original", use_container_width=True)
            st.success("✅ Imagen cargada correctamente")
        else:
            st.info("👆 Carga una imagen para comenzar")

    if 'cap9_img' not in st.session_state:
        return

    # === EXTRACCIÓN DE DESCRIPTORES ===
    st.markdown("### 🔍 Extracción de Descriptores con ORB")

    img = st.session_state['cap9_img'].copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) == 0:
        st.error("❌ No se detectaron descriptores. Prueba con otra imagen.")
        return

    img_kp = cv2.drawKeypoints(
        img, keypoints, None, color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # === CLUSTERIZACIÓN K-MEANS ===
    st.markdown("### 📊 Histograma de Palabras Visuales (BoW)")
    col_img, col_hist = st.columns([1.1, 1.9])

    with col_img:
        st.image(img_kp, caption=f"🔑 {len(keypoints)} puntos detectados (ORB)", width=320)

    with col_hist:
        k = st.slider("Número de clusters (K)", 4, 64, 16, step=1)
        descriptors = descriptors.astype(np.float32)

        with st.spinner("🧠 Agrupando descriptores con K-Means..."):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(descriptors)
            labels = kmeans.labels_

        # Crear histograma normalizado
        hist, _ = np.histogram(labels, bins=np.arange(k + 1))
        hist_norm = hist / np.sum(hist)

        # === VISUALIZAR HISTOGRAMA (matplotlib) ===
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(np.arange(k), hist_norm, color="#3da9fc")
        ax.set_title("Distribución de Palabras Visuales", fontsize=12, weight='bold')
        ax.set_xlabel("Cluster (Palabra Visual)")
        ax.set_ylabel("Frecuencia Normalizada")
        ax.grid(alpha=0.3)

        st.pyplot(fig)
        st.success("✅ Histograma generado correctamente")

    # === DESCARGAS ===
    st.markdown("### 💾 Descarga del Resultado")

    # Guardar como imagen PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    st.download_button(
        label="🖼️ Descargar Histograma (PNG)",
        data=buf,
        file_name="histograma_visual.png",
        mime="image/png"
    )

    # Guardar como CSV
    df = pd.DataFrame({
        "Cluster": np.arange(k),
        "Frecuencia": hist_norm
    })
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📄 Descargar Datos (CSV)",
        data=csv,
        file_name="histograma_visual.csv",
        mime="text/csv"
    )

    st.info("""
    💡 **Explicación:**
    - Cada punto clave se agrupa en un “cluster” visual (una “palabra visual”).
    - El histograma muestra la distribución de esas palabras visuales.
    - Esta técnica permite reconocer objetos o comparar imágenes similares.
    """)
