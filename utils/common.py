import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def load_image(uploaded_file):
    """Carga imagen desde Streamlit uploader"""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return np.array(image)
    return None

def display_images(original, processed, titles=["Original", "Procesada"]):
    """Muestra imágenes lado a lado"""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {titles[0]}")
        st.image(original, use_container_width=True)
    with col2:
        st.markdown(f"### {titles[1]}")
        st.image(processed, use_container_width=True)

def download_button(image, filename="resultado.png"):
    """Crea botón de descarga para imagen"""
    is_success, buffer = cv2.imencode(".png", image)
    if is_success:
        btn = st.download_button(
            label="📥 Descargar Resultado",
            data=buffer.tobytes(),
            file_name=filename,
            mime="image/png"
        )
        return btn
    return None

def show_loading(message="Procesando..."):
    """Muestra spinner de carga"""
    return st.spinner(message)