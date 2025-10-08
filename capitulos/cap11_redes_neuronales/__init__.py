import streamlit as st
import cv2
import numpy as np
import pyplot as plt

def run():

    st.markdown("""
    Este ejemplo demuestra cómo una **red neuronal multicapa (MLP)** puede aprender la operación lógica XOR:
    
    | Entrada A | Entrada B | Salida Esperada |
    |:----------:|:----------:|:----------------:|
    | 0 | 0 | 0 |
    | 0 | 1 | 1 |
    | 1 | 0 | 1 |
    | 1 | 1 | 0 |
    """)

    # === Datos de entrenamiento XOR ===
    train_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    labels = np.array([[0], [1], [1], [0]], dtype=np.float32)

    # === Configurar red neuronal MLP ===
    model = cv2.ml.ANN_MLP_create()
    model.setLayerSizes(np.array([2, 4, 1]))  # 2 entradas, 4 neuronas ocultas, 1 salida
    model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
    model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
    model.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10000, 1e-6))

    if st.button("⚡ Entrenar Red Neuronal (XOR)", use_container_width=True):
        with st.spinner("Entrenando red neuronal..."):
            # Entrenar
            model.train(train_data, cv2.ml.ROW_SAMPLE, labels)

            # Guardar modelo
            model.save("mlp_xor_model.xml")

            # Evaluar la red
            _, outputs = model.predict(train_data)

        # === Mostrar tabla de resultados ===
        st.success("✅ Entrenamiento completado")

        st.markdown("### 📊 Resultados de la Red Neuronal")

        results = []
        for i, (inp, expected, out) in enumerate(zip(train_data, labels, outputs)):
            binary_output = 1 if out[0] > 0.5 else 0
            results.append({
                "A": int(inp[0]),
                "B": int(inp[1]),
                "Esperado": int(expected[0]),
                "Salida Decimal": float(out[0]),
                "Aproximado": binary_output
            })

        # Mostrar tabla con resultados numéricos
        st.table(results)

        # === Descarga del modelo ===
        with open("mlp_xor_model.xml", "rb") as f:
            st.download_button(
                "💾 Descargar Modelo Entrenado (MLP XOR)",
                f,
                file_name="mlp_xor_model.xml",
                mime="application/xml"
            )

        # === Gráfica de activación ===
        st.markdown("### 📈 Visualización de Activación de la Red")

        x_values = np.linspace(0, 1, 50)
        y_values = np.linspace(0, 1, 50)
        z = np.zeros((50, 50))

        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):
                _, out = model.predict(np.array([[x, y]], dtype=np.float32))
                z[j, i] = out[0][0]

        fig, ax = plt.subplots()
        c = ax.imshow(z, extent=[0, 1, 0, 1], origin="lower", cmap="coolwarm", vmin=0, vmax=1)
        plt.colorbar(c)
        ax.set_xlabel("Entrada A")
        ax.set_ylabel("Entrada B")
        ax.set_title("Mapa de activación de la red XOR")
        st.pyplot(fig)

    else:
        st.info("Presiona el botón **Entrenar Red Neuronal (XOR)** para iniciar el proceso.")