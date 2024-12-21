import streamlit as st
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

st.title("Generador de Retratos Artísticos")
st.sidebar.write("Opciones de Generación")

# Configuración de parámetros
num_images = st.sidebar.slider("Número de imágenes", 1, 16, 4)
latent_dim = 128

if st.button("Generar Imágenes"):
    # Cargar el modelo ONNX
    ort_session = ort.InferenceSession("generator.onnx")

    # Generar vectores latentes aleatorios
    random_latent_vectors = np.random.normal(size=(num_images, latent_dim)).astype(np.float32)

    # Realizar inferencia (asegúrate de usar el nombre correcto de entrada)
    input_name = ort_session.get_inputs()[0].name  # Detectar el nombre de entrada automáticamente
    onnx_output = ort_session.run(None, {input_name: random_latent_vectors})

    # Mostrar las imágenes generadas
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow((onnx_output[0][i] + 1) / 2)  # Transformación para visualizar
        ax.axis("off")
    st.pyplot(fig)
