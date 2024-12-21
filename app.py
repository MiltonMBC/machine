import streamlit as st
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

st.title("Generador de Retratos Artísticos")
st.sidebar.write("Opciones de Generación")

num_images = st.sidebar.slider("Número de imágenes", 1, 16, 4)
latent_dim = 32

if st.button("Generar Imágenes"):
    ort_session = ort.InferenceSession("generator.onnx")
    random_latent_vectors = np.random.normal(size=(num_images, latent_dim)).astype(np.float32)
    onnx_output = ort_session.run(None, {"args_0": random_latent_vectors})

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow((onnx_output[0][i] + 1) / 2)
        ax.axis("off")
    st.pyplot(fig)
