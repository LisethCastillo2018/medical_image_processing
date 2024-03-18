import streamlit as st
import nibabel as nib
import numpy as np
import tempfile
import os

from algoritmos.isodata_thresholding import isodata_thresholding
from algoritmos.k_means import k_means
from algoritmos.region_growing import region_growing
from algoritmos.thresholding import thresholding


def load_nii_image(uploaded_file):
    # Guardar el archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file.close()
        tmp_file_path = tmp_file.name

    # Cargar la imagen
    nii_image = nib.load(tmp_file_path)
    nii_data = nii_image.get_fdata()

    # Eliminar el archivo temporal
    os.unlink(tmp_file_path)
    return nii_data


def apply_algorithm(image_data, algorithm, **kwargs):
    if algorithm == "Thresholding":
        threshold = st.slider("Threshold", 0, 500, 150)
        segmented_image = thresholding(image_data, threshold)
    elif algorithm == "Isodata Thresholding":
        tolerance = st.slider("Tolerance", 1, 500, 10)
        segmented_image = isodata_thresholding(image_data, tolerance)
    elif algorithm == "Region Growing":
        initial_position = (st.slider("Initial X", 0, image_data.shape[0] - 1, image_data.shape[0] // 2),
                            st.slider("Initial Y", 0, image_data.shape[1] - 1, image_data.shape[1] // 2),
                            st.slider("Initial Z", 0, image_data.shape[2] - 1, image_data.shape[2] // 2))
        tolerance = st.slider("Tolerance", 10, 500, 80)
        segmented_image = region_growing(image_data, initial_position, tolerance, 150)
    elif algorithm == "K-Means":
        k = st.slider("Number of Clusters", 1, 10, 3)
        max_iterations = st.slider("Max Iterations", 1, 100, 10)
        _, segmented_image = k_means(image_data, k, max_iterations)
    print("*** algorithm: ", algorithm)
    return segmented_image


def main():
    st.title("Visualizador de Imágenes Médicas 3D")
    st.sidebar.title("Cargar Imagen")
    uploaded_file = st.sidebar.file_uploader("Selecciona una imagen .nii", type=["nii"])

    if uploaded_file is not None:
        image_data = load_nii_image(uploaded_file)
        shape = image_data.shape

        x_slice = st.sidebar.slider("Slice en el eje X", 0, shape[0] - 1, shape[0] // 2)
        y_slice = st.sidebar.slider("Slice en el eje Y", 0, shape[1] - 1, shape[1] // 2)
        z_slice = st.sidebar.slider("Slice en el eje Z", 0, shape[2] - 1, shape[2] // 2)

        normalized_image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(normalized_image_data[x_slice, :, :], caption=f"Slices (X: {x_slice})")

        with col2:
            st.image(normalized_image_data[:, y_slice, :], caption=f"Slices (Y: {y_slice})")

        with col3:
            st.image(normalized_image_data[:, :, z_slice], caption=f"Slices (Z: {z_slice})")


        # Selector de algoritmo
        algorithm = st.sidebar.selectbox("Select Algorithm", ["Thresholding", "Isodata Thresholding", "Region Growing", "K-Means"])

        # Aplicar el algoritmo seleccionado
        segmented_image = apply_algorithm(image_data, algorithm)
        print(segmented_image)

        # Convertir los valores booleanos en valores enteros (0 para False, 255 para True)
        segmented_image = segmented_image.astype(np.uint8) * 255

        # Mostrar la imagen segmentada
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(segmented_image[x_slice, :, :], caption=f"Slices (X: {x_slice})")

        with col2:
            st.image(segmented_image[:, y_slice, :], caption=f"Slices (Y: {y_slice})")

        with col3:
            st.image(segmented_image[:, :, z_slice], caption=f"Slices (Z: {z_slice})")


if __name__ == "__main__":
    main()
