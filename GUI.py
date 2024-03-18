import streamlit as st
import nibabel as nib
import numpy as np
import tempfile
import os

from algorithms.isodata_thresholding import isodata_thresholding
from algorithms.k_means import k_means
from algorithms.region_growing import region_growing
from algorithms.thresholding import thresholding


@st.cache_data
def generate_thresholding(image_data, threshold):
    return thresholding(image_data, threshold)

@st.cache_data
def generate_isodata_thresholding(image_data, tolerance):
    return isodata_thresholding(image_data, tolerance)

@st.cache_data
def generate_region_growing(image_data, initial_position, tolerance):
    return region_growing(image_data, initial_position, tolerance, 150)

@st.cache_data
def generate_k_means(image_data, k, max_iterations):
    return k_means(image_data, k, max_iterations)[1]


class ImageSegmentationApp:
    def __init__(self):
        self.image_data = None
        self.segmented_image = None
        self.algorithm = None

    def load_nii_image(self, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file.close()
            tmp_file_path = tmp_file.name

        nii_image = nib.load(tmp_file_path)
        self.image_data = nii_image.get_fdata()
        os.unlink(tmp_file_path)

    def apply_algorithm(self):
        self.segmented_image = None

        if self.algorithm == "Thresholding":
            threshold = st.slider("Threshold", 0, 500, 150)
            self.segmented_image = generate_thresholding(self.image_data, threshold)
           
        elif self.algorithm == "Isodata Thresholding":
            tolerance = st.slider("Tolerance", 1, 500, 10)
            self.segmented_image = generate_isodata_thresholding(self.image_data, tolerance)
           
        elif self.algorithm == "Region Growing":
            col1, col2, col3 = st.columns(3)
            with col1:
                x_slice = st.slider("Initial X", 0, self.image_data.shape[0] - 1, self.image_data.shape[0] // 2)
            with col2:
                y_slice = st.slider("Initial Y", 0, self.image_data.shape[1] - 1, self.image_data.shape[1] // 2)
            with col3:
                z_slice = st.slider("Initial Z", 0, self.image_data.shape[2] - 1, self.image_data.shape[2] // 2)

            initial_position = (x_slice, y_slice, z_slice)
            tolerance = st.slider("Tolerance", 10, 500, 80)
            self.segmented_image = generate_region_growing(self.image_data, initial_position, tolerance)
     
        elif self.algorithm == "K-Means":
            col1, col2 = st.columns(2)
            with col1:
                k = st.slider("Number of Clusters", 1, 10, 3)
            with col2:
                max_iterations = st.slider("Max Iterations", 1, 100, 10)
            self.segmented_image = generate_k_means(self.image_data, k, max_iterations)
  
    def run(self):
        st.title("Visualizador de Imágenes Médicas 3D")
        st.sidebar.title("Cargar Imagen")
        uploaded_file = st.sidebar.file_uploader("Selecciona una imagen .nii", type=["nii"])

        if uploaded_file is not None:
            self.load_nii_image(uploaded_file)
            shape = self.image_data.shape

            x_slice = st.sidebar.slider("Slice en el eje X", 0, shape[0] - 1, shape[0] // 2)
            y_slice = st.sidebar.slider("Slice en el eje Y", 0, shape[1] - 1, shape[1] // 2)
            z_slice = st.sidebar.slider("Slice en el eje Z", 0, shape[2] - 1, shape[2] // 2)

            normalized_image_data = (self.image_data - np.min(self.image_data)) / (np.max(self.image_data) - np.min(self.image_data))
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(normalized_image_data[x_slice, :, :], caption=f"Slices (X: {x_slice})")

            with col2:
                st.image(normalized_image_data[:, y_slice, :], caption=f"Slices (Y: {y_slice})")

            with col3:
                st.image(normalized_image_data[:, :, z_slice], caption=f"Slices (Z: {z_slice})")

            self.algorithm = st.sidebar.selectbox("Select Algorithm", ["Selecciona una opción", "Thresholding", "Isodata Thresholding", "Region Growing", "K-Means"])
            
            st.divider()

            if self.algorithm != "Selecciona una opción":
                st.text('Segmented image')
                self.apply_algorithm()

            if self.segmented_image is not None and np.any(self.segmented_image):

                self.segmented_image = self.segmented_image.astype(np.uint8) * 255
                # Mostrar la imagen segmentada
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(self.segmented_image[x_slice, :, :], caption=f"Slices (X: {x_slice})")

                with col2:
                    st.image(self.segmented_image[:, y_slice, :], caption=f"Slices (Y: {y_slice})")

                with col3:
                    st.image(self.segmented_image[:, :, z_slice], caption=f"Slices (Z: {z_slice})")


if __name__ == "__main__":
    app = ImageSegmentationApp()
    app.run()
