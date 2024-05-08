import streamlit as st
import nibabel as nib
import numpy as np
import tempfile
import os
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from matplotlib.path import Path
import matplotlib.pyplot as plt

from algorithms.isodata_thresholding import isodata_thresholding
from algorithms.k_means import k_means
from algorithms.region_growing import region_growing
from algorithms.thresholding import thresholding

from algorithms.intensity_standardization.histogram_matching import histogram_matching
from algorithms.intensity_standardization.intensity_rescaling import intensity_rescaling
from algorithms.intensity_standardization.white_stripe import white_stripe
from algorithms.intensity_standardization.z_score import z_score_transformation
from algorithms.denoising.mean_filter import mean_filter
from algorithms.denoising.median_filter import median_filter


@st.cache_data
def generate_thresholding(image_data, threshold):
    return thresholding(image_data, threshold)

@st.cache_data
def generate_isodata_thresholding(image_data, tolerance):
    return isodata_thresholding(image_data, tolerance)

@st.cache_data
def generate_region_growing(image_data, initial_position, tolerance, max_iterations):
    return region_growing(image_data, initial_position, tolerance, max_iterations)

@st.cache_data
def generate_k_means(image_data, k, max_iterations):
    return k_means(image_data, k, max_iterations)[1]

@st.cache_data
def generate_mean_filter(image_data, initial_position):
    return mean_filter(image_data, initial_position)

@st.cache_data
def generate_median_filter(image_data, initial_position):
    return median_filter(image_data, initial_position)


class ImageSegmentationApp:
    def __init__(self):
        self.nii_image = None
        self.image_data = None
        self.segmented_image = None
        self.algorithm = None
        self.canvas = None
        self.stroke_width = None
        self.stroke_color = None
        self.drawing_data = {}
        self.standardization_algorithm = None
        self.standardized_image = None
        self.reference_image = None
        self.denoising_algorithm = None
        self.denoising_image = None

    def load_nii_image(self, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file.close()
            tmp_file_path = tmp_file.name

        nii_image = nib.load(tmp_file_path)
        image_data = nii_image.get_fdata()
        os.unlink(tmp_file_path)
        return nii_image, image_data

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
                x_slice = st.number_input('Initial X', value=self.image_data.shape[0] // 2, min_value=0, max_value=self.image_data.shape[0] - 1)
            with col2:
                y_slice = st.number_input('Initial Y', value=self.image_data.shape[1] // 2, min_value=0, max_value=self.image_data.shape[1] - 1)
            with col3:
                z_slice = st.number_input('Initial Z', value=self.image_data.shape[2] // 2, min_value=0, max_value=self.image_data.shape[2] - 1)

            initial_position = (x_slice, y_slice, z_slice)
            col1, col2 = st.columns(2)
            with col1:
                tolerance = st.number_input('Tolerance', value=80)
            with col2:
                max_iterations = st.number_input('Max iterations', value=100)
            self.segmented_image = generate_region_growing(self.image_data, initial_position, tolerance, max_iterations)

        elif self.algorithm == "K-Means":
            col1, col2 = st.columns(2)
            with col1:
                k = st.number_input('Number of Clusters', value=3, min_value=1, max_value=10)
            with col2:
                max_iterations = st.number_input('Max Iterations', value=10, min_value=1, max_value=100)
            self.segmented_image = generate_k_means(self.image_data, k, max_iterations)

    def load_reference_image(self):
        uploaded_reference_image = st.sidebar.file_uploader("Select a reference image (.nii)", type=["nii"])
        if uploaded_reference_image:
            self.reference_image = self.load_nii_image(uploaded_reference_image)[1]

    def apply_intensity_standardization(self):
        self.standardized_image = None

        if self.standardization_algorithm == "Histogram Matching":
            if self.reference_image is not None:
                self.standardized_image = histogram_matching(self.reference_image, self.image_data)

        elif self.standardization_algorithm == "Intensity Rescaling":
            self.standardized_image = intensity_rescaling(self.image_data)

        elif self.standardization_algorithm == "White Stripe":
            self.standardized_image = white_stripe(self.image_data)
       
        elif self.standardization_algorithm == "Z-score":
            self.standardized_image = z_score_transformation(self.image_data)

    def apply_denoising(self):
        self.denoising_image = None

        if self.denoising_algorithm in ["Mean Filter", "Median Filter"]:

            col1, col2, col3 = st.columns(3)
            with col1:
                x_slice = st.number_input('Initial X', value=self.image_data.shape[0] // 2, min_value=0, max_value=self.image_data.shape[0] - 1)
            with col2:
                y_slice = st.number_input('Initial Y', value=self.image_data.shape[1] // 2, min_value=0, max_value=self.image_data.shape[1] - 1)
            with col3:
                z_slice = st.number_input('Initial Z', value=self.image_data.shape[2] // 2, min_value=0, max_value=self.image_data.shape[2] - 1)

            initial_position = (x_slice, y_slice, z_slice)

            if self.denoising_algorithm == "Mean Filter":
                self.denoising_image = generate_mean_filter(self.image_data, initial_position)
            else:
                self.denoising_image = generate_median_filter(self.image_data, initial_position)

    def canvas_component(self, normalized_image_data, key):
        # Convertir los datos seleccionados en una imagen PIL
        image_pil = Image.fromarray(normalized_image_data)

        # Create a canvas component
        canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",  # Sin relleno
            stroke_width=self.stroke_width,
            stroke_color=self.stroke_color,
            update_streamlit=True,
            background_image=image_pil,
            drawing_mode="freedraw",
            key=key,
            width=image_pil.width,
            height=image_pil.height,
        )

        self.drawing_data[key] = canvas.json_data
        # st.image(canvas.image_data)
        # st.write("Dibuja sobre la imagen:")
        # st.write(canvas.json_data)
        return canvas
    
    def generate_new_nii_image(self, key_drawing_data):
        original_nii = self.nii_image
        original_data = self.image_data
        drawing_data = self.drawing_data.get(key_drawing_data)

        if drawing_data is None or (drawing_data is not None and len(drawing_data['objects']) == 0):
            st.warning("No realizó trazos para guardar")
            return
        
        # Crear una máscara binaria basada en el trazo
        mask = np.zeros(original_data.shape[:2], dtype=np.uint8)

        for obj in drawing_data['objects']:
            if obj['type'] == 'path':
                path = obj['path']
                for i in range(len(path)):
                    # x, y = path[i][1], path[i][2]
                    # # Marcar el píxel en la máscara
                    # mask[int(y), int(x)] = 1

                    path = obj['path']
                    x_coords, y_coords = zip(*[(int(point[1]), int(point[2])) for point in path])
                    mask[y_coords, x_coords] = 1

        # expanded_mask = np.expand_dims(mask, axis=2)
        # masked_data = np.where(expanded_mask == 1, original_data, 0)

        masked_data = np.where(mask[..., np.newaxis], original_data, 0)
       
        new_nii = nib.Nifti1Image(masked_data, original_nii.affine, original_nii.header)

        folder_path = "store/"
        current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        new_nii_filename = f"img_{current_datetime}.nii"
        
        new_nii_path = os.path.join(folder_path, new_nii_filename)
        nib.save(new_nii, new_nii_path)
        st.success(f"Segmentación guardada como '{new_nii_filename}'")

    def run(self):
        st.title("3D Medical Image Viewer")
        st.sidebar.title("Upload Image")
        uploaded_file = st.sidebar.file_uploader("Select a .nii image", type=["nii"])
        key_c_x_slice = 'c_x_slice'
        key_c_y_slice = 'c_y_slice'
        key_c_z_slice = 'c_z_slice'

        if uploaded_file is not None:
            self.nii_image, self.image_data = self.load_nii_image(uploaded_file)
            shape = self.image_data.shape

            st.sidebar.divider()
            st.sidebar.subheader("Image viewer")
            x_slice = st.sidebar.slider("Slice en el eje X", 0, shape[0] - 1, shape[0] // 2)
            y_slice = st.sidebar.slider("Slice en el eje Y", 0, shape[1] - 1, shape[1] // 2)
            z_slice = st.sidebar.slider("Slice en el eje Z", 0, shape[2] - 1, shape[2] // 2)

            st.sidebar.divider()
            st.sidebar.subheader("Segmented image")
            self.algorithm = st.sidebar.selectbox("Select Algorithm", ["Selecciona una opción", "Thresholding", "Isodata Thresholding", "Region Growing", "K-Means"])

            # Specify canvas parameters in application
            st.sidebar.divider()
            st.sidebar.subheader("Drawing tool")
            self.stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 5)
            self.stroke_color = st.sidebar.color_picker("Stroke color hex: ", value="#ff4b4b")
             
            normalized_image_data = (self.image_data - np.min(self.image_data)) / (np.max(self.image_data) - np.min(self.image_data))
            normalized_data_canva = (normalized_image_data * 255).astype(np.uint8)

            image_view = None
            col1, col2, col3 = st.columns(3)

            with col1:
                image_view = normalized_data_canva[x_slice, :, :]
                # st.image(image_view, caption=f"Slices (X: {x_slice})")
                st.caption(f"Slices (X: {x_slice})")
                self.canvas_component(normalized_image_data=image_view, key=key_c_x_slice)

            with col2:
                image_view = normalized_data_canva[:, y_slice, :]
                # st.image(image_view, caption=f"Slices (Y: {y_slice})")
                st.caption(f"Slices (Y: {y_slice})")
                self.canvas_component(normalized_image_data=image_view, key=key_c_y_slice)

            with col3:
                image_view = normalized_data_canva[:, :, z_slice]
                # st.image(image_view, caption=f"Slices (Z: {z_slice})")
                st.caption(f"Slices (Z: {z_slice})")
                self.canvas_component(normalized_image_data=image_view, key=key_c_z_slice)

        
            st.divider()

            if self.algorithm != "Selecciona una opción":
                st.subheader('Segmented image')
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


            if st.sidebar.button("Guardar Trazos (Slide X)"):
                self.generate_new_nii_image(key_c_x_slice)

            if st.sidebar.button("Guardar Trazos (Slide Y)"):
                self.generate_new_nii_image(key_c_y_slice)

            if st.sidebar.button("Guardar Trazos (Slide Z)"):
                self.generate_new_nii_image(key_c_z_slice)

            # Intensity Standardization
            st.sidebar.divider()
            st.sidebar.subheader("Intensity Standardization")
            self.standardization_algorithm = st.sidebar.selectbox("Select Standardization Algorithm", ["Selecciona una opción", "Histogram Matching", "Intensity Rescaling", "White Stripe", "Z-score"])

            if self.standardization_algorithm != "Selecciona una opción":
                
                if self.standardization_algorithm == "Histogram Matching":
                    self.load_reference_image()

                if self.image_data is not None:
                    self.apply_intensity_standardization()

                if self.image_data is not None and self.standardized_image is not None:
                    st.subheader(self.standardization_algorithm)
                    st.subheader("Standardized Image and Histogram")

                    # Mostrar la nueva imagen estandarizada
                    norm_standardized_image = (self.standardized_image - np.min(self.standardized_image)) / (np.max(self.standardized_image) - np.min(self.standardized_image))
                    norm_standardized_image = (norm_standardized_image * 255).astype(np.uint8)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(norm_standardized_image[x_slice, :, :], caption=f"Slices (X: {x_slice})")

                    with col2:
                        st.image(norm_standardized_image[:, y_slice, :], caption=f"Slices (Y: {y_slice})")

                    with col3:
                        st.image(norm_standardized_image[:, :, z_slice], caption=f"Slices (Z: {z_slice})")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Mostrar el histograma de la imagen resultante
                        st.subheader("Standardized Image")
                        fig, ax = plt.subplots()
                        ax.hist(self.standardized_image[self.image_data > 10], 100)
                        ax.set_title('Histogram of Standardized Image')
                        ax.set_xlabel('Pixel Intensity')
                        ax.set_ylabel('Frequency')
                        st.pyplot(fig)

                    with col2:
                        # Mostrar el histograma de la imagen original
                        st.subheader("Original Image")
                        fig, ax = plt.subplots()
                        ax.hist(self.image_data[self.image_data > 10], 100)
                        ax.set_title('Histogram of Original Image')
                        ax.set_xlabel('Pixel Intensity')
                        ax.set_ylabel('Frequency')
                        st.pyplot(fig)
                    
                    if self.reference_image is not None:
                        # Mostrar el histograma de la imagen de referencia
                        st.divider()
                        st.subheader("Reference Image Histogram")
                        fig, ax = plt.subplots()
                        ax.hist(self.reference_image[self.image_data > 10], 100)
                        ax.set_title('Histogram of Reference Image')
                        ax.set_xlabel('Pixel Intensity')
                        ax.set_ylabel('Frequency')
                        st.pyplot(fig)
                    
                else:
                    st.warning("Please upload a reference image first.")

            # Denoising
            st.sidebar.divider()
            st.sidebar.subheader("Denoising")
            self.denoising_algorithm = st.sidebar.selectbox("Select Denoising Algorithm", ["Selecciona una opción", "Mean Filter", "Median Filter"])

            if self.denoising_algorithm != "Selecciona una opción" and self.image_data is not None:
                st.subheader(self.denoising_algorithm)
                self.apply_denoising()

                if self.denoising_image is not None:
                    st.subheader("Filtered Image")
                    # Mostrar la nueva imagen filtrada
                    norm_denoising_image = (self.denoising_image - np.min(self.denoising_image)) / (np.max(self.denoising_image) - np.min(self.denoising_image))
                    norm_denoising_image = (norm_denoising_image * 255).astype(np.uint8)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(norm_denoising_image[x_slice, :, :], caption=f"Slices (X: {x_slice})")

                    with col2:
                        st.image(norm_denoising_image[:, y_slice, :], caption=f"Slices (Y: {y_slice})")

                    with col3:
                        st.image(norm_denoising_image[:, :, z_slice], caption=f"Slices (Z: {z_slice})")

if __name__ == "__main__":
    app = ImageSegmentationApp()
    app.run()
