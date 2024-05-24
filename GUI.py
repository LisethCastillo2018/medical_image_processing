import streamlit as st
import nibabel as nib
import numpy as np
import tempfile
import os
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import matplotlib.pyplot as plt

from algorithms.segmentation.isodata_thresholding import isodata_thresholding
from algorithms.segmentation.k_means import k_means
from algorithms.segmentation.region_growing import region_growing
from algorithms.segmentation.thresholding import thresholding
from algorithms.segmentation.laplacian_coordinates import laplacian_coordinates

from algorithms.intensity_standardization.histogram_matching import histogram_matching
from algorithms.intensity_standardization.intensity_rescaling import intensity_rescaling
from algorithms.intensity_standardization.white_stripe import white_stripe
from algorithms.intensity_standardization.z_score import z_score_transformation
from algorithms.denoising.mean_filter import mean_filter
from algorithms.denoising.median_filter import median_filter
from algorithms.borders.borders import border_x, border_y, magnitud
from algorithms.registration.registration import registration
from utils.constants import Colors
from utils.utils import draw_line, normalize_image, resize_image


# @st.cache_data
def generate_laplacian_coordinates(image_data, drawing_data, original_shape):
    return laplacian_coordinates(image_data, drawing_data, original_shape)

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

@st.cache_data
def generate_border_x(img, x_slice, y_slice, z_slice):
    return border_x(img, x_slice, y_slice, z_slice)

@st.cache_data
def generate_border_y(img, x_slice, y_slice, z_slice):
    return border_y(img, x_slice, y_slice, z_slice)

@st.cache_data
def generate_magnitud(img_filt_x, img_filt_y):
    return magnitud(img_filt_x, img_filt_y)

@st.cache_data
def generate_registration(fixed_image_path, moving_image_path):
    return registration(fixed_image_path, moving_image_path)


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
        self.registered_image = None
        self.moving_image = None
        self.original_shape = (600, 600)

    def load_nii_image(self, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file.close()
            tmp_file_path = tmp_file.name

        nii_image = nib.load(tmp_file_path)
        image_data = nii_image.get_fdata()
        return nii_image, image_data, tmp_file_path

    def apply_segmentation_algorithm(self):
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
        image_pil = Image.fromarray(normalized_image_data).resize(self.original_shape)

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
        return canvas

    def generate_drawing_image(self, key_drawing_data):
        
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
                for i in range(len(path) - 1):
                    x0, y0 = int(path[i][1]), int(path[i][2])
                    x1, y1 = int(path[i + 1][1]), int(path[i + 1][2])
                    draw_line(mask, x0, y0, x1, y1)

        # Crear una imagen RGB en blanco y negro (0-255) para guardar el trazo
        image_rgb = np.zeros((*original_data.shape[:2], 3), dtype=np.uint8)
        image_rgb[mask == 1] = [255, 255, 255]

        # Convertir los datos a una imagen PIL
        image_pil = Image.fromarray(image_rgb)

        # Guardar la imagen
        folder_path = "store/"
        current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        new_image_filename = f"drawing_{current_datetime}.png"

        new_image_path = os.path.join(folder_path, new_image_filename)
        image_pil.save(new_image_path)
        st.success(f"Trazo guardado como '{new_image_filename}'")

    def show_bordered_image(self, img_filt, x_slice, y_slice, z_slice, umbral=None):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(img_filt[0] > umbral if umbral else img_filt[0], cmap='gray')
        axs[0].set_title(f"Filtered Slice (X: {x_slice})")
        axs[0].axis('off')

        axs[1].imshow(img_filt[1] > umbral if umbral else img_filt[1], cmap='gray')
        axs[1].set_title(f"Filtered Slice (Y: {y_slice})")
        axs[1].axis('off')

        axs[2].imshow(img_filt[2] > umbral if umbral else img_filt[2], cmap='gray')
        axs[2].set_title(f"Filtered Slice (Z: {z_slice})")
        axs[2].axis('off')
        st.pyplot(fig)

    def handle_registration(self, uploaded_registration_image):
        if uploaded_registration_image:
            _, self.moving_image, moving_image_path = self.load_nii_image(uploaded_registration_image)
            if not st.session_state.get('exec_registered_image', False):
                generate_registration(self.image_path, moving_image_path)
                st.session_state['exec_registered_image'] = True
        else:
            st.session_state['exec_registered_image'] = False

    def toggle_image_display(self, image_view, slice_type, slice_value, canvas_key, show=True):
        show_image = st.checkbox(f"Show {slice_type} Slice (Value: {slice_value})", show)
        if show_image:
            st.caption(f"Slices ({slice_type}: {slice_value})")
            self.canvas_component(normalized_image_data=image_view, key=canvas_key)

    def run(self):   

        if 'exec_registered_image' not in st.session_state:
            st.session_state['exec_registered_image'] = False

        st.title("3D Medical Image Viewer")
        st.sidebar.title("Upload Image")
        uploaded_file = st.sidebar.file_uploader("Select a .nii image", type=["nii"])
        key_c_x_slice = 'c_x_slice'
        key_c_y_slice = 'c_y_slice'
        key_c_z_slice = 'c_z_slice'

        if uploaded_file is not None:
            self.nii_image, self.image_data, self.image_path = self.load_nii_image(uploaded_file)
            shape = self.image_data.shape

            st.sidebar.divider()
            st.sidebar.subheader("Image viewer")
            x_slice = st.sidebar.slider("Slice en el eje X", 0, shape[0] - 1, shape[0] // 2)
            y_slice = st.sidebar.slider("Slice en el eje Y", 0, shape[1] - 1, shape[1] // 2)
            z_slice = st.sidebar.slider("Slice en el eje Z", 0, shape[2] - 1, shape[2] // 2)

            on_bordered = st.sidebar.toggle("Show Image Borders")

            st.sidebar.divider()
            st.sidebar.subheader("Segmented image")
            self.algorithm = st.sidebar.selectbox("Select Algorithm", ["Selecciona una opción", "Thresholding", "Isodata Thresholding", "Region Growing", "K-Means"])

            # Specify canvas parameters in application
            st.sidebar.divider()
            st.sidebar.subheader("Drawing tool")
            self.stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 5)

            # Seleccionar el color de trazo
            colors = {"Red": Colors.RED_COLOR, "Green": Colors.GREEN_COLOR}
            selected_color = st.sidebar.radio("Stroke color:", list(colors.keys()))
            self.stroke_color = colors[selected_color]
             
            normalized_image_data = (self.image_data - np.min(self.image_data)) / (np.max(self.image_data) - np.min(self.image_data))
            normalized_data_canva = (normalized_image_data * 255).astype(np.uint8)

            self.toggle_image_display(normalized_data_canva[x_slice, :, :], 'X', x_slice, key_c_x_slice)
            self.toggle_image_display(normalized_data_canva[:, y_slice, :], 'Y', y_slice, key_c_y_slice, show=False)
            self.toggle_image_display(normalized_data_canva[:, :, z_slice], 'Z', z_slice, key_c_z_slice, show=False)
        
            st.divider()

            # Bordes
            if on_bordered:
                st.write("Visualización de bordes")
                img_filt_x = generate_border_x(self.image_data, x_slice, y_slice, z_slice)
                img_filt_y = generate_border_y(self.image_data, x_slice, y_slice, z_slice)
                self.show_bordered_image(img_filt_x, x_slice, y_slice, z_slice)

                st.write("Visualización de maginitud")
                img_filt = generate_magnitud(img_filt_x, img_filt_y)
                self.show_bordered_image(img_filt, x_slice, y_slice, z_slice)

                st.write("Visualización de bordes sobre un umbral")
                v_umbral = st.slider("Umbral", 0, 100, 40)
                self.show_bordered_image(img_filt, x_slice, y_slice, z_slice, v_umbral)

            # Segmentación
            if self.algorithm != "Selecciona una opción":
                st.subheader('Segmented image')
                self.apply_segmentation_algorithm()

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

            if st.sidebar.button("Laplacian Coordinates X"):
                st.subheader("Laplacian Coordinates Image Segmentation X")
                image_resize = resize_image(self.image_data[x_slice, :, :])
                self.segmented_image = generate_laplacian_coordinates(image_resize, self.drawing_data.get(key_c_x_slice), self.original_shape)
                norm_standardized_image = normalize_image(self.segmented_image)
                st.image(norm_standardized_image, caption="Laplacian Coordinates X", width=600)

            if st.sidebar.button("Laplacian Coordinates Y"):
                st.subheader("Laplacian Coordinates Image Segmentation Y")
                image_resize = resize_image(self.image_data[:, y_slice, :])
                self.segmented_image = generate_laplacian_coordinates(image_resize, self.drawing_data.get(key_c_y_slice), self.original_shape)
                norm_standardized_image = normalize_image(self.segmented_image)
                st.image(norm_standardized_image, caption="Laplacian Coordinates Y", width=600)

            if st.sidebar.button("Laplacian Coordinates Z"):
                st.subheader("Laplacian Coordinates Image Segmentation Z")
                image_resize = resize_image(self.image_data[:, :, z_slice])
                self.segmented_image = generate_laplacian_coordinates(image_resize, self.drawing_data.get(key_c_z_slice), self.original_shape)
                norm_standardized_image = normalize_image(self.segmented_image)
                st.image(norm_standardized_image, caption="Laplacian Coordinates Z", width=600)

            st.sidebar.divider()

            if st.sidebar.button("Guardar Trazos (Slide X)"):
                self.generate_drawing_image(key_c_x_slice)

            if st.sidebar.button("Guardar Trazos (Slide Y)"):
                self.generate_drawing_image(key_c_y_slice)

            if st.sidebar.button("Guardar Trazos (Slide Z)"):
                self.generate_drawing_image(key_c_z_slice)

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
                    st.subheader("Standardized Image")

                    # Mostrar la nueva imagen estandarizada
                    norm_standardized_image = normalize_image(self.standardized_image)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(norm_standardized_image[x_slice, :, :], caption=f"Slices (X: {x_slice})")

                    with col2:
                        st.image(norm_standardized_image[:, y_slice, :], caption=f"Slices (Y: {y_slice})")

                    with col3:
                        st.image(norm_standardized_image[:, :, z_slice], caption=f"Slices (Z: {z_slice})")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Mostrar el histograma de la imagen original
                        st.subheader("Original Image")
                        fig, ax = plt.subplots()
                        ax.hist(self.image_data[self.image_data > 10], 100)
                        ax.set_title('Histogram of Original Image')
                        ax.set_xlabel('Pixel Intensity')
                        ax.set_ylabel('Frequency')
                        st.pyplot(fig)

                    with col2:
                        # Mostrar el histograma de la imagen resultante
                        st.subheader("Standardized Image")
                        fig, ax = plt.subplots()
                        ax.hist(self.standardized_image[self.image_data > 10], 100)
                        ax.set_title('Histogram of Standardized Image')
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

            # Registrar una nueva imagen
            st.sidebar.divider()
            if not st.sidebar.checkbox("Register images"):
                st.session_state['exec_registered_image'] = False
            else:
                uploaded_registration_image = st.sidebar.file_uploader("Upload moving image (.nii)", type=["nii"])
                self.handle_registration(uploaded_registration_image)

            if st.session_state['exec_registered_image'] and self.moving_image is not None:
                registered_image = nib.load('./store/imagen_registrada.nii')
                self.registered_image = registered_image.get_fdata()

                st.subheader("Image registration")

                st.text("Moving image")
                norm_moving_image = (self.moving_image - np.min(self.moving_image)) / (np.max(self.moving_image) - np.min(self.moving_image))
                norm_moving_image = (norm_moving_image * 255).astype(np.uint8)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(norm_moving_image[x_slice, :, :], caption=f"Slices (X: {x_slice})")

                with col2:
                    st.image(norm_moving_image[:, y_slice, :], caption=f"Slices (Y: {y_slice})")

                with col3:
                    st.image(norm_moving_image[:, :, z_slice], caption=f"Slices (Z: {z_slice})")

                st.text("Registered image")
                norm_registered_image = (self.registered_image - np.min(self.registered_image)) / (np.max(self.registered_image) - np.min(self.registered_image))
                norm_registered_image = (norm_registered_image * 255).astype(np.uint8)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(norm_registered_image[x_slice, :, :], caption=f"Slices (X: {x_slice})")

                with col2:
                    st.image(norm_registered_image[:, y_slice, :], caption=f"Slices (Y: {y_slice})")

                with col3:
                    st.image(norm_registered_image[:, :, z_slice], caption=f"Slices (Z: {z_slice})")


if __name__ == "__main__":
    app = ImageSegmentationApp()
    app.run()
