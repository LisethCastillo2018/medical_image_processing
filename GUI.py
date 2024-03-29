import streamlit as st
import nibabel as nib
import numpy as np
import tempfile
import os
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from matplotlib.path import Path

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
def generate_region_growing(image_data, initial_position, tolerance, max_iterations):
    return region_growing(image_data, initial_position, tolerance, max_iterations)

@st.cache_data
def generate_k_means(image_data, k, max_iterations):
    return k_means(image_data, k, max_iterations)[1]


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

    def load_nii_image(self, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file.close()
            tmp_file_path = tmp_file.name

        self.nii_image = nib.load(tmp_file_path)
        self.image_data = self.nii_image.get_fdata()
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
            self.load_nii_image(uploaded_file)
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

if __name__ == "__main__":
    app = ImageSegmentationApp()
    app.run()
