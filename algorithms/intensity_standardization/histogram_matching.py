import numpy as np


def histogram_matching(reference_image, input_image, k_landmarks=256):
    """
    Desc: Aplica el algoritmo de Histogram Matching para transformar una imagen de entrada
    de acuerdo al histograma de una imagen de referencia.

    Params:
        reference_image (numpy.ndarray): La imagen de referencia.
        input_image (numpy.ndarray): La imagen de entrada que se transformará.
        num_landmarks (int): El número de landmarks (percentiles) a utilizar.

    Returns:
        numpy.ndarray: La imagen transformada.
    """
    # Paso de entrenamiento
    ref_percentiles = np.linspace(5, 95, k_landmarks)
    ref_intensity_in_percentiles = np.percentile(reference_image, ref_percentiles)
    
    # Paso de transformación
    input_intensity_in_percentiles = np.percentile(input_image, ref_percentiles)

    result_transformation_function = transformation_function(
        input_image,
        input_intensity_in_percentiles, 
        ref_intensity_in_percentiles
    )
    transformed_image = apply_transformation(
        input_image, 
        result_transformation_function
    )
    
    return transformed_image


def transformation_function(input_image, input_intensity_in_percentiles, ref_intensity_in_percentiles):
    # Interpolar los valores de intensidad de la imagen de entrada a los valores de intensidad de la referencia
    # Devuelve una estimación de los valores de ref_intensity_in_percentiles correspondientes a los valores de input_intensity_in_percentiles.
    return np.interp(
        input_image, 
        input_intensity_in_percentiles, 
        ref_intensity_in_percentiles
    )


def apply_transformation(input_image, transformation_function):
    # Aplicar la transformación a la imagen de entrada
    # Convertir las intensidades resultantes al mismo tipo de datos que la imagen original
    transformed_image = transformation_function.astype(input_image.dtype)
    return transformed_image
