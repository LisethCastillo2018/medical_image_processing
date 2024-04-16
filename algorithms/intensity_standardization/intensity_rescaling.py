import numpy as np


def intensity_rescaling(image):
    """
    Desc: Aplica el reescalado de intensidades a una imagen.

    Params:
        image (numpy.ndarray): La imagen de entrada.

    Returns:
        numpy.ndarray: La imagen reescalada.
    """
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    
    scaled_image = (image - min_intensity) / (max_intensity - min_intensity)

    # imprimir histograma: el minimo debe ser 0, y el maximo debe ser 1
    
    return scaled_image