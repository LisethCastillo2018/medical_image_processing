import numpy as np


def z_score_transformation(image):
    """
    Desc: Aplica la transformación z-score a una imagen.

    Params:
        image (numpy.ndarray): La imagen de entrada.

    Returns:
        numpy.ndarray: La imagen transformada.
    """

    # se debe exluir el fondo -> image > 10
    mean_intensity = np.mean(image[image > 10])
    std_intensity = np.std(image[image > 10])
    transformed_image = (image - mean_intensity) / std_intensity

    # imprimir histograma, se debe ver centrado en cero
    
    return transformed_image
