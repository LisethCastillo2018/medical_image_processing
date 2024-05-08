import numpy as np
from collections import Counter


def white_stripe(image):
    """
    Desc: Normaliza una imagen utilizando el algoritmo White Stripe.

    El algoritmo White Stripe busca el último pico en la distribución de intensidades de la imagen y normaliza
    la imagen dividiendo todas las intensidades por el valor de este pico.
    Por lo tanto en la imagen normalizada el valor del pico queda al rededor de 1.

    Params:
        image (numpy.ndarray): La imagen de entrada a normalizar.

    Returns:
        numpy.ndarray: La imagen normalizada.
    """
    # Paso 1: Encontrar el último pico de la distribución
    ranges = np.linspace(5, 95, 5)
    percentiles_images = np.percentile(image.flatten(), ranges)
    intensities = image[(image < percentiles_images[-1]) & (image > percentiles_images[-2])]
    last_peak_value = (Counter(intensities)).most_common(1)[0][0]
    # Paso 2: Estandarizar la imagen dividiendo por el valor del último pico
    normalized_image = image / last_peak_value

    return normalized_image
