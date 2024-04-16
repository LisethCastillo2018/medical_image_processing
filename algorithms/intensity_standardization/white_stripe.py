import numpy as np


def white_stripe(image):
    """
    Desc: Normaliza una imagen utilizando el algoritmo White Stripe.

    El algoritmo White Stripe busca el último pico en la distribución de intensidades de la imagen y normaliza
    la imagen dividiendo todas las intensidades por el valor de este pico.

    Params:
        image (numpy.ndarray): La imagen de entrada a normalizar.

    Returns:
        numpy.ndarray: La imagen normalizada.
    """
    # Paso 1: Encontrar el último pico de la distribución
    histogram, bins = np.histogram(image.flatten(), bins=256, range=(0, 255))
    last_peak_value = np.max(bins[:-1][histogram > 0])

    # Paso 2: Estandarizar la imagen dividiendo por el valor del último pico
    normalized_image = image / last_peak_value

    # Asegurar que los valores estén dentro del rango [0, 1]
    normalized_image = np.clip(normalized_image, 0, 1)

    return normalized_image