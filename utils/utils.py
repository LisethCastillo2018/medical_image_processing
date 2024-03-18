


def get_image_center(image_shape):
    """
    Desc: Obtiene las coordenadas del punto central de la imagen
    Params:
        image_shape (tuple): Dimesión de la imagen
    Returns:
        center (tuple): Coordenada del punto central
    """
    center = tuple(dim // 2 for dim in image_shape)
    return center