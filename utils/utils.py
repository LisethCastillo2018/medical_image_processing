


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


def draw_line(image, x0, y0, x1, y1):
    """
    Dibuja una línea en una imagen binaria utilizando el algoritmo de Bresenham.

    Parámetros:
    - image (numpy.ndarray): La imagen donde se dibujará la línea. Debe ser una matriz binaria (0 o 1).
    - x0 (int): La coordenada x del primer punto de la línea.
    - y0 (int): La coordenada y del primer punto de la línea.
    - x1 (int): La coordenada x del segundo punto de la línea.
    - y1 (int): La coordenada y del segundo punto de la línea.
    """
    # Calcula las diferencias en las coordenadas x e y
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    # Determina la dirección de cambio en x e y
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    # Calcula el error inicial
    err = dx - dy

    # Bucle principal del algoritmo de Bresenham
    while True:
        # Marca el píxel actual como parte de la línea
        image[y0, x0] = 1
        # Verifica si se alcanzó el punto final de la línea
        if x0 == x1 and y0 == y1:
            break
        # Calcula el doble del error actual
        e2 = err * 2
        # Ajusta el error según la dirección de cambio en y
        if e2 > -dy:
            err -= dy
            x0 += sx
        # Ajusta el error según la dirección de cambio en x
        if e2 < dx:
            err += dx
            y0 += sy