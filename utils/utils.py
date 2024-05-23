import numpy as np
from skimage.transform import resize
from utils.constants import Colors


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


def identify_and_store_drawing(canvas_data):
    red_points = []
    green_points = []

    # Iterar sobre los objetos dibujados en el lienzo
    for obj in canvas_data['objects']:
        if obj['type'] == 'path':
            path = obj['path']
            color = obj['stroke']
            # Identificar el color del trazo
            if color == Colors.RED_COLOR:
                # Almacenar los puntos del trazo rojo
                for point in path:
                    red_points.append((int(point[1]), int(point[2])))
            elif color == Colors.GREEN_COLOR:
                # Almacenar los puntos del trazo verde
                for point in path:
                    green_points.append((int(point[1]), int(point[2])))
    return red_points, green_points


def normalize_image(image):
    norm_standardized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return (norm_standardized_image * 255).astype(np.uint8)


def resize_image(image_data):
    slice_index = image_data.shape[2] // 2 
    image_2d = image_data[:, :, slice_index] 

    factor_reduccion = 0.1
    nueva_altura = int(image_2d.shape[0] * factor_reduccion) 
    nueva_anchura = int(image_2d.shape[1] * factor_reduccion) 
    return resize(image_2d, (nueva_altura, nueva_anchura)) 