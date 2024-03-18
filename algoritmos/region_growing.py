import numpy as np


def region_growing(image, initial_position, tolerance, max_iterations=150):
    """ 
    Desc: Realiza la segmentación con base en el crecimiento de regiones
    Params:
        iamge (matriz): Imagen de entrada
        centroid_position (tuple): Punto inicial
        tolerance (float): Toleracia para el crecimiento de la región
    Returns:
        segmented_image (matrys): Imagen segmentada binaria
    """
    cluster = set([initial_position])
    cont = 0

    while cont < max_iterations:
        mean_cluster = np.mean([image[point] for point in cluster])
        prev_cluster = cluster.copy() 

        for point in prev_cluster:
            neighbors = get_neighbors(point, image.shape)
            for neighbor in neighbors:
                if neighbor not in cluster and np.linalg.norm(image[neighbor] - mean_cluster) < tolerance:
                    cluster.add(neighbor)

        if cluster == prev_cluster:
            break

        cont += 1

    segmented_image = np.zeros_like(image)
    for point in cluster:
        segmented_image[point] = 1

    return segmented_image


def get_neighbors(point, shape):
    """
    Desc: Obtiene los vecinos de arriba, abajo, izquierda, derecha, delate y atrás de un punto dado
    Params:
        point (tuple): Coordenada del punto
        shape (tuple): Dimensión de la imagen o matriz
    Returns: 
        neighbors (list): Lista con las coordenadas de los vecinos
    """
    x, y, z = point
    neighbors = []
    
    neighbor_coords = [(x - 1, y, z),  # izquierda
                       (x + 1, y, z),  # derecha
                       (x, y - 1, z),  # abajo
                       (x, y + 1, z),  # arriba
                       (x, y, z - 1),  # atrás
                       (x, y, z + 1)]  # adelante
    
    for coord in neighbor_coords:
        if all(0 <= c < shape[i] for i, c in enumerate(coord)):
            neighbors.append(coord)
    return neighbors