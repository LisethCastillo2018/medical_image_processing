import numpy as np

from algorithms.segmentation.region_growing import get_neighbors


def median_filter(image, initial_position):

    """
    Desc: Aplica el filtro de mediana a una imagen para la remoción de ruido.
    Params:
        image (numpy.ndarray): La imagen de entrada.
        initial_position (tuple): La posición inicial para comenzar el filtrado.
        
    Returns:
        numpy.ndarray: La imagen filtrada.
    """
    
    cluster = set([initial_position])
    visited = set([])

    dims = image.shape
    filtered_image = np.zeros_like(image)

    while cluster and len(cluster) < np.prod(dims):

        point = cluster.pop()
        if point in visited:
            continue

        visited.add(point)
        neighbors = get_neighbors(point, dims)
        neighborhood_median = np.median([image[p] for p in (neighbors + [point])])
        filtered_image[point] = neighborhood_median
        
        for neighbor in neighbors:
            if neighbor not in visited:
                cluster.add(neighbor)

    return filtered_image


