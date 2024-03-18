

import numpy as np


def k_means(image, k, max_iterations):
    """
    Desc: Implementa el algoritmo de k-medias para segmentación de imágenes.
    Params:
        image (matriz): La imagen de entrada.
        k (int): El número de clusters.
        max_iterations (int): El número máximo de iteraciones permitidas.
    Returns:
        centroids (matriz): Los centroides finales de los clusters.
        segmented_image (matriz): La imagen segmentada.
    """
    # 1. Select initial cluster centres
    centroids = initialize_centroids(image, k)
    cont = 0
    
    while cont < max_iterations:
        # 2. Assign voxels to each group based on distance
        clusters = assign_to_clusters(image, centroids)
        
        # 3. Update cluster centre (mean)
        new_centroids = update_centroids(image, clusters, k)
        
        # Si no hay cambios se detiene el algoritmo
        if np.array_equal(new_centroids, centroids):
            break
        else:
            centroids = new_centroids

        cont += 1

    segmented_image = clusters.reshape(image.shape)
    return centroids, segmented_image


def initialize_centroids(image, k):
    """
    Desc: Inicializa aleatoriamente k centroides dentro del rango de valores de la imagen.
    Params:
        image (matriz): La imagen de entrada.
        k (int): El número de centroides a inicializar.
    Returns:
        centroids (matriz): Los centroides iniciales.
    """
    # Aplanar la imagen para obtener un arreglo unidimensional
    flat_image = image.flatten()
    # Obtener la cantidad total de píxeles en la imagen
    num_pixels = len(flat_image)
    # Seleccionar k índices aleatorios sin reemplazo
    centroids_indices = np.random.choice(num_pixels, k, replace=False)
    # Obtener los valores de los píxeles en los índices seleccionados como centroides iniciales
    centroids = flat_image[centroids_indices]
    return centroids


def assign_to_clusters(image, centroids):
    """
    Desc: Asigna cada píxel de la imagen al cluster cuyo centroide esté más cercano.
    Params:
        image (matriz): La imagen de entrada.
        centroids (matriz): Los centroides de los clusters.
    Returns:
        clusters (matriz): La asignación de cada píxel a un cluster.
    """
    # Inicializar una matriz de clusters del mismo tamaño que la imagen
    num_z, num_rows, num_cols = image.shape
    clusters = np.zeros((num_z, num_rows, num_cols))
    # Para cada píxel en la imagen, encontrar el centroide más cercano y asignarlo a su cluster
    for z in range(num_z):
        for i in range(num_rows):
            for j in range(num_cols):
                # Obtener la intensidad del píxel actual
                pixel_value = image[z, i, j]
                # Calcular la distancia a cada centroide
                distances = np.abs(centroids - pixel_value)
                # Encontrar el índice del centroide más cercano
                closest_centroid_index = np.argmin(distances)
                # Asignar el píxel al cluster correspondiente
                clusters[z, i, j] = closest_centroid_index
    return clusters


def update_centroids(image, clusters, k):
    """
    Desc: Recalcula los centroides de los clusters como el promedio de los píxeles asignados a cada cluster.
    Params:
        image (matriz): La imagen de entrada.
        clusters (matriz): La asignación de cada píxel a un cluster.
        k (int): El número de clusters.
    Returns:
        new_centroids (matriz): Los nuevos centroides de los clusters.
    """
    # Inicializar un arreglo para almacenar los nuevos centroides
    new_centroids = np.zeros(k)
    # Para cada cluster, calcular el promedio de los píxeles asignados a él
    for cluster_index in range(k):
        # Obtener los píxeles asignados al cluster actual
        cluster_pixels = image[clusters == cluster_index]
        # Calcular el promedio de los valores de los píxeles
        if len(cluster_pixels) > 0:
            new_centroids[cluster_index] = np.mean(cluster_pixels)
    return new_centroids