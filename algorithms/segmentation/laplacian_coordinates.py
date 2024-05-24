import numpy as np
from functools import reduce


from utils.utils import identify_and_store_drawing, resize_image

def get_neighbors(matrix, i, j):
    neighbors = [(-1, -1), (-1, 0), (-1, 1), 
                 (0, -1),           (0, 1), 
                 (1, -1),   (1, 0), (1, 1)]
    neighbor_positions = []
    for n in neighbors:
        ni, nj = i + n[0], j + n[1]
        if 0 <= ni < matrix.shape[0] and 0 <= nj < matrix.shape[1]:
            neighbor_positions.append((i * matrix.shape[1] + j, ni * matrix.shape[1] + nj))
    return neighbor_positions

def generate_neighbor_matrix(matrix):
    neighbor_matrix = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            neighbor_matrix.append(get_neighbors(matrix, i, j))
    return neighbor_matrix

def calculate_sigma(image, E):
    max_difference = 0
    
    # Recorremos cada píxel de la imagen
    for i in range(len(E)):
        for j in range(len(E[i])):
            # Obtenemos las posiciones de los vecinos del píxel actual
            neighbor_positions = E[i][j]
            current_pixel_value = image[i // image.shape[1], i % image.shape[1]]
            
            # Calculamos la diferencia absoluta máxima entre el píxel actual y sus vecinos
            for neighbor_pos in neighbor_positions:
                neighbor_pixel_value = image[neighbor_pos // image.shape[1], neighbor_pos % image.shape[1]]
                diff = abs(current_pixel_value - neighbor_pixel_value)
                max_difference = max(max_difference, diff)
    
    # Devolvemos el valor máximo encontrado
    return max_difference

def calculate_weight(image, beta, sigma, pixel_i, pixel_j):
    # Obtener los valores de los píxeles
    intensity_i = image[pixel_i // image.shape[1], pixel_i % image.shape[1]]
    intensity_j = image[pixel_j // image.shape[1], pixel_j % image.shape[1]]
    
    # Calcular la diferencia de intensidad
    intensity_difference = np.abs(intensity_i - intensity_j)
    
    # Calcular el peso usando la fórmula proporcionada
    weight = np.exp((-beta * intensity_difference ** 2) / sigma)
    
    return weight

def calculate_weighted_valency(image, E, beta, sigma, pixel_i):
    weighted_valency = 0
    
    # Iterar sobre los vecinos de pixel_i
    for neighbor in E[pixel_i]:
        _, pixel_j = neighbor
        # Calcular el peso entre pixel_i y pixel_j
        weight = calculate_weight(image, beta, sigma, pixel_i, pixel_j)
        # Sumar el peso al valor de la valencia ponderada
        weighted_valency += weight
        
    return weighted_valency

def calculate_weighted_valencies(image, neighbor_matrix, beta, sigma):
    weighted_valencies = np.zeros(image.size, dtype=float)
    
    # Iterar sobre todos los píxeles de la imagen
    for pixel_i in range(image.size):
        # Calcular la valencia ponderada para el píxel actual
        weighted_valency = calculate_weighted_valency(image, neighbor_matrix, beta, sigma, pixel_i)
        # Almacenar el resultado en el vector de valencias ponderadas
        weighted_valencies[pixel_i] = weighted_valency
        
    return weighted_valencies

def generate_weighted_adjacency_matrix(image, E, beta, sigma):
    num_pixels = image.shape[0] * image.shape[1]
    W = np.zeros((num_pixels, num_pixels))
    
    for pixel_i in range(num_pixels):
        for neighbor in E[pixel_i]:
            _, pixel_j = neighbor
            weight = calculate_weight(image, beta, sigma, pixel_i, pixel_j)
            W[pixel_i, pixel_j] = weight
    
    return W

def laplacian_coordinates(image_data, canvas_data):
  
    # image = resize_image(image_data)

    image = np.array([[0, 1, 2, 3], 
                   [4, 5, 6, 7], 
                   [8, 9, 10, 11], 
                   [12, 13, 14, 15]])

    B, F = identify_and_store_drawing(canvas_data)

    E = generate_neighbor_matrix(image)

    beta = 0.5  # Parámetro beta
    sigma = calculate_sigma(image, E)
    weighted_valencies_d_i = calculate_weighted_valencies(image, E, beta, sigma)

    D = np.diag(weighted_valencies_d_i)
    W = generate_weighted_adjacency_matrix(image, E, beta, sigma)


    print("*** image: ", image)
    print("=== B: ", B)
    print("=== F: ", F)
    print("=== E: ", E)
    print("=== sigma: ", sigma)
    print("=== weighted_valencies_d_i: ", weighted_valencies_d_i)
    print("=== D: ", D)
    print("=== W: ", W)

    
    return image
