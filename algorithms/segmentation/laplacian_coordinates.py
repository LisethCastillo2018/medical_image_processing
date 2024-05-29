import numpy as np


from utils.utils import identify_and_store_drawing

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
    
    return max_difference

def calculate_weight(image, beta, sigma, pixel_i, pixel_j):
    intensity_i = image[pixel_i // image.shape[1], pixel_i % image.shape[1]]
    intensity_j = image[pixel_j // image.shape[1], pixel_j % image.shape[1]]
    
    intensity_difference = np.abs(intensity_i - intensity_j)
    weight = np.exp((-beta * intensity_difference ** 2) / sigma)
    
    return weight

def calculate_weighted_valency(image, E, beta, sigma, pixel_i):
    weighted_valency = 0
    
    for neighbor in E[pixel_i]:
        _, pixel_j = neighbor
        # Calcular el peso entre pixel_i y pixel_j
        weight = calculate_weight(image, beta, sigma, pixel_i, pixel_j)
        weighted_valency += weight
        
    return weighted_valency

def calculate_weighted_valencies(image, neighbor_matrix, beta, sigma):
    weighted_valencies = np.zeros(image.size, dtype=float)
    
    for pixel_i in range(image.size):
        # Calcular la valencia ponderada para el píxel actual
        weighted_valency = calculate_weighted_valency(image, neighbor_matrix, beta, sigma, pixel_i)
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

def create_IS(image_shape, B, F):
    num_rows, num_cols = image_shape
    
    I_S = np.zeros((num_rows * num_cols, num_rows * num_cols))
    
    # Asignar valores en I_S para los píxeles en S = B ∪ F
    for pixel_index in range(num_rows * num_cols):
        row_index = pixel_index // num_cols
        col_index = pixel_index % num_cols
        
        if (row_index, col_index) in B or (row_index, col_index) in F:
            I_S[pixel_index, pixel_index] = 1
                
    return I_S

def create_b(image_shape, B, F, x_B, x_F):
    num_rows, num_cols = image_shape
    b = np.zeros((num_rows * num_cols, 1))
    
    # Asignar el valor de x_B o x_F en b según corresponda
    for pixel_index in range(num_rows * num_cols):
        row_index = pixel_index // num_cols
        col_index = pixel_index % num_cols
        # print("(row_index, col_index) = ", f"({row_index}, {col_index})")
        if (row_index, col_index) in B:
            b[pixel_index] = x_B
        elif (row_index, col_index) in F:
            b[pixel_index] = x_F
                
    return b

def segment_image(x, x_B, x_F):
    segmented_image = np.where(x >= (x_B + x_F) / 2, x_F, x_B)
    return segmented_image

def resize_traces(traces, original_shape, resized_shape):
    resized_traces = []
    for trace in traces:
        resized_trace = (int(trace[0] * resized_shape[0] / original_shape[0]),
                         int(trace[1] * resized_shape[1] / original_shape[1]))
        resized_traces.append(resized_trace)
    return resized_traces

def laplacian_coordinates(image_data, canvas_data, original_shape):
    """
    image_data: de .nii image
    canvas_data: data del canvas
    """
    # print("*** image_data: ", image_data)
    print("*** == shape: ", image_data.shape)
    image = image_data

    # image = np.array([[1, 2, 10, 3, 4, 10], 
    #                   [7, 4, 120, 3, 17, 10], 
    #                   [7, 120, 120, 110, 16, 10],
    #                   [7, 120, 100, 110, 16, 10],
    #                   [7, 6, 100, 5, 16, 10],
    #                   [8, 11, 10, 9, 15, 10]])

    # Identificar y almacenar los trazos del usuario como B y F
    B, F = identify_and_store_drawing(canvas_data)
    B = resize_traces(B, original_shape, image.shape)
    F = resize_traces(F, original_shape, image.shape)
    # B = [(0,4), (1,4), (2,4), (3,4)]
    # F = [(2,1), (2,2), (2,3)]
    E = generate_neighbor_matrix(image)

    beta = 0.5  
    sigma = calculate_sigma(image, E)
    weighted_valencies_d_i = calculate_weighted_valencies(image, E, beta, sigma)

    D = np.diag(weighted_valencies_d_i)
    W = generate_weighted_adjacency_matrix(image, E, beta, sigma)
    L = D - W

    # Asignar valores iniciales de umbralización para x_B y x_F
    x_B = 0 
    x_F = 1 

    I_S = create_IS(image.shape, B, F)
    b = create_b(image.shape, B, F, x_B, x_F)
    x = np.linalg.solve(I_S + np.dot(L, L), b)

    segmented_image = segment_image(x, x_B, x_F)
    segmented_image = segmented_image.reshape(image.shape)  # Redimensionar la imagen segmentada

    # print('*** shape: ', image.shape)
    # print("*** image: ", image)
    # print("=== B: ", B)
    # print("=== F: ", F)
    # print("=== E: ", E)
    # print("=== sigma: ", sigma)
    # print("=== weighted_valencies_d_i: ", weighted_valencies_d_i)
    # print("=== D: ", D)
    # print("=== W: ", W)
    # print("=== L: ", L)
    # print("=== I_S: ", I_S)
    # print("=== b: ", b)
    # print("=== x: ", x)
    print("=== segmented_image: ", segmented_image)

    return segmented_image
