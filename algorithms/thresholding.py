def thresholding(image, threshold):
    """
    Desc: Realiza la umbralización de una imagen dada utilizando un valor de umbral específico.
    Params:
        image (matriz): Imagen de entrada
        threshold (float): Valor de umbral
    Returns:
        segmented_image (matriz): Imagen segmentada

    """
    segmented_image = image >= threshold
    return segmented_image
