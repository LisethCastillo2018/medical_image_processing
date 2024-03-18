import numpy as np

from .thresholding import thresholding

def isodata_thresholding(image, tolerance):
    """
    Desc: Realiza la umbralización de la imagen dada utilizando un valor de tolerancia específico. 
          El valor de umbral es calculado.
    Params:
        image (matriz): Imagen de entrada
        tolerance (float): Valor de tolerancia
    Returns:
        segmented_image (matriz): Imagen segmentada
    """

    prev_tau = np.mean(image)

    while True:
        img_th = thresholding(image, prev_tau)
        m_foreground = np.mean(image[img_th == 1])
        m_background = np.mean(image[img_th == 0])
        new_tau = 0.5 * (m_foreground + m_background)

        if abs(new_tau - prev_tau) < tolerance:
            break
        prev_tau = new_tau

    return thresholding(image, new_tau)