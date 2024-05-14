import numpy as np
import scipy


kernel_y = np.array([[0,-1,0], [0,0,0], [0,1,0]]) / 2
kernel_x = np.array([[0,0,0], [-1,0,1], [0,0,0]]) / 2
kernel_adelante = np.array([[0,0,0], [0,-1,1], [0,0,0]]) 


def border_x(img, x_slice, y_slice, z_slice):
    img_filt_x_x = scipy.ndimage.convolve(img[x_slice, :, :], kernel_x)
    img_filt_x_y = scipy.ndimage.convolve(img[:, y_slice, :], kernel_x)
    img_filt_x_z = scipy.ndimage.convolve(img[:, :, z_slice], kernel_x)
    return img_filt_x_x, img_filt_x_y, img_filt_x_z


def border_y(img, x_slice, y_slice, z_slice):
    img_filt_y_x = scipy.ndimage.convolve(img[x_slice, :, :], kernel_y)
    img_filt_y_y = scipy.ndimage.convolve(img[:, y_slice, :], kernel_y)
    img_filt_y_z = scipy.ndimage.convolve(img[:, :, z_slice], kernel_y)
    return img_filt_y_x, img_filt_y_y, img_filt_y_z


def magnitud(img_filt_x, img_filt_y):
   
    image_filt_x = np.sqrt(img_filt_x[0] ** 2 + img_filt_y[0] ** 2)
    image_filt_y = np.sqrt(img_filt_x[1] ** 2 + img_filt_y[1] ** 2)
    image_filt_z = np.sqrt(img_filt_x[2] ** 2 + img_filt_y[2] ** 2)

    return image_filt_x, image_filt_y, image_filt_z
