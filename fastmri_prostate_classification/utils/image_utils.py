import numpy as np
import cv2
import torch
import torchvision.transforms as T

def normalise_adc(adc_image_in):
    """
    Normalize an apparent diffusion coefficient (ADC) image.

    Parameters:
    - adc_image_in (numpy.ndarray): Input ADC image.

    Returns:
    - adc_image_out (numpy.ndarray): Normalized ADC image.
    """
    """ this normalisation scheme: https://pubmed.ncbi.nlm.nih.gov/34245943/ """
    upper_bound = 3000
    adc_image_in[adc_image_in > 3000] = 3000
    adc_image_out = adc_image_in/np.max(adc_image_in)

    return adc_image_out


def center_crop_2d(im_2d, crop_to_size):
    """
    Crop a 2D image to the center.

    Parameters:
    - im_2d (numpy.ndarray): Input 2D image.
    - crop_to_size (tuple): Target size (height, width).

    Returns:
    - cropped_im (numpy.ndarray): Cropped 2D image.
    """
    x_crop = im_2d.shape[1]/2 - crop_to_size[0]/2
    y_crop = im_2d.shape[0]/2 - crop_to_size[1]/2
    return im_2d[int(y_crop):int(crop_to_size[1] + y_crop), int(x_crop):int(crop_to_size[0] + x_crop)]  

def normalisation_2d(image_2d, type_of_norm):
    """
    Apply different types of normalization to a 2D image.

    Parameters:
    - image_2d (numpy.ndarray): Input 2D image.
    - type_of_norm (int): Type of normalization to apply (1-5).

    Returns:
    - image_2d_out (numpy.ndarray): Normalized 2D image.
    """
    if type_of_norm == 1:   
        # Normalisation TYPE 1: 
        # Clip image at 99% and 1% - then divide by max value to scale values between [0,1]
        upper_lim = np.percentile(image_2d[:], 99)
        lower_lim = np.percentile(image_2d[:], 1)
        image_2d_out = image_2d
        image_2d_out[image_2d_out > upper_lim] = upper_lim
        image_2d_out[image_2d_out < lower_lim] = lower_lim
        image_2d_out = image_2d_out/np.max(image_2d_out)

    if type_of_norm == 2:
        # Normalisation TYPE 2 (standardisation):
        # image = (image - mean(image))/std(image)
        mean = np.mean(image_2d, axis=(0,1), keepdims=True) # get mean of image along x,y
        std = np.std(image_2d, axis=(0,1), keepdims=True)   # get std of image along x,y
        image_2d_out = (image_2d - mean) / std              # get final image
    
    if type_of_norm == 3:
        # Normalisation TYPE 3:
        # min max scaling
        image_2d_out = (image_2d - np.min(image_2d)) / (np.max(image_2d) - np.min(image_2d)) 
    
    if type_of_norm == 4:
        # Normalisation TYPE 4:
        # normalise by 3times SD (Gaussian type curve assumed) and then scale 0-1
        image_2d_out = image_2d / (np.nanmean(image_2d) + 3 * np.nanstd(image_2d))
        image_2d_out[image_2d_out > 1] = 1
        image_2d_out[image_2d_out < 0] = 0 

    if type_of_norm == 5:
        image_2d_out = cv2.normalize(image_2d, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
        normalize = T.Normalize(
            mean=[0.19233476646025852], std=[0.1625128199949673]
            )
        image_2d_out = normalize(torch.unsqueeze(torch.tensor(image_2d_out),0))
        image_2d_out = np.squeeze(image_2d_out.numpy())


    return image_2d_out

def norm_adc(image_2d):
    """
    Normalize an apparent diffusion coefficient (ADC) 2D image.

    Parameters:
    - image_2d (numpy.ndarray): Input ADC 2D image.

    Returns:
    - image_2d_out (numpy.ndarray): Normalized ADC 2D image.
    """
    image_2d_out = np.where(image_2d > 3053, 3053, image_2d)
    image_2d_out = image_2d_out/np.max(image_2d_out)
    return image_2d_out

def norm_b1500(image_2d):
    """
    Normalize a b=1500 2D image.

    Parameters:
    - image_2d (numpy.ndarray): Input b=1500 2D image.

    Returns:
    - image_2d_out (numpy.ndarray): Normalized b=1500 2D image.
    """
    mean = np.mean(image_2d, axis=(0,1), keepdims=True) # get mean of image along x,y
    std = np.std(image_2d, axis=(0,1), keepdims=True)   # get std of image along x,y
    image_2d_out = (image_2d - mean) / std              # get final image
    return image_2d_out

def diffusion_resize(image_3d, resize_shape):
    """
    Resize a diffusion-weighted 3D image.

    Parameters:
    - image_3d (numpy.ndarray): Input diffusion-weighted 3D image.
    - resize_shape (tuple): Target size (height, width).

    Returns:
    - image_out (numpy.ndarray): Resized diffusion-weighted 3D image.
    """
    image_out = np.zeros((image_3d.shape[0],resize_shape[0], resize_shape[1]))
    for i in range(image_3d.shape[0]):
        image_out[i,:,:] = cv2.resize(image_3d[i,:,:], (resize_shape[0], resize_shape[1]), interpolation = cv2.INTER_AREA)
    return image_out
