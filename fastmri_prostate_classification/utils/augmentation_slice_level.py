import scipy.ndimage as ndimage
from random import randrange
import skimage 
import random
import numpy as np


def augment_image_diffusion(b1500_in, adc_in):
    """
    Augments diffusion-weighted images by applying random transformations.

    Parameters:
    - b1500_in (numpy.ndarray): Input diffusion-weighted image at b=1500.
    - adc_in (numpy.ndarray): Input apparent diffusion coefficient (ADC) map.

    Returns:
    - augmented_b1500 (numpy.ndarray): Augmented diffusion-weighted image at b=1500.
    - augmented_adc (numpy.ndarray): Augmented ADC map.
    - operation_list (numpy.ndarray): List of applied augmentation operations.
    """
    augmented_im = np.stack((b1500_in,adc_in),axis = 0)
    
    operation_list = np.unique(np.random.choice(3, 3, replace=True))
        
    if 0 in operation_list:
        # (1) Translation of image in x, y and z (this will move the lesion and feature map around)
        # x and y will have max translation of 15, z will have max = 2 to avoid missing information
        dim1 = randrange(-3,3)
        dim2 = randrange(-16,16)
        for i in range(2):
            augmented_im[i,:,:] = ndimage.shift(augmented_im[i,:,:], [dim1,dim2]) 
    
    if 1 in operation_list:
        # (2) Flip image in LR direction
        for i in range(2):
            augmented_im[i,:,:] = np.flip(augmented_im[i,:,:], axis = 1) 

    if 2 in operation_list: 
        # (3) Rotate in x-y, between -12 and 12 degrees
        angle = random.randint(-12, 12)
        for i in range(2):
            augmented_im[i,:,:] = ndimage.rotate(augmented_im[i,:,:], angle, axes=(0,1), mode='constant', cval=0.0, reshape=False)

    return augmented_im[0,:,:], augmented_im[1,:,:], operation_list

def augment_image_t2(input_im_2d):
    """
    Augments a 2D image by applying random transformations.

    Parameters:
    - input_im_2d (numpy.ndarray): Input 2D image.

    Returns:
    - augmented_im (numpy.ndarray): Augmented 2D image.
    - operation_list (numpy.ndarray): List of applied augmentation operations.
    """

    augmented_im = input_im_2d
    
    operation_list = np.unique(np.random.choice(4, 4, replace=True))
        
    if 0 in operation_list:
        # (1) Translation of image in x, y and z (this will move the lesion and feature map around)
        # x and y will have max translation of 15, z will have max = 2 to avoid missing information
        augmented_im = ndimage.shift(augmented_im, [randrange(-3,3), randrange(-16,16)]) # shift image in either neg or pos direction, 15 in x-y plane
    
    if 1 in operation_list:
        # (2) Contrast stretching (Maximise contrast at different values)
        # maximum lower bound = 10%, minimum higher bound = 90%
        lower_bound = np.percentile(augmented_im, randrange(10))                                                    
        upper_bound = np.percentile(augmented_im, randrange(90,100))                                               
        augmented_im = skimage.exposure.rescale_intensity(augmented_im, in_range=(lower_bound, upper_bound))        
    
    if 2 in operation_list:
        # (3) Flip image in LR direction
        augmented_im = np.flip(augmented_im, axis = 1) # flipped in LR

    if 3 in operation_list: 
        # (4)   Rotate in x-y, between -12 and 12 degrees
        angle = random.randint(-12, 12) 
        augmented_im = ndimage.rotate(augmented_im, angle, axes=(0,1), mode='constant', cval=0.0, reshape=False)

    return augmented_im, operation_list




