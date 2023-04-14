import numpy as np

from numpy.fft import fftshift, ifftshift, ifftn
from typing import List, Optional, Sequence, Tuple


def ifftnd(kspace: np.ndarray, axes: Optional[Sequence[int]] = [-1]) -> np.ndarray:
    """
    Compute the n-dimensional inverse Fourier transform of the k-space data along the specified axes.

    Parameters:
    -----------
    kspace: np.ndarray
        The input k-space data.
    axes: list or tuple, optional
        The list of axes along which to compute the inverse Fourier transform. Default is [-1].

    Returns:
    --------
    img: ndarray
        The output image after inverse Fourier transform.
    """

    if axes is None:
        axes = range(kspace.ndim)
    img = fftshift(ifftn(ifftshift(kspace, axes=axes), axes=axes), axes=axes)   
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))    

    return img


def flip_im(vol, slice_axis):
    """
    Flips a 3D image volume along the slice axis.

    Parameters
    ----------
    vol : numpy.ndarray of shape (slices, height, width)
        The 3D image volume to be flipped.
    slice_axis : int
        The slice axis along which to perform the flip

    Returns
    -------
    numpy.ndarray
        The flipped 3D image volume 
    """

    for i in range(vol.shape[slice_axis]):
        vol[i] = np.flipud(vol[i])
    return vol

  
def center_crop_im(im_3d: np.ndarray, crop_to_size: Tuple[int, int]) -> np.ndarray:
    """
    Center crop an image to a given size.
    
    Parameters:
    -----------
    im_3d : numpy.ndarray
        Input image of shape (slices, x, y).
    crop_to_size : list
        List containing the target size for x and y dimensions.
    
    Returns:
    --------
    numpy.ndarray
        Center cropped image of size {slices, x_cropped, y_cropped}. 
    """
    x_crop = im_3d.shape[2]/2 - crop_to_size[0]/2
    y_crop = im_3d.shape[1]/2 - crop_to_size[1]/2

    return im_3d[:, int(y_crop):int(crop_to_size[1] + y_crop), int(x_crop):int(crop_to_size[0] + x_crop)]  

