import os
import numpy as np

from fastmri_prostate.data.mri_data import zero_pad_kspace_hdr
from fastmri_prostate.reconstruction.utils import center_crop_im, ifftnd
from fastmri_prostate.reconstruction.grappa import Grappa


def t2_reconstruction(kspace_data: np.ndarray, calib_data: np.ndarray, hdr: str) -> np.ndarray:
    """
    Perform T2-weighted image reconstruction using GRAPPA technique.

    Parameters:
    -----------
    kspace_data: numpy.ndarray
        Input k-space data with shape (num_aves, num_slices, num_coils, num_ro, num_pe)
    calib_data: numpy.ndarray
        Calibration data for GRAPPA with shape (num_slices, num_coils, num_pe_cal)
    hdr: str
         The XML header string.
         
    Returns:
    --------
    im_final: numpy.ndarray
        Reconstructed image with shape (num_slices, 320, 320)
    """
    num_aves = kspace_data.shape[0]
    num_slices = kspace_data.shape[1]
    num_coils = kspace_data.shape[2]
    num_ro = kspace_data.shape[3]
    num_pe = kspace_data.shape[4]
    num_pe_cal = calib_data.shape[3]
    
    # Calib_data shape: num_slices, num_coils, num_pe_cal
    grappa_weight_dict = {}

    kspace_slice_regridded = kspace_data[0, 0, ...]
    grappa_obj = Grappa(np.transpose(kspace_slice_regridded, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1)
    
    # calculate GRAPPA weights
    for slice_num in range(num_slices):
        calibration_regridded = calib_data[slice_num, ...]
        grappa_weight_dict[slice_num] = grappa_obj.compute_weights(
            np.transpose(calibration_regridded, (2, 0 ,1))
        )

    # apply GRAPPA weights
    kspace_post_grappa_all = np.zeros(shape=kspace_data.shape, dtype=complex)
    for average in range(num_aves):
        for slice_num in range(num_slices):
            kspace_slice_regridded = kspace_data[average, slice_num, ...]
            kspace_post_grappa = grappa_obj.apply_weights(
                np.transpose(kspace_slice_regridded, (2, 0, 1)),
                grappa_weight_dict[slice_num]
            )
            kspace_post_grappa = np.moveaxis(np.moveaxis(kspace_post_grappa, 0,1),1,2)
            kspace_post_grappa_all[average][slice_num] = kspace_post_grappa

    # recon image for each average
    im = np.zeros((num_aves, num_slices, num_ro, num_ro))
    for average in range(num_aves): 
        kspace_grappa = kspace_post_grappa_all[0,:,:,:,:]
        kspace_grappa_padded = zero_pad_kspace_hdr(hdr, kspace_grappa)
        im[average] = create_coil_combined_im(kspace_grappa_padded)

    im_3d = np.mean(im, axis = 0) 
    # center crop image to 320 x 320
    img_dict = {}
    img_dict['reconstruction_rss'] = center_crop_im(im_3d, [320, 320]) 

    return img_dict
  

def create_coil_combined_im(multicoil_multislice_kspace: np.ndarray) -> np.ndarray:
    """
    Create a coil combined image from a multicoil-multislice k-space array.
    
    Parameters:
    -----------
    multicoil_multislice_kspace : array-like
        Input k-space data with shape (slices, coils, readout, phase encode).
    
    Returns:
    --------
    image_mat : array-like
        Coil combined image data with shape (slices, x, y).
    """

    k = multicoil_multislice_kspace
    image_mat = np.zeros((k.shape[0], k.shape[2], k.shape[3]))     
    for i in range(image_mat.shape[0]):                             
        data_sl = k[i,:,:,:]                                        
        image = ifftnd(data_sl, [1,2])                             
        image = rss(image, axis = 0)                         
        image_mat[i,:,:] = np.flipud(image)                         
    return image_mat
    

def rss(sig: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute the Root Sum-of-Squares (RSS) value of a complex signal along a specified axis.

    Parameters
    ----------
    sig : np.ndarray
        The complex signal to compute the RMS value of.
    axis : int, optional
        The axis along which to compute the RMS value. Default is -1.

    Returns
    -------
    rss : np.ndarray
        The RSS value of the complex signal along the specified axis.
    """
    return np.sqrt(np.sum(abs(sig)**2, axis))
