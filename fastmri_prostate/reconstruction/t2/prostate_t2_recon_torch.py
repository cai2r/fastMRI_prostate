import os
import torch

from fastmri_prostate.data.mri_data import get_padding
from fastmri_prostate.reconstruction.utils_torch import center_crop_im, ifftnd
from fastmri_prostate.reconstruction.grappa_torch import Grappa


import torch
import torch.nn.functional as F

def zero_pad_kspace_hdr(hdr: str, unpadded_kspace: torch.Tensor) -> torch.Tensor:
    """
    Perform zero-padding on k-space data to have the same number of
    points in the x- and y-directions.

    Parameters
    ----------
    hdr : str
        The XML header string.
    unpadded_kspace : torch.Tensor of shape (sl, ro, coils, pe)
        The k-space data to be padded.

    Returns
    -------
    padded_kspace : torch.Tensor of shape (sl, ro_padded, coils, pe_padded)
        The zero-padded k-space data, where ro_padded and pe_padded are
        the dimensions of the readout and phase-encoding directions after
        padding.

    Notes
    -----
    The padding value is calculated using the `get_padding` function, which
    extracts the padding value from the XML header string. If the difference
    between the readout dimension and the maximum phase-encoding dimension
    is not divisible by 2, the padding is applied asymmetrically, with one
    side having an additional zero-padding.
    """
    padding = get_padding(hdr)
    if padding % 2 != 0:
        padding_left = int(torch.floor(torch.tensor(padding)))
        padding_right = int(torch.ceil(torch.tensor(padding)))
    else:
        padding_left = int(padding)
        padding_right = int(padding)

    # PyTorch's pad function requires specifying padding for each dimension in reverse order
    padded_kspace = F.pad(unpadded_kspace, (padding_left, padding_right, 0, 0, 0, 0, 0, 0))

    return padded_kspace


def t2_reconstruction(kspace_data: torch.Tensor, calib_data: torch.Tensor, hdr: str) -> torch.Tensor:
    """
    Perform T2-weighted image reconstruction using GRAPPA technique.

    Parameters:
    -----------
    kspace_data: torch.Tensor
        Input k-space data with shape (num_aves, num_slices, num_coils, num_ro, num_pe)
    calib_data: torch.Tensor
        Calibration data for GRAPPA with shape (num_slices, num_coils, num_pe_cal)
    hdr: str
         The XML header string.
         
    Returns:
    --------
    im_final: torch.Tensor
        Reconstructed image with shape (num_slices, 320, 320)
    """
    num_avg, num_slices, num_coils, num_ro, num_pe = kspace_data.shape
    
    # Calib_data shape: num_slices, num_coils, num_pe_cal
    grappa_weight_dict = {}
    grappa_weight_dict_2 = {}

    kspace_slice_regridded = kspace_data[0, 0, ...]
    grappa_obj = Grappa(kspace_slice_regridded.permute(2, 0, 1), kernel_size=(5, 5), coil_axis=1)

    kspace_slice_regridded_2 = kspace_data[1, 0, ...]
    grappa_obj_2 = Grappa(kspace_slice_regridded_2.permute(2, 0, 1), kernel_size=(5, 5), coil_axis=1)
    
    # calculate GRAPPA weights
    for slice_num in range(num_slices):
        calibration_regridded = calib_data[slice_num, ...]
        grappa_weight_dict[slice_num] = grappa_obj.compute_weights(
            calibration_regridded.permute(2, 0, 1)
        )
        grappa_weight_dict_2[slice_num] = grappa_obj_2.compute_weights(
            calibration_regridded.permute(2, 0, 1)
        )

    # apply GRAPPA weights
    kspace_post_grappa_all = torch.zeros(kspace_data.shape, dtype=torch.complex64)

    for average, grappa_obj, grappa_weight_dict in zip(
        [0, 1, 2],
        [grappa_obj, grappa_obj_2, grappa_obj],
        [grappa_weight_dict, grappa_weight_dict_2, grappa_weight_dict]
    ):
        for slice_num in range(num_slices):
            kspace_slice_regridded = kspace_data[average, slice_num, ...]
            kspace_post_grappa = grappa_obj.apply_weights(
                kspace_slice_regridded.permute(2, 0, 1),
                grappa_weight_dict[slice_num]
            )
            kspace_post_grappa_all[average, slice_num, ...] = kspace_post_grappa.permute(1, 2, 0)

    # recon image for each average
    im = torch.zeros((num_avg, num_slices, num_ro, num_ro))
    for average in range(num_avg): 
        kspace_grappa = kspace_post_grappa_all[average, ...]
        kspace_grappa_padded = zero_pad_kspace_hdr(hdr, kspace_grappa)
        im[average] = create_coil_combined_im(kspace_grappa_padded)

    im_3d = torch.mean(im, dim=0) 
    # center crop image to 320 x 320
    img_dict = {}
    img_dict['reconstruction_rss'] = center_crop_im(im_3d, [320, 320]) 

    return img_dict
  

def create_coil_combined_im(multicoil_multislice_kspace: torch.Tensor) -> torch.Tensor:
    """
    Create a coil combined image from a multicoil-multislice k-space array.
    
    Parameters:
    -----------
    multicoil_multislice_kspace : torch.Tensor
        Input k-space data with shape (slices, coils, readout, phase encode).
    
    Returns:
    --------
    image_mat : torch.Tensor
        Coil combined image data with shape (slices, x, y).
    """

    k = multicoil_multislice_kspace
    image_mat = torch.zeros((k.shape[0], k.shape[2], k.shape[3]))     
    for i in range(image_mat.shape[0]):                             
        data_sl = k[i,:,:,:]                                        
        image = ifftnd(data_sl, [1,2])                             
        image = rss(image, axis=0)                         
        image_mat[i,:,:] = torch.flipud(image)                         
    return image_mat
    

def rss(sig: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """
    Compute the Root Sum-of-Squares (RSS) value of a complex signal along a specified axis.

    Parameters
    ----------
    sig : torch.Tensor
        The complex signal to compute the RMS value of.
    axis : int, optional
        The axis along which to compute the RMS value. Default is -1.

    Returns
    -------
    rss : torch.Tensor
        The RSS value of the complex signal along the specified axis.
    """
    return torch.sqrt(torch.sum(sig.abs()**2, dim=axis))