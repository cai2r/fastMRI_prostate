import logging
import torch
from pathlib import Path
from time import time
from typing import Dict, Tuple
import xml.etree.ElementTree as etree

from fastmri_prostate.reconstruction.dwi.regridding_torch import trapezoidal_regridding
from fastmri_prostate.reconstruction.dwi.diffusion_metrics_torch import compute_trace_adc_b1500
from fastmri_prostate.reconstruction.grappa_torch import Grappa
from fastmri_prostate.reconstruction.utils_torch import ifftnd, flip_im, center_crop_im

def compute_averages(img_vol: torch.Tensor) -> Dict:
    """
    Computes the average of the given image volume for different diffusion-weighted directions.

    Parameters:
    ----------
        img_vol : torch.Tensor
            The input image volume containing diffusion-weighted images.

    Returns:
    -------
        dict: A dictionary containing the computed averages for different diffusion-weighted directions.

    Notes:
    -----
    There are 4 averages for each b50 diffusion direction and 12 averages for each b1000 direction
    """

    return {
        'b50x': torch.sum(img_vol[2:21:6, ...], dim=0) / 4,
        'b50y': torch.sum(img_vol[3:22:6, ...], dim=0) / 4,
        'b50z': torch.sum(img_vol[4:23:6, ...], dim=0) / 4,
        'b1000x': torch.sum(
            torch.cat([
                img_vol[5:24:6, ...],
                img_vol[26:48:3, ...]
            ], dim=0), dim=0
        ) / 12,
        'b1000y': torch.sum(
            torch.cat([
                img_vol[6:25:6, ...],
                img_vol[27:49:3, ...]
            ], dim=0), dim=0
        ) / 12,        
        'b1000z': torch.sum(
            torch.cat([
                img_vol[7:26:6, ...],
                img_vol[28:50:3, ...]
            ], dim=0), dim=0
        ) / 12,
    }


def dwi_reconstruction(kspace: torch.Tensor, calibration: torch.Tensor, coil_sens_maps: torch.Tensor, hdr: Dict) -> Dict:
    """ The reconstruction uses trapezoidal regridding to regrid the k-space data and computes GRAPPA weights for each slice 
    of the input k-space data using the calibration data. It applies the computed GRAPPA weights to the k-space data 
    to obtain image data, which is then combined with the coil sensitivity maps to reconstruct the DWI images. 
    The resulting images are cropped and returned as a dictionary with b50, b1000, trace, ADC, and b1500 values.

    Parameters:
    -----------
    kspace : torch.Tensor
        The k-space data with dimensions (averages, slices, coils, readout, phase).
    calibration : torch.Tensor
        The calibration data with dimensions (slices, coils, readout, phase).
    coil_sens_maps : torch.Tensor
        The coil sensitivity maps with dimensions (slices, coils, readout, phase).
    hdr : dict
        The header information for the diffusion-weighted imaging.

    Returns:
    --------
    img_dict : dict
        A dictionary containing the reconstructed DW images and trace, ADC, and b1500 values

    """   
    
    kspace_slice_regridded = trapezoidal_regridding(kspace[0, 0, ...], hdr)
    grappa_obj = Grappa(kspace_slice_regridded.permute(2, 0, 1), kernel_size=(5, 5), coil_axis=1)

    grappa_weight_dict = {}
    for slice_num in range(kspace.shape[1]):
        t = time()
        calibration_regridded = trapezoidal_regridding(calibration[slice_num, ...], hdr)
        grappa_weight_dict[slice_num] = grappa_obj.compute_weights(
            calibration_regridded.permute(2, 0 ,1)
        )
        print(f"time taken to compute weights for slice {slice_num}: {time() - t}")
    img_post_grappa = torch.zeros(size=kspace.shape, dtype=torch.complex64, device=kspace.device)
    for average in range(kspace.shape[0]):
        t = time()
        for slice_num in range(kspace.shape[1]):
            kspace_slice_regridded = trapezoidal_regridding(kspace[average, slice_num, ...], hdr)
            # print(f"time taken to trapezoidal_regridding for average {average}/{kspace.shape[0]}: {time() - t}")
            # t = time()
            kspace_post_grappa = grappa_obj.apply_weights(
                kspace_slice_regridded.permute(2, 0, 1),
                grappa_weight_dict[slice_num]
            )
            # print(f"time taken to apply weights for average {average}/{kspace.shape[0]}, slice {slice_num}/{kspace.shape[1]}: {time() - t}")
            # t = time()
            img = ifftnd(kspace_post_grappa, [0, -1])
            # print(f"time taken to ifftnd for average {average}/{kspace.shape[0]}: {time() - t}")
            # t = time()
            img_post_grappa[average][slice_num] = img.permute(1, 2, 0)
            # print(f"time taken to permute for average {average}/{kspace.shape[0]}: {time() - t}")
        print(f"time taken to apply weights for average {average}/{kspace.shape[0]}: {time() - t}")
        # if average % 5 == 0:
        #     logging.info("Processed {0} averages of {1}".format(average, kspace.shape[0]))

    img_vol = torch.zeros(size=(kspace.shape[0], kspace.shape[1], kspace.shape[3], kspace.shape[4]), dtype=torch.complex64, device=kspace.device)

    for average in range(img_post_grappa.shape[0]):
        t = time()
        coil_comb_img = torch.sum(img_post_grappa[average] * (coil_sens_maps.conj()), dim=1)
        img_vol[average] = coil_comb_img
        print(f"time taken to multiply coil se for average {average}/{kspace.shape[0]}: {time() - t}")

    img_vol = torch.abs(img_vol)

    t = time()
    img_dict = compute_averages(img_vol)
    print(f"time taken to compute_averages function: {time() - t}")

    t = time()
    img_dict = compute_trace_adc_b1500(img_dict)
    print(f"time taken to compute trace, ADC, b1500: {time() - t}")

    center_crop_size = (100, 100)
    for src_img in img_dict.keys():
        img_dict[src_img] = center_crop_im(flip_im(img_dict[src_img], 0), center_crop_size)

    return img_dict