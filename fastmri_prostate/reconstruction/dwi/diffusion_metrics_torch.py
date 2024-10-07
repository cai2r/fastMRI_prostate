import torch
import numpy as np
from functools import partial
from typing import Dict, List, Tuple

def trace(img_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the trace of the diffusion tensor at b-value 50 and 1000.

    Parameters
    ----------
    img_dict : dict
        A dictionary containing the diffusion-weighted imaging data.

    Returns
    -------
    tuple of tensors
        A tuple containing the trace of the diffusion tensor at b-value 50 and 1000, respectively.
    """

    trace_b50 = torch.pow(img_dict['b50x'] * img_dict['b50y'] * img_dict['b50z'], 1/3)
    trace_b1000 = torch.pow(img_dict['b1000x'] * img_dict['b1000y'] * img_dict['b1000z'], 1/3)
    
    return trace_b50, trace_b1000


def adc(raw_images: torch.Tensor, adc_scale: float, b_values: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the ADC (Apparent Diffusion Coefficient) 

    Parameters
    ----------
    raw_images : tensor
        Raw input diffusion-weighted images.
    adc_scale : float
        Scaling factor for the ADC map.
    b_values : list
        List of b-values used for acquiring the diffusion-weighted images.

    Returns
    -------
    tuple of tensors
        A tuple containing ADC map calculated from the input images, 
        and the baseline signal intensity (b=0)
    """
    if torch.mean(raw_images) < 1e-3:
        raw_images = 1e5 * raw_images

    log_image = torch.log(raw_images + 1.0)
    sum_log_image = torch.mean(log_image, dim=2)

    X = torch.stack((torch.tensor(b_values), torch.ones(2)), dim=1)
    Y = sum_log_image.view(-1, len(b_values)).T

    res = torch.linalg.lstsq(X, Y).solution
    # res = res[:X.size(1)]
    tmp = res[0, :].view(sum_log_image.shape[:2])
    b0_img = torch.exp(res[1, :].view(sum_log_image.shape[:2]))
    b0_img[torch.isnan(b0_img)] = 0

    adc_map = tmp * adc_scale
    adc_map[(adc_map < 0) | (torch.isnan(adc_map))] = 0
    
    return adc_map, b0_img


def b1500(adc_map: torch.Tensor, b0_img: torch.Tensor, adc_scale: float, b_values: List[int]) -> torch.Tensor:
    """
    Compute the b1500 image from the ADC map and baseline signal intensity.

    Parameters
    ----------
    adc_map : tensor
        The ADC map calculated from the input images.
    b0_img : tensor
        The baseline signal intensity when b=0.
    adc_scale : float
        The ADC scale factor.
    b_values : list of int
        The b-values used in the acquisition.

    Returns
    -------
    tensor
        The b1500 image.
    """

    noise_level = 12
    noise_threshold_max_adc = 300
    calculated_b_value = 1500
    noise_threshold_min_b0 = noise_level

    # Get noise level based on b0 intensities within threshold
    minimal_pixel_fraction = 0.01
    b0_intensity = b0_img[(adc_map < noise_threshold_max_adc) & (b0_img > noise_threshold_min_b0)]
    if len(b0_intensity) > ((minimal_pixel_fraction * adc_map.numel()) + 1):
        noise_level = torch.percentile(b0_intensity, 50) * 3
    
    # Calculate ADC offset for each pixel
    noise_estimation_adc_offset = 1000
    adc_offset = torch.where(
        (noise_level > 0) & (b0_img < noise_level), 
        noise_estimation_adc_offset * torch.sqrt(torch.maximum(1 - ((b0_img / noise_level) ** 2), torch.tensor(0.0))),
        torch.tensor(0.0)
    )
    
    # Calculate exponent for each pixel
    neg_calc_b_value = calculated_b_value / adc_scale
    neg_max_b_value = b_values[-1] / adc_scale
    tmp_exponent = (neg_calc_b_value - neg_max_b_value) * torch.maximum(adc_map, adc_offset) + neg_max_b_value * adc_map

    # Calculate b1500 image
    return b0_img * torch.exp(tmp_exponent)    


def compute_trace_adc_b1500(img_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Compute the ADC map, b-value 1500, trace for a given DWI volume.

    Parameters
    ----------
    img_dict : dict
        A dictionary containing the diffusion-weighted imaging data.

    Returns
    -------
    dict
        A dictionary containing the ADC map, b-value 1500, and trace, 
        and the original diffusion-weighted imaging data 
    """

    img_dict['trace_b50'], img_dict['trace_b1000'] = trace(img_dict)

    # adc params
    adc_scale = -1e+6
    b_values = [50, 1000]

    recon_shape = img_dict['b50x'].shape
    adc_vol = torch.zeros(size=recon_shape + (3, 2,))

    for i, b_value in enumerate([50, 1000]):
        for j, axis in enumerate(['x', 'y', 'z']):
            key = f"b{b_value}{axis}"
            adc_vol[:, :, :, j, i] = img_dict[key]
    
    # print(adc_vol.shape, adc_vol.dtype)
    print("changed")
    adc_map, b0_img = map(
        torch.stack, 
        zip(*[adc(adc_vol[sl, ...], adc_scale, b_values) for sl in range(recon_shape[0])])
    )
    
    img_dict['adc_map'] = adc_map
    img_dict['b1500'] = b1500(adc_map, b0_img, adc_scale, b_values) 
    
    return img_dict