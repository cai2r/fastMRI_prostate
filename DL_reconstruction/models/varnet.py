"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Tuple, Optional

from DL_reconstruction import (
    save_reconstructions,
    rss,
    rss_complex,
    fftshift,
    ifftshift,
    roll,
    fft2c,
    ifft2c,
    complex_abs,
    complex_abs_sq,
    complex_conj,
    complex_mul,
    tensor_to_complex_np,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from data import transforms

from .unet import Unet


class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int,
        out_chans: int,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 6: #for regularizer run but not sensitivies run
            x = torch.squeeze(x,1)
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)
        mean = x.mean(dim=1).reshape(b, 1, 1, 1)
        std = x.std(dim=1).reshape(b, 1, 1, 1)
        
        x = x.reshape(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")
        
        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        
        b, c, a, h, w, comp = x.shape

        return x.view(b * c,a, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def forward(
        self,
        espirit: torch.Tensor,
    ) -> torch.Tensor:
        images, batches = self.chans_to_batch_dim(espirit)
        # estimate sensitivities from initial espirit estimates
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )
def sens_reduce(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    sens_maps = torch.unsqueeze(sens_maps, 2)
    return complex_mul(
        ifft2c(x), complex_conj(sens_maps)
    ).sum(dim=1, keepdim=False)

class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        bvalue: str = 'b50',
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
    ):
        """
        Args:
            bvalue: String to indicate b50 or b1000 model training
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super().__init__()
        #different number of in and out chans for b50 and b1000 architectures because we keep 4 averages (For B1000: 4 NEX * 3 directions * 2)
        if bvalue == 'b50':
            in_chans = 6
            out_chans = 6
        elif bvalue == 'b1000':
            in_chans = 24
            out_chans = 24

        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
        )
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools, in_chans, out_chans)) for _ in range(num_cascades)]
        )
     
    def forward(
        self,
        kspace: torch.Tensor,
        espirit: torch.Tensor,
    ) -> torch.Tensor:
        sens_maps = self.sens_net(espirit.float())
        kspace_pred = kspace.clone()
        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, kspace, sens_maps)
        result = complex_abs(sens_reduce(kspace_pred,sens_maps))
        
        return(result)


class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        sens_maps = torch.unsqueeze(sens_maps,2)
        return fft2c(complex_mul(x, sens_maps))

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        
        zero = torch.zeros(ref_kspace.shape).to(current_kspace)
        # Create a mask where each element is True if the corresponding element in ref_kspace is non-zero
        mask = torch.ne(ref_kspace, 0)
        
        soft_dc = torch.where(mask, current_kspace - ref_kspace, 0) * self.dc_weight
        
        model_term = self.sens_expand(
            self.model(sens_reduce(current_kspace, sens_maps)), sens_maps
        )

        return current_kspace - soft_dc - model_term
