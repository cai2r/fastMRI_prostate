"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

from DL_reconstruction import SSIMLoss
import torch
from data import transforms
from models import VarNet
import numpy as np
from .mri_module import MriModule


class VarNetModule(MriModule):
    """
    VarNet training module.

    This can be used to train variational networks from the paper:

    fastMRI Prostate ....
 
    which was inspired by earlier papers:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055–3071, 2018.
    """

    def __init__(
        self,
        bvalue: str = 'b50',
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            bvalue: String b50 or b1000 
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults. To prevent this, either set
                `num_sense_lines`, or set `skip_low_freqs` and `skip_around_low_freqs`
                to `True` in the EquispacedMaskFunc. Note that setting this value may
                lead to undesired behaviour when training on multiple accelerations
                simultaneously.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
        self.bvalue = bvalue
        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.varnet = VarNet(
            bvalue = self.bvalue,
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            pools=self.pools,
        )

        self.loss = SSIMLoss()

    def forward(self, kspace, espirit):
        return self.varnet(kspace, espirit)

    def training_step(self, batch, batch_idx):
        
        output = self.forward(
		batch.kspace,batch.espirit
        )
        
        if output.shape[1] ==12: #B1000 case, combine 4 averages 
            x = torch.sum(output[:,[0,3,6,9]],1)/4
            y = torch.sum(output[:,[1,4,7,10]],1)/4
            z = torch.sum(output[:,[2,5,8,11]],1)/4
            output = torch.pow(x*y*z,1/3) #geometric mean 
        else:
            output = torch.pow(torch.prod(output,1),1/3) #geometric mean
        
        target, output = transforms.center_crop_to_smallest(batch.target, output)
       
        output = transforms.center_crop(output.float(),[100,100])
        target = transforms.center_crop(target.float(), [100,100]) 
        loss = self.loss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
        )

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(
            batch.kspace, batch.espirit
        )
         
        if output.shape[1] ==12: #B1000 case
            x = torch.sum(output[:,[0,3,6,9]],1)/4
            y = torch.sum(output[:,[1,4,7,10]],1)/4
            z = torch.sum(output[:,[2,5,8,11]],1)/4
            output = torch.pow(x*y*z,1/3)
        else:
            output = torch.pow(torch.prod(output,1),1/3)
        
        target, output = transforms.center_crop_to_smallest(batch.target, output)
        
        
        target = transforms.center_crop(target.float(),[100,100])
        output = transforms.center_crop(output.float(), [100,100])
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
            ),
        }

    def test_step(self, batch, batch_idx):
        output = self(batch.kspace, batch.espirit)    
        if output.shape[1] ==12: #B1000 case
            x = torch.sum(output[:,[0,3,6,9]],1)/4
            y = torch.sum(output[:,[1,4,7,10]],1)/4
            z = torch.sum(output[:,[2,5,8,11]],1)/4
            output = torch.pow(x*y*z,1/3)
        else:
            output = torch.pow(torch.prod(output,1),1/3)

        output = transforms.center_crop(output.float(), [100,100])
        
        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
