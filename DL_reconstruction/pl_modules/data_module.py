"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union

import data
import pytorch_lightning as pl
import torch
from data import SliceDataset

def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    """worker_info = torch.utils.data.get_worker_info()
    data: Union[
        SliceDataset, CombinedSliceDataset
    ] = worker_info.dataset  # pylint: disable=no-member
    """
    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True


def _check_both_not_none(val1, val2):
    if (val1 is not None) and (val2 is not None):
        return True

    return False


class FastMriDataModule(pl.LightningDataModule):
    """
    Data module class for fastMRI data sets.

    This class handles configurations for training on fastMRI data. It is set
    up to process configurations independently of training modules.

    Note that subsampling mask and transform configurations are expected to be
    done by the main client training scripts and passed into this data module.

    For training with ddp be sure to set distributed_sampler=True to make sure
    that volumes are dispatched to the same GPU for the validation loop.
    """

    def __init__(
        self,
        data_path: Path,
        bvalue: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        test_path: Optional[Path] = None,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
    ):
        """
        Args:
            data_path: Path to root data directory. For example, if knee/path
                is the root directory with subdirectories multicoil_train and
                multicoil_val, you would input knee/path for data_path.
            bvalue: a string, b50 or b1000, do determine which data we need from dwi kspace
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            test_path: An optional test path. Passing this overwrites data_path
                and test_split.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler. This
                should be set to True if training with ddp.
        """
        super().__init__()

        self.data_path = data_path
        self.bvalue = bvalue
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(
        self,
        bvalue: str,
        data_transform: Callable,
        data_partition: str,
    ) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
        else:
            is_train = False
        if self.test_path is not None:
           data_path = self.test_path
        else:
           data_path = self.data_path / f"{data_partition}/DIFFUSION"

        dataset = SliceDataset(
            root=data_path,
            bvalue=bvalue,
            transform=data_transform,
        )

        # ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None
        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                sampler = data.VolumeSampler(dataset, shuffle=False)
                
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )

        return dataloader

    def train_dataloader(self):
        return self._create_data_loader(self.bvalue, self.train_transform, data_partition="training")

    def val_dataloader(self):
        return self._create_data_loader(self.bvalue, self.val_transform, data_partition="validation")

    def test_dataloader(self):
        return self._create_data_loader(self.bvalue, self.test_transform, data_partition="test")

    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--data_path",
            default=None,
            type=Path,
            help="Path to fastMRI data root",
        )
        # client arguments
        parser.add_argument(
            "--bvalue",
            default="b50",
            choices=("b50", "b1000"),
            type=str,
            help="B50 or B1000 model?",
        )

        parser.add_argument(
            "--test_path",
            default=None,
            type=Path,
            help="Path to data for test mode. This overwrites data_path and test_split",
        )
        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parser
