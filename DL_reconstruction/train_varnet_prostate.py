"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser
from pathlib import Path
import pytorch_lightning as pl
from data.mri_data import fetch_dir
from data.transforms import VarNetDataTransform
from pl_modules import FastMriDataModule, VarNetModule
import torch

torch.cuda.device_count()
def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
       
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        bvalue = args.bvalue,
        train_transform=VarNetDataTransform(mask_func = None), #no retrospective undersampling of kspace
        val_transform=VarNetDataTransform(mask_func = None),
        test_transform=VarNetDataTransform(mask_func = None),
        test_path = args.test_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.strategy in ("ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------
    model = VarNetModule(
        bvalue = args.bvalue,
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=args.sens_chans,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        # Ensure the checkpoint path is provided
        if args.state_dict_file is None:
            raise ValueError("Checkpoint path must be provided for testing mode with --state_dict_file argument.")
        # Use the provided checkpoint path
        trainer.test(model, ckpt_path=args.state_dict_file, datamodule=data_module)
    
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("../../fastmri_dirs.yaml")
    backend = "ddp"
    num_gpus = 1 if backend == "ddp" else 1
    batch_size = 1

    # set defaults based on optional directory config
    data_path = fetch_dir("prostate_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "varnet" / "varnet_demo"

    
    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )
   # data transform params
    parser.add_argument(
        "--state_dict_file",
        default=None,
        type=str,  
        help="Path to the model checkpoint for testing",
)

# data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path, # path to fastMRI prostate data
        bvalue = "b50",
        batch_size=batch_size,  # number of samples per batch
        test_path = None,
    )

    # module config
    parser = VarNetModule.add_model_specific_args(parser)
    parser.set_defaults(
        bvalue = 'b50',
        num_cascades=10,  # number of unrolled iterations
        pools=4,  # number of pooling layers for U-Net
        chans=20,  # number of top-level channels for U-Net
        sens_pools=4,  # number of pooling layers for sense est. U-Net
        sens_chans=8,  # number of top-level channels for sense est. U-Net
        lr=0.001,  # Adam learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    
    parser.set_defaults(
        devices=num_gpus,  # number of gpus to use
        strategy=backend,  # distributed training strategy
        accelerator = 'gpu',
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=100,  # max number of epochs
    )

    args = parser.parse_args()
    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
