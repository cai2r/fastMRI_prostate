"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn

import h5py
import numpy as np
import pandas as pd
import requests
import torch
import yaml
import math


def get_regridding_params(hdr):
    res = {
        'rampUpTime': None,
        'rampDownTime': None,
        'flatTopTime': None,
        'acqDelayTime': None,
        'echoSpacing': None
    }
    
    et_root = etree.fromstring(hdr)
    namespace = {'ns': "http://www.ismrm.org/ISMRMRD"}

    for node in et_root.findall('ns:encoding/ns:trajectoryDescription/ns:userParameterLong', namespace):
        if node[0].text in res.keys():
            res[node[0].text] = float(node[1].text)
    
    return res


def get_grid_mat(epi_params, os_factor, keep_oversampling):
    """
    Generate a matrix for gridding reconstruction.

    Parameters:
    -----------
        epi_params : (dict)
            Dictionary containing EPI sequence parameters.
        os_factor : (float)
            Oversampling factor for the readout direction.
        keep_oversampling : (bool)
            Flag to keep the readout direction oversampling.
        
    Returns:
    --------
        grid_mat (numpy.ndarray): The gridding matrix.

    """
    
    t_rampup = epi_params['rampUpTime']
    t_rampdown = epi_params['rampDownTime']
    t_flattop = epi_params['flatTopTime']
    t_delay = epi_params['acqDelayTime']

    adc_nos = 200.0
    t_adcdur = 580.0

    if keep_oversampling:
        i_pts_readout = adc_nos
    else:
        i_pts_readout = adc_nos/os_factor

    if t_rampup == 0:
        grid_mat = np.eye(i_pts_readout, adc_nos)
        return
    
    t_step = t_adcdur/(adc_nos-1)

    tt = np.linspace(t_delay, t_delay + t_adcdur, int(adc_nos))
    kk = np.zeros(shape=(int(adc_nos)))

    for zz in range(int(adc_nos)):
        if tt[zz] < t_rampup:
            kk[zz] = (0.5/t_rampup) * np.square(tt[zz])
        elif tt[zz] > (t_rampup + t_flattop):
            kk[zz] = (0.5/t_rampup) * np.square(t_rampup) + (tt[zz] - t_rampup) - (0.5/t_rampdown) * (np.square(tt[zz] - t_rampup - t_flattop))
        else:
            kk[zz] = (0.5/t_rampup) * np.square(t_rampup) + (tt[zz] - t_rampup)

    kk = kk - kk[int(np.floor(adc_nos/2))-1]
    need_kk = np.linspace(kk[0], kk[len(kk)-1], int(i_pts_readout))
    delta_k = need_kk[1] - need_kk[0]

    density = np.diff(kk)
    density = np.append(density, density[0])

    grid_mat = np.sinc(
        (np.tile(need_kk, (int(adc_nos), 1)).T - np.tile(kk, (int(i_pts_readout), 1)))/delta_k
    )

    grid_mat = np.tile(density, (int(i_pts_readout), 1)) * grid_mat
    grid_mat = grid_mat/(1e-12 + np.tile(np.sum(grid_mat, axis=1), (int(adc_nos), 1)).T)

    return grid_mat


def trapezoidal_regridding(img, epi_params):
    """
    Perform trapezoidal regridding on an image.

    Parameters:
    -----------
        img : (np.ndarray)
            3D array of the input undersampled image.
        epi_params : (dict)
            A dictionary of EPI sequence parameters.
    
    Returns:
    --------        
        np.ndarray: A 3D array representing the regridded image.

    """
    s = img.shape
    
    os_factor = 2
    keep_oversampling = True
    
    grid_mat = get_grid_mat(epi_params, os_factor, keep_oversampling)
    grid_mat = grid_mat.astype('float32')
    
    img2 = np.transpose(img, (1, 2, 0))
    s2 = img2.shape
    img2 = np.reshape(img2, (img2.shape[0], np.prod(img2.shape[1:])))
    
    img_out = grid_mat @ img2
    img_out = np.reshape(img_out, s2)
    
    img_out = np.transpose(img_out, (2, 0, 1))
    return img_out


def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)

def fetch_dir(
    key: str, data_config_file: Union[str, Path, os.PathLike] = "fastmri_dirs.yaml"
) -> Path:
    """
    Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.

    Args:
        key: key to retrieve path from data_config_file. Expected to be in
            ("prostate_path", "brain_path", "log_path").
        data_config_file: Optional; Default path config file to fetch path
            from.

    Returns:
        The path to the specified directory.
    """
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        default_config = {
            "knee_path": "/path/to/knee",
            "prostate_path": "/path/to/prostate",
            "log_path": ".",
        }
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        data_dir = default_config[key]

        warn(
            f"Path config at {data_config_file.resolve()} does not exist. "
            "A template has been created for you. "
            "Please enter the directory paths for your system to have defaults."
        )
    else:
        with open(data_config_file, "r") as f:
            data_dir = yaml.safe_load(f)[key]

    return Path(data_dir)


class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        bvalue: str,
        transform: Optional[Callable] = None,
        num_cols: Optional[Tuple[int]] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            bvalue: string for b50 or b1000: need this to determine which acquisitions 
                to pull from dwi kspace
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
        """
        self.transform = transform
        self.examples = []
        self.bvalue = bvalue
        files = list(Path(root).iterdir())
        for fname in sorted(files):
            metadata, num_slices = self._retrieve_metadata(fname)
        
            self.examples += [
                (fname, slice_ind, metadata) for slice_ind in range(num_slices)
            ]
  
        if num_cols:
            self.examples = [
                ex
                for ex in self.examples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]
    
    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])
            kspace = hf['kspace']

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )
            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1 
            
            padding_left  = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max
            num_slices = hf["kspace"].shape[1]
                      
        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, slice_ind, metadata = self.examples[i]
        with h5py.File(fname, "r") as hf:
            kspace = hf['kspace'][:, slice_ind]
            espirit = hf["coil_sens_maps"][slice_ind]
            
            regrid_params = get_regridding_params(hf["ismrmrd_header"][()])
            kspace_regridded = np.empty(shape=kspace.shape, dtype=kspace.dtype)

            for average in range(kspace.shape[0]):
                kspace_regridded[average] = trapezoidal_regridding(kspace[average,...], regrid_params)

            if self.bvalue =='b50': #for 4X acceleration of B50 acquisition, take a single NEX of x,y,z 
                kspace = kspace_regridded[[8,9,10]]
                target = hf['trace_b50'][slice_ind] #if self.recons_key in hf else None      
            elif self.bvalue == 'b1000':
                kspace = kspace_regridded[[5,6,7,11,12,13,17,18,19,23,24,25]] #for 3x acceleration of B1000, keep 4 averages of x,y,z
                target = hf['trace_b1000'][slice_ind]   

            #convert complex kspace to real and imaginary channels
            kspace = np.stack((np.real(kspace),np.imag(kspace)), axis = -1)
            espirit = np.stack((np.real(espirit),np.imag(espirit)), axis = -1)
                            
            target = np.flip(target,0)
            espirit = np.expand_dims(espirit, 1)
            kspace = np.transpose(kspace, [1,0,2,3,4])
            
            attrs = dict(hf.attrs)
            if target is not None:
                attrs['max'] = np.max(target)
            
            attrs.update(metadata)
            
        if self.transform is None:
            sample = (kspace.copy(), target.copy(), espirit, attrs, fname.name, slice_ind)
        else:
            sample = self.transform(kspace.copy(), target.copy(), espirit, attrs, fname.name, slice_ind)
        return sample

