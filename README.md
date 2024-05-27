# FastMRI Prostate

[[`Paper`](https://www.nature.com/articles/s41597-024-03252-w)] [[`Dataset`](https://fastmri.med.nyu.edu/)] [[`Github`](https://github.com/cai2r/fastMRI_prostate)] [[`BibTeX`](#cite)]

### Updates
02-07-2024: Updated [files](https://github.com/cai2r/fastMRI_prostate/pull/11) for slice-, volume-, exam-level labels and their paths for T2 and Diffusion sequences in the [fastMRI prostate dataset](https://fastmri.med.nyu.edu/).

[Classification](https://github.com/cai2r/fastMRI_prostate/tree/main/fastmri_prostate_classification): The classification folder contains code for training deep learning models to detect clinically significant prostate cancer.
[Reconstruction](https://github.com/cai2r/fastMRI_prostate/tree/main/DL_reconstruction): The reconstruction folder contains code for training deep learning models for reconstructing diffusion MRI images from undersampled k-space.

## Overview

This repository contains code to facilitate the reconstruction of prostate T2 and DWI (Diffusion-Weighted Imaging) images from raw (k-space) data from the fastMRI Prostate dataset. It includes reconstruction methods along with utilities for pre-processing and post-processing the data. 

The package is intended to serve as a starting point for those who want to experiment and develop alternate reconstruction techniques. 

## Installation

The code requires `python >= 3.9`

Install FastMRI Prostate: clone the repository locally and install with

```
pip install git+https://github.com/cai2r/fastMRI_prostate.git
```

## Usage
The repository is centered around the ```fastmri_prostate``` package. The following breaks down the basic structure:

```fastmri_prostate```: Contains a number of basic tools for T2 and DWI reconstruction
 - ```fastmri_prostate.data```: Provides data utility functions for accessing raw data fields like kspace, calibration, phase correction, and coil sensitivity maps.
 - ```fastmri.reconstruction.t2```: Contains functions required for prostate T2 reconstruction
 - ```fastmri.reconstruction.dwi```: Contains functions required for prostate DWI reconstruction

```fastmri_prostate_recon.py``` contains code to read files from the dataset and call the T2 and DWI reconstruction functions for a single h5 file. 

```fastmri_prostate_tutorial.ipynb``` walks through an example of loading a h5 file from the fastMRI prostate dataset and reconstructing T2/DW images.

To reconstruct T2/DW images from the fastMRI prostate raw data, users can [download the dataset](https://fastmri.med.nyu.edu/) and run ```fastmri_prostate_recon.py``` with appropriate arguments, specifying the path to the root of the downloaded dataset, output path to store reconstructions, and the sequence (T2, DWI, or both).
```
python fastmri_prostate_recon.py \  
    --data_path <path to dataset> \  
    --output_path <path to store recons> \  
    --sequence <t2/dwi/both>
```

## Hardware Requirements
The reconstruction algorithms implemented in this package requires the following hardware:
- A computer with at least 32GB of RAM
- A multi-core CPU

### Run Time
The run time of a single T2 reconstruction takes ~15 minutes while the Diffusion Weighted reconstructions take ~7 minutes on a multi-core CPU Linux machine with 64GB RAM. A bulk of the time is spent in applying GRAPPA weights to the undersampled raw kspace data.

## License
fastMRI_prostate is MIT licensed, as found in [LICENSE file](https://github.com/cai2r/fastMRI_prostate/blob/main/LICENSE)

## Cite
If you use the fastMRI Prostate data or code in your research, please use the following BibTeX entry.

```
@article{tibrewala2024fastmri,
  title={FastMRI Prostate: A public, biparametric MRI dataset to advance machine learning for prostate cancer imaging},
  author={Tibrewala, Radhika and Dutt, Tarun and Tong, Angela and Ginocchio, Luke and Lattanzi, Riccardo and Keerthivasan, Mahesh B and Baete, Steven H and Chopra, Sumit and Lui, Yvonne W and Sodickson, Daniel K and others},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={404},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## Acknowedgements
The code for the GRAPPA technique was based off [pygrappa](https://github.com/mckib2/pygrappa), and ESPIRiT maps provided in the dataset were computed using [espirit-python](https://github.com/mikgroup/espirit-python) 
