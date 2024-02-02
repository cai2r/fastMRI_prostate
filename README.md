# FastMRI Prostate

[[`Paper`](https://arxiv.org/abs/2304.09254)] [[`Dataset`](https://fastmri.med.nyu.edu/)] [[`Github`](https://github.com/cai2r/fastMRI_prostate)] [[`BibTeX`](#cite)]

### Updates
02-02-2024: Updated [files](https://github.com/cai2r/fastMRI_prostate/pull/5) for slice-, volume-, exam-level labels for T2 and Diffusion sequences in the [fastMRI prostate dataset](https://fastmri.med.nyu.edu/).

## Overview

This repository contains code to facilitate the reconstruction of prostate T2 and DWI (Diffusion-Weighted Imaging) images from raw data in the fastMRI Prostate dataset. It includes reconstruction methods along with utilities for pre-processing and post-processing the data. 

The package is intended to serve as a starting point for those who want to experiment and develop alternate reconstruction techniques.

## Installation

The code requires `python >= 3.9`

Install FastMRI Prostate: clone the repository locally and install with

```
git clone git@github.com:tarund1996/fastmri_prostate_test.git
cd fastmri_prostate_test
pip install -e .
```

## Usage
The repository is centered around the ```fastmri_prostate``` package. The following breaks down the basic structure:

```fastmri_prostate```: Contains a number of basic tools for T2 and DWI reconstruction
 - ```fastmri_prostate.data```: Provides data utility functions for accessing raw data fields like kspace, calibration, phase correction, and coil sensitivity maps.
 - ```fastmri.reconstruction.t2```: Contains functions required for prostate T2 reconstruction
 - ```fastmri.reconstruction.dwi```: Contains functions required for prostate DWI reconstruction

```fastmri_prostate_recon.py``` contains code to read files from the dataset and call the T2 and DWI reconstruction functions for a single h5 file. 

```fastmri_prostate_tutorial.ipynb``` walks through an example of loading a raw h5 file from the fastMRI prostate dataset and displaying the reconstructions

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
@misc{tibrewala2023fastmri,
      title={FastMRI Prostate: A Publicly Available, Biparametric MRI Dataset to Advance Machine Learning for Prostate Cancer Imaging}, 
      author={Radhika Tibrewala and Tarun Dutt and Angela Tong and Luke Ginocchio and Mahesh B Keerthivasan and Steven H Baete and Sumit Chopra and Yvonne W Lui and Daniel K Sodickson and Hersh Chandarana and Patricia M Johnson},
      year={2023},
      eprint={2304.09254},
      archivePrefix={arXiv},
      primaryClass={physics.med-ph}
}
```

## Acknowedgements
The code for the GRAPPA technique was based off [pygrappa](https://github.com/mckib2/pygrappa), and ESPIRiT maps provided in the dataset were computed using [espirit-python](https://github.com/mikgroup/espirit-python) 
