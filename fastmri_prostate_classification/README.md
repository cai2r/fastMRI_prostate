## Code for deep learning based classification for predicting clinically significant prostate cancer - fastMRI prostate 

## Installation

The code requires `python >= 3.9` and uses [torch](https://pytorch.org/docs/stable/torch.html)

### To run the classification model on T2 images:

1. Set hyperparameters and path to your fastMRI prostate data ("data_location" field) in the [config file](https://github.com/cai2r/fastMRI_prostate/blob/classification_code_review/fastmri_prostate_classification/configs/t2_final.yaml)
2. To run the classification script:
``` python -u train_t2.py \ --config_file configs/t2_final.yaml \ --index_seed 123 ```

### To run the classification model on Diffusion images:

1. Set hyperparameters and path to your fastMRI prostate data ("data_location" field) in the [config file](https://github.com/cai2r/fastMRI_prostate/blob/classification_code_review/fastmri_prostate_classification/configs/diffusion_final.yaml)
2. To run the classification script:
``` python -u train_dwi.py \ --config_file configs/diffusion_final.yaml \ --index_seed 123 ```

