## fastMRI Prostate: Deep Learning Based Classification 

This repository contains code for deep learning based prediction of clinically significant prostate cancer built for fastMRI prostate data. We train two separate models for T2 and diffusion-weighted images.

## Features

- **Data Compatibility**: Designed to directly use the label files and filenames for the fastMRI prostate dataset
- **Built-in image processing**: augmentations and image processing for data normalization are supported for both T2 and diffusion weighted images
- **Model Output**: test.py generates an ROC curve for predictions on the test set for the T2 and diffusion weighted images

## Installation

The code requires `python >= 3.9` and uses [torch](https://pytorch.org/docs/stable/torch.html)

## Usage
#### Training: To run the classification model on T2 images:

```bash
 python -u train_t2.py \ --config_file configs/t2_final.yaml \ --index_seed [seed]
```

Replace `[seed]` with a seed configuration. Set hyperparameters and path to your fastMRI prostate data ("data_location" field) in the [config file](https://github.com/cai2r/fastMRI_prostate/blob/classification_code_review/fastmri_prostate_classification/configs/t2_final.yaml)

#### Training: To run the classification model on Diffusion images:

```bash
 python -u train_dwi.py \ --config_file configs/diffusion_final.yaml \ --index_seed [seed]
```
Replace `[seed]` with a seed configuration number between 1-10. Set hyperparameters and path to your fastMRI prostate data ("data_location" field) in the [config file](https://github.com/cai2r/fastMRI_prostate/blob/classification_code_review/fastmri_prostate_classification/configs/diffusion_final.yaml)

#### Testing: To test the classification model on T2 and Diffusion images:

```bash
 python -u test.py \ --config_file_t2 configs/diffusion_final.yaml  \ --config_file_diff configs/diffusion_final.yaml \ --index_seed [seed]
```
Replace `[seed]` with a seed configuration number between 1-10. Set the selected model instance in the [T2](https://github.com/cai2r/fastMRI_prostate/blob/classification_code_review/fastmri_prostate_classification/configs/diffusion_final.yaml) and [Diffusion](https://github.com/cai2r/fastMRI_prostate/blob/classification_code_review/fastmri_prostate_classification/configs/diffusion_final.yaml) config files in the field "load_model_epoch".

## Contributing

Contributions to improve this project are welcome. Please consider submitting a pull request or opening an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

