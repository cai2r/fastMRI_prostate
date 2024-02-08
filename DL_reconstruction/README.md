# Prostate Diffusion MRI Reconstruction

This repository contains code for reconstruction accelerated prostate diffusion MRI, derived from the [fastMRI](https://github.com/facebookresearch/fastMRI) repository and modified specifically for the fastmri prostate diffusion data. Our model supports reconstruction for b50 and b1000 diffusion images, outputting b50 or b1000 trace images accordingly.

## Features

- **Data Compatibility**: Designed for reconstruction of prostate diffusion MRI data, specifically compatible with the fastMRI prostate dataset.
- **Model Output**: Generates b50 or b1000 trace images
- **Based on fastMRI**: Leverages the framework and model from the fastMRI repository.

## Usage

To run the model, you can use the following commands for training and testing:

### Training

```bash
python train_varnet_prostate.py --mode train --data_path [path_to_data] --bvalue b50 --test_path [path_to_test_data] --state_dict_file [path_to_checkpoint_for_testing] --batch_size 1 --num_workers 4
```

Replace `[script_name]` with the name of your script, `[path_to_data]` with the path to your training data, `[path_to_test_data]` with the path to your testing data, and `[path_to_checkpoint_for_testing]` with the path to a model checkpoint file if you have one for testing.

### Testing

```bash
python train_varnet_prostate.py --mode test --data_path [path_to_data] --bvalue b1000 --test_path [path_to_test_data] --state_dict_file [path_to_checkpoint] --batch_size 1 --num_workers 4
```

Ensure you specify the `--state_dict_file` argument during testing to provide the path to your model checkpoint.

Note: For deep learning recon of the T2-weighted data please refer to the [fastMRI VarNet](https://github.com/facebookresearch/fastMRI/tree/main/fastmri_examples/varnet) implementation.
## Configuration

The script accepts various command-line arguments to customize the data paths, model parameters, and training settings. Refer to the source code for a complete list of available options.

## Contributing

Contributions to improve this project are welcome. Please consider submitting a pull request or opening an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

