import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights, convnext_tiny, ConvNeXt_Tiny_Weights

def ConvNext_model(args, diff=False):
    """
    Load ConvNext model.

    Parameters:
    - args (dict): Dictionary containing model arguments, where 'model' specifies the model type.
                    It should have a key 'model_args' containing another dictionary with a key 'model'.
                    Accepted values for 'model' are "convnext" and "tiny".
    - diff (bool, optional): If True, apply changes to the model to have two channels for diffusion. Default is False.

    Returns:
    - model: Loaded ConvNext model based on the specified arguments.

    Raises:
    - ValueError: If the specified model in args is not "convnext" or "tiny".

    Note:
    - The model architecture is modified based on the specified model type and any differential changes.
    """
    if args['model_args']['model'] == "convnext":
        weights_cn = ConvNeXt_Base_Weights.DEFAULT                                  
        model = convnext_base(weights=weights_cn)                                  
        model.features[0][0] = nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))   
        if diff:
            model.features[0][0] = nn.Conv2d(2, 128, kernel_size=(4, 4), stride=(4, 4))   

        model.classifier[2] = nn.Linear(in_features=1024, out_features=1, bias=True)  

    elif args['model_args']['model'] == "tiny":
        weights_cn = ConvNeXt_Tiny_Weights.DEFAULT                                 
        model = convnext_tiny(weights=weights_cn)                                   
        model.features[0][0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))   
        model.classifier[2] = nn.Linear(in_features=768, out_features=1, bias=True)  
    else:
        raise ValueError("Wrong model selection. Accepted values are 'convnext' or 'tiny'.")

    return model