
import os
os.chdir('/gpfs/data/brownrlab/radhika/solo-learn-main')
import sys
sys.path.append('/gpfs/data/brownrlab/radhika/solo-learn-main/')
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, alexnet
import torch.nn.functional as F
import yaml
import warnings
import types
warnings.filterwarnings("ignore")
from torchsummary import summary

def get_lr(optimizer, scheduler):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# #     return scheduler.get_last_lr()
# def forward_modified_resnet18(args, model, data, device):
#     """ Based on resnet18! Implement to add flexible drop out, remove any layers I am not interested in, primarily for overfitting """
#     layers_model = list(model.children())
#     names_layers = [x[0] for x in model.named_children()]

#     # add the dropout layers after each BLOCK in resnet
#     indexes = [5, 7, 9, 11]
#     for j, i in enumerate(indexes):
#         layers_model.insert(i,  nn.Dropout(args['model_args']['dropout_prob'][j]))
#         names_layers.insert(i, 'dropout')
#     # now remove the chunky layers if I want
#     if len(args['model_args']['removelayers']) > 0:
#         del_ind = []
#         for i in args['model_args']['removelayers']:
#             ind = np.argwhere(np.asarray(names_layers) == i)[0][0]
#             del_ind.append(ind)
#             del_ind.append(ind+1)
    
#         for index in sorted(del_ind, reverse=True):
#             del layers_model[index]
#             del names_layers[index]
    
#     for names_l, layer in zip(names_layers[:-1], layers_model[:-1]):
#         data = layer(data)
#         if 'avg' in names_l:   
#             num_features = data.shape[1]
#             data = data.view(data.size(0), -1).to(device)
    
#     layers_model[-1] = nn.Linear(in_features = num_features, out_features = 1, bias = True)
#     print('layer device')
#     print(layers_model[-1].device)    
#     out = layers_model[-1](data)
    
#     return out, layers_model, names_layers
def modify_model(model, args, device, sample_data):
    """ Based on resnet18! Implement to add flexible drop out, remove any layers I am not interested in, primarily for overfitting """

    model.eval()
    layers_model = list(model.children())
    names_layers = [x[0] for x in model.named_children()]

    # add the dropout layers after each BLOCK in resnet
    indexes = [5, 7, 9, 11]
    for j, i in enumerate(indexes):
        layers_model.insert(i,  nn.Dropout(args['model_args']['dropout_prob'][j]))
        names_layers.insert(i, 'dropout')
    # now remove the chunky layers if I want
    if len(args['model_args']['removelayers']) > 0:
        del_ind = []
        for i in args['model_args']['removelayers']:
            ind = np.argwhere(np.asarray(names_layers) == i)[0][0]
            del_ind.append(ind)
            del_ind.append(ind+1)

        for index in sorted(del_ind, reverse=True):
            del layers_model[index]
            del names_layers[index]
    for names_l, layer in zip(names_layers[:-1], layers_model[:-1]):
        sample_data = layer(sample_data)
        if 'avg' in names_l:   
            num_features = sample_data.shape[1]
            sample_data = sample_data.view(sample_data.size(0), -1).to(device)
    layers_model[-2] = nn.AdaptiveAvgPool2d(1)   
    layers_model[-1] = nn.Identity()
    new_model = nn.Sequential(*list(layers_model))

    new_model = new_model.to(device)

    return new_model

def modified_forward(self, x):

    model_len = len(list(self.children()))
    
    # if you reach the last len, you should do the operation to flatten or reshape
    
    for i in range(model_len):
        if i == model_len-1:
            x = x.view(x.size(0), -1)
        x = self[i](x)
    return x 



# def get_model(args, device):
    
#     if args['model_args']['model'] == 'alexnet':
#         model = alexnet(pretrained = args['model_args']['pretrained_imagenet'])
#         model.features[0] = nn.Conv2d(1 , 64, kernel_size=11, stride=4, padding=2, bias=False)
#         model.classifier[6] = nn.Linear(in_features = 4096, out_features = 1, bias = True)

#         if args['model_args']['freeze_first']:
#             for name, param in model.named_parameters(): 
#                 if name == 'features.0.weight':
#                     param.requires_grad = False

#     elif args['model_args']['model'] == 'resnet':  
#         model = resnet18(pretrained = args['model_args']['pretrained_imagenet'])
#         model.conv1 = nn.Conv2d(1 , 64, kernel_size=3, stride=1, padding=2, bias=False)
  

#     elif args['model_args']['model'] == 'efficientnet':    
#         model = efficientnet_b0(pretrained = args['model_args']['pretrained_imagenet'])
#         model.features[0] = nn.Conv2d(1, 32, 3, 2, 1, bias = False)
#         eff.classifier[1] = nn.Linear(1280, 1, bias = True)

#     model =  model.to(device)
#     return model

def get_optim_sched(model, args, device):
    
    if args['model_args']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = args['hyperparams']['lr'], momentum = args['hyperparams']['momentum'])
    elif args['model_args']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args['hyperparams']['lr'], weight_decay = args['hyperparams']['weight_decay'])
        
    if args['model_args']['scheduler'] == 'exponentiallr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5) # fixed gamma
    elif args['model_args']['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = args['hyperparams']['lr_decay_steps'], gamma = args['hyperparams']['ms_gamma'])
    elif args['model_args']['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args['hyperparams']['lr_decay_steps'], gamma = args['hyperparams']['gamma'])
    elif args['model_args']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 3) # fixed cycle 
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 1) # this should be equivalent to having no scheduler

    
    if args['model_args']['scheduler_plat_loss']:
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = 1, factor = 0.5, verbose = True) # fixed patience
    elif args['model_args']['scheduler_plat_auc']:
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', patience = 1, factor = 0.5,  verbose = True) # fixed patience
    else:
        scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 1) # this should be equivalent to having no scheduler

    print('Optimizer', optimizer)
    print('scheduler', scheduler)
    print('scheduler2', scheduler2)

    return optimizer, scheduler, scheduler2

