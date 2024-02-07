import argparse
import numpy as np
import os
import torch
from sklearn import metrics
from utils.load_fastmri_data_convnext_diff import load_data as load_data_diff
from utils.load_fastmri_data_convnext_t2 import load_data as load_data_t2

from model.model import ConvNext_model
import yaml
import matplotlib.pyplot as plt

def test(model, test_loader, device):
    """
    Test the ConvNext model on the test set.

    Parameters:
    - model: The ConvNext model.
    - test_loader: The PyTorch DataLoader for the test set.
    - device: The device (CPU or GPU) on which to perform the testing.

    Returns:
    - auc_test (float): The area under the ROC curve on the test set.
    - raw_preds_test (Tensor): Concatenated raw predictions from the test set.
    """
    total_num_test, all_out, all_labels_test = 0,  [], [] 
    model.eval()                                                             
    with torch.no_grad():                                                    
        for _, (data, target) in enumerate(test_loader):
            data, target = data.to(device), torch.flatten(target.to(device))   
            out = model(data)                                                 

            out = torch.flatten(out)                                          
            
            all_out.append(out)                       
            all_labels_test.append(target)               
            total_num_test += 1                         

    all_labels_npy = torch.cat(all_labels_test).detach().cpu().numpy().astype(np.int32) 
    all_preds_npy = torch.sigmoid(torch.cat(all_out)).detach().cpu().numpy()           

    auc_test = metrics.roc_auc_score(all_labels_npy, all_preds_npy)                    
        
    return auc_test, all_preds_npy, all_labels_npy

def run_inference(config_diff, config_t2):
    """
    Run an inference session on the trained ConvNext network

    Parameters:
    - config (dict): Configuration parameters for inference on the trained ConvNext model (same file used during training).
    """
    use_cuda = torch.cuda.is_available()                               
    device = torch.device("cuda" if use_cuda else "cpu")                
    print('Found this device:{}'.format(device))
    
    _, _, test_loader_diff = load_data_diff(config_diff['data']['datasheet'],  config_diff["data"]["data_location"], int(config_diff['data']['norm_type']),  config_diff['training']['augment'], config_diff['training']['saveims'], config_diff['model_args']['rundir'])
    _, _, test_loader_t2 = load_data_t2(config_t2['data']['datasheet'], config_t2["data"]["data_location"],  int(config_t2['data']['norm_type']),  config_t2['training']['augment'], config_t2['training']['saveims'], config_t2['model_args']['rundir'])

    print('Lengths diffusion:Test:{}'.format(len(test_loader_diff)))  
    print('Lengths T2:Test:{}'.format(len(test_loader_t2)))  


    model_diff = ConvNext_model(config_diff, diff = True)
    
    model_path_diff = os.path.join(config_diff['model_args']['rundir'], "model_epoch_" + str(config_diff['load_model_epoch']) +'.pth')
    print("Loading model:{}".format(model_path_diff))
    model_diff.load_state_dict(torch.load(model_path_diff))
    model_diff.to(device)

    model_t2 = ConvNext_model(config_t2)
    model_path_t2 = os.path.join(config_t2['model_args']['rundir'], "model_epoch_" + str(config_t2['load_model_epoch']) +'.pth')
    print("Loading model:{}".format(model_path_t2))
    model_t2.load_state_dict(torch.load(model_path_t2))
    model_t2.to(device)


    AUC_test_diff, raw_preds_test_diff, labels_diff  = test(model_diff, test_loader_diff, device)    
    AUC_test_t2, raw_preds_test_t2, labels_t2  = test(model_t2, test_loader_t2, device)    
    print("Test AUC - T2 is:{:.3f}".format(AUC_test_t2))
    print("Test AUC - diffusion is:{:.3f}".format(AUC_test_diff))
    
    fpr_diff, tpr_diff, _ = metrics.roc_curve(labels_diff, raw_preds_test_diff)
    fpr_t2, tpr_t2, _ = metrics.roc_curve(labels_t2, raw_preds_test_t2)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_diff, tpr_diff, 'b', label = 'AUC diff = %0.2f' % AUC_test_diff, c= 'red')
    plt.plot(fpr_t2, tpr_t2, 'b', label = 'AUC T2= %0.2f' % AUC_test_t2, c= 'blue')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1.02])
    plt.ylim([0, 1.02])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    save_png_file = "Test_ROC_Curve_fastMRI_prostate.png"
    plt.savefig(save_png_file, bbox_inches = "tight")



def get_parser():
    """
    Create an argument parser for the main script.

    Returns:
    - parser: The argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_t2', type=str, required=True)           # config file which has all the training inputs
    parser.add_argument('--config_file_diff', type=str, required=True)         # config file which has all the training inputs
    parser.add_argument('--index_seed', type=int, required=True)            # Seed number for reproducibility for all numpy, random, torch

    return parser


if __name__ == '__main__':
    """
    Main script for training the ConvNext model.
    """
    args_con = get_parser().parse_args() 

    seed_list = [10383, 44820, 238, 3939, 74783, 92938, 143, 2992, 7373, 988]           
    seed_select =  seed_list[args_con.index_seed]                                      
    
    with open(args_con.config_file_t2) as f:
        args_t2 = yaml.load(f, Loader=yaml.UnsafeLoader)  

    with open(args_con.config_file_diff) as f:
        args_diff = yaml.load(f, Loader=yaml.UnsafeLoader)  
    
    main_fol_t2 = args_t2["results_fol"]
    main_fol_dwi = args_diff["results_fol"]

    args_t2['model_args']['rundir'] = os.path.join(main_fol_t2, args_t2['model_args']['rundir'] + '_SEED_' + str(seed_select)) 
    args_diff['model_args']['rundir'] = os.path.join(main_fol_dwi, args_diff['model_args']['rundir'] + '_SEED_' + str(seed_select)) 

    print("Model rundir T2:{}".format(args_t2['model_args']['rundir']))   
    print("Model rundir diffusion:{}".format(args_diff['model_args']['rundir']))   

    torch.manual_seed(seed_select)                                           
    torch.cuda.manual_seed(seed_select)                               
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_select)

    run_inference(args_diff, args_t2)

# %%
