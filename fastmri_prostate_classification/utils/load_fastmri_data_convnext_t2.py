# import modules
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from utils.augmentation_slice_level import augment_image_t2
from utils.image_utils import normalisation_2d, center_crop_2d
import h5py

class Dataset(data.Dataset):
    """
    Custom PyTorch dataset for loading FastMRI prostate data at the slice level.

    Parameters:
    - datasheet (str): Path to the datasheet containing information about the data.
    - norm_type (int): Integer representing the chosen normalization scheme (1-4).
    - crop_type: Type of cropping (not explicitly used in the provided code).
    - augment (bool): Flag indicating whether to apply data augmentation.
    - saveims (bool): Flag indicating whether to save PNG images during training.
    - rundir (str): Directory where model logs, outputs, etc., will be stored.
    - istrain (bool): Flag indicating whether the dataset is for training.
    - isval (bool): Flag indicating whether the dataset is for validation.
    - istest (bool): Flag indicating whether the dataset is for testing.
    """

    def __init__(self, datasheet, datapath, norm_type, augment, saveims, rundir, istrain, isval, istest):
        super().__init__()
        self.paths_t2 = []            
        self.labels = []              
        self.istrain = istrain         
        self.isval = isval            
        self.istest = istest        
        self.nums = []               
        self.aug = int(augment)     
        self.rundir = rundir          
        self.saveims = saveims        
        self.norm_type = norm_type   
        self.slice_num = []
        self.datapath = datapath
        data = pd.read_csv(datasheet) 
        
        if istrain:
            data = data[data['data_split'] == 'training'].reset_index(drop = True)  
        elif isval:
            data = data[data['data_split'] == 'validation'].reset_index(drop = True) 
            self.aug = 0
        else:
            data = data[data['data_split'] == 'test'].reset_index(drop = True) 
            self.aug = 0

        for i in range(0,len(data)):           
            pt_id = data['fastmri_pt_id'].iloc[i]  
            file_t2 = data['fastmri_rawfile'].iloc[i]  
            fol_t2 =  os.path.join(data['folder'].iloc[i])
            path_t2 = os.path.join(datapath, fol_t2, file_t2)
            self.paths_t2.append(path_t2)                          
            label_PIRADS = data['PIRADS'].iloc[i]                   
            label = (label_PIRADS > 2).astype(np.int32)            
            self.labels.append(int(label))                         
            num = data['fastmri_pt_id'].iloc[i]                    
            self.nums.append(int(num))                             
            slice_num = data['slice'].iloc[i] - 1 # slice numbers are for DICOMs, make pythonic                     
            self.slice_num.append(int(slice_num))                  

        self.labels = np.asarray(self.labels)                       
        neg_weight = np.mean(self.labels)                          
        self.weights = [neg_weight, 1 - neg_weight]                
        
        print("Weights for binary CE:{}".format(self.weights))     
        print("Number of paths:{}, Number of labels:{}".format(len(self.paths_t2), len(self.labels))) 
    #  https://doi.org/10.1371/journal.pmed.1002699.s001 
    def weighted_loss(self, prediction, target):
        """
        Compute the weighted cross-entropy loss.

        Parameters:
        - prediction (Tensor): Model predictions.
        - target (Tensor): Ground truth labels.

        Returns:
        - loss (Tensor): Weighted cross-entropy loss.
        """
        weights_npy = np.array([self.weights[int(t)] for t in target.data])    
        weights_tensor = torch.FloatTensor(weights_npy).cuda()                 
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor)) 
        return loss

    def __getitem__(self, index):

        load_path_t2 = self.paths_t2[index]    
        with h5py.File(load_path_t2, "r") as hf:
            im_recon_320 = hf["reconstruction_rss"][:]   
        im_recon_320 = im_recon_320[self.slice_num[index],:,:]
        
        if random.randint(0, 100) > 30 and self.aug: 
            im_recon_320, op_list = augment_image_t2(im_recon_320)
        else:
            op_list = ['none']

        im_recon_224 = center_crop_2d(im_recon_320, (224,224))
        im_normalised = normalisation_2d(im_recon_224, self.norm_type)
        
        #! remove  save image PNG after normalisation in train_dir so we can glance through them and make sure they look ok
        sl = self.slice_num[index] 
        
        slice_tensor = torch.FloatTensor(im_normalised)                      
        slice_tensor = torch.unsqueeze(slice_tensor, dim = 0)               

        label_tensor = torch.FloatTensor([self.labels[index]])         

        return slice_tensor, label_tensor

    def __len__(self):
        return len(self.paths_t2)

def load_data(datasheet, datapath, norm_type, augment, saveims, rundir):
    """
    Load FastMRI prostate data and create DataLoader instances for training, validation, and testing.

    Parameters:
    - datasheet (str): Path to the datasheet containing information about the data.
    - norm_type (int): Integer representing the chosen normalization scheme (1-4).
    - crop_type: Type of cropping (not explicitly used in the provided code).
    - augment (bool): Flag indicating whether to apply data augmentation.
    - saveims (bool): Flag indicating whether to save PNG images during training.
    - rundir (str): Directory where model logs, outputs, etc., will be stored.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set.
    - valid_loader (DataLoader): DataLoader for the validation set.
    - test_loader (DataLoader): DataLoader for the test set.
    """
    train_dataset = Dataset(datasheet, datapath, norm_type, augment, saveims, rundir, istrain = True, isval = False, istest = False)
    valid_dataset = Dataset(datasheet, datapath, norm_type, augment, saveims, rundir, istrain = False, isval = True, istest = False)
    test_dataset  = Dataset(datasheet, datapath, norm_type, augment, saveims, rundir, istrain = False, isval = False, istest = True)

    # https://pytorch-lightning.readthedocs.io/en/stable/guides/speed.html#:~:text=A%20general%20place%20to%20start,you%20may%20overflow%20RAM%20memory.
    train_loader = data.DataLoader(train_dataset, batch_size=32, num_workers=1, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=32, num_workers=1, shuffle=False)
    test_loader =  data.DataLoader(test_dataset,  batch_size=32, num_workers=1, shuffle=False)
    
    return train_loader, valid_loader, test_loader


# %%
