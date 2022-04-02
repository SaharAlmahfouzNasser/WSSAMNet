import torch
from torch.utils.data import Dataset
import os 
import numpy as np
import nibabel as nib
from torchvision import transforms as T
import random
import torch.nn.functional as F
import nrrd
import pandas as pd
from torchio import Subject, ScalarImage, RandomAffine
import scipy

def NORM(img):
    img_out = np.zeros(img.shape)
    Max = np.max(img)
    Min = np.min(img)
    out = (img-Min)/(Max-Min+1e-10)
    img_out=out
    return img_out
    
def GNORM(img):
    img_out = np.zeros(img.shape)
    Mean = np.mean(img)
    Std = np.std(img)
    out = (img-Mean)/(Std+1e-10)
    img_out=out
    return img_out
class MRI_Dataset(Dataset):
    def __init__(self, path_us,path_mr,path_us_mask,path_mr_mask,NORM,FLAG,transforms = None):
        super().__init__()
        self.path_us = path_us
        self.path_mr = path_mr
        self.path_us_mask = path_us_mask
        self.path_mr_mask = path_mr_mask
        self.data_list_us = sorted(os.listdir(path_us))
        self.data_list_mr = sorted(os.listdir(path_mr))
        self.data_list_us_mask = sorted(os.listdir(path_us_mask))
        self.data_list_mr_mask = sorted(os.listdir(path_mr_mask))
        self.transforms = transforms
        self.NORM = NORM
        self.FLAG = FLAG
    def __len__(self):
        return len(self.data_list_us)

    def __getitem__(self,item):
       
        name = self.data_list_us[item]
        us_img = np.load(self.path_us+self.data_list_us[item]).astype('float64')
        mr_img = np.load(self.path_mr+self.data_list_mr[item]).astype('float64')
        us_mask = np.load(self.path_us_mask+self.data_list_us_mask[item]).astype('float64')
        mr_mask = np.load(self.path_mr_mask+self.data_list_mr_mask[item]).astype('float64')
        
        inverse_mr_mask = 1-mr_mask
        inv_mr_distance_map = scipy.ndimage.distance_transform_edt(inverse_mr_mask)
        mr_distance_map = scipy.ndimage.distance_transform_edt(mr_mask)
        Distance_map_mr = inv_mr_distance_map-mr_distance_map+1 
        
        inverse_us_mask = 1-us_mask
        inv_us_distance_map = scipy.ndimage.distance_transform_edt(inverse_us_mask)
        us_distance_map = scipy.ndimage.distance_transform_edt(us_mask)
        Distance_map_us = inv_us_distance_map-us_distance_map+1
        
        if self.NORM:
            us_img = NORM(us_img)
            mr_img = NORM(mr_img)
           
        
        if self.transforms is not None:
            us_img = self.transforms(us_img).type(torch.FloatTensor).unsqueeze(0)
            mr_img = self.transforms(mr_img).type(torch.FloatTensor).unsqueeze(0)
            Distance_map_mr= self.transforms(Distance_map_mr).type(torch.FloatTensor).unsqueeze(0)
            us_mask = self.transforms(us_mask).type(torch.LongTensor).unsqueeze(0)
            mr_mask = self.transforms(mr_mask).type(torch.LongTensor).unsqueeze(0)
            Distance_map_us= self.transforms(Distance_map_us).type(torch.FloatTensor).unsqueeze(0)
            
            s1=1.0
            s2=1.0
            
            
            d1=0.0
            d2=0.0
           
            
            t1=0.0
            t2=0.0
            
            
            subject1 = Subject(
            us_img=ScalarImage(tensor=us_img),  # this class is new
            us_mask=ScalarImage(tensor=us_mask),)
            
            subject2 = Subject(
            mr_img=ScalarImage(tensor=mr_img),  # this class is new
            mr_mask=ScalarImage(tensor=mr_mask),)
            
            transform_new1 = RandomAffine(scales=(s1,s2),degrees=(d1,d2),translation=(t1,t2))
            transformed1 = transform_new1(subject1)
            
            transform_new2 = RandomAffine(scales=(s1,s2),degrees=(d1,d2),translation=(t1,t2))
            transformed2 = transform_new2(subject2)
            
            us_img= torch.tensor(transformed1.us_img.numpy())
            us_mask = torch.tensor(transformed1.us_mask.numpy())
            
            mr_img = torch.tensor(transformed2.mr_img.numpy())
            mr_mask = torch.tensor(transformed2.mr_mask.numpy())
           
            
        return us_img,mr_img,name,us_mask,mr_mask,Distance_map_us,Distance_map_mr 
      

