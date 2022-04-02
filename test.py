###### Training/Validation ################

import numpy as np
import os
import torch
import dill
from tqdm import tqdm
import random
from torch import nn
from tensorboardX import SummaryWriter
import monai
from monai.losses import BendingEnergyLoss,LocalNormalizedCrossCorrelationLoss
import torch.nn.functional as F
import PIL
from PIL import Image
from losses import LOG
from torch.utils.data import DataLoader
import dataloader_test
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
import monai
#from monai.networks.nets import RegUNet,GlobalNet,LocalNet,UNet
from networks import init_weights, Reg_Net, U_Net, Reg_Net_MS,Reg_Net_MS_DF,Reg_Net_MS_DF_Deformable
from monai.losses import LocalNormalizedCrossCorrelationLoss,DiceFocalLoss, BendingEnergyLoss,GlobalMutualInformationLoss
from monai.networks.layers import Norm
import test_options
from train import *
import pandas as pd
import nibabel as nib
import time
from losses import *

def JacboianDet(pred):
    d1 = int((155-120)/2)
    d2=int((240-192)/2)
    d3=int((240-192)/2)
    J = pred 
    #print("The shape of the deformation feild is:", J.shape)
    
    dy = J[:,:, 1:, :-1] - J[:,:, :-1, :-1]
    dx = J[:,:, :-1, 1:] - J[:,:, :-1, :-1]
    
    Jdet0 = dx[:,0,:,:] * (dy[:,1,:,:])
    Jdet1 = dx[:,1,:,:] * (dy[:,0,:,:])
    
    Jdet = Jdet0 - Jdet1
    #Pad_func = nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0)
    #Jdet = Pad_func(Jdet)
    
    #Jdet = Jdet.squeeze(0)
    
    #Pad_func2 = nn.ConstantPad3d((d2, 240-(d2+192), d3, 240-(d3+192),d1, 155-(d1+120)), 0)
    #Jdet2 = Pad_func2(Jdet)
    #print("Jdet is of shape:", Jdet2.shape)
    return Jdet


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.BCE = nn.BCEWithLogitsLoss()
        #self.sigmoid = nn.Sigmoid()
    def forward(self, inputs, targets):
        #print("inputs max",torch.max(inputs))
        #inputs = self.sigmoid(inputs) 
        #print(inputs.shape)
        #print(targets.shape)
        BCE_loss = self.BCE(inputs,targets)
        n_BCE_loss = -BCE_loss
        pt = torch.exp(n_BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        #print("FOCALLOSS",F_loss)
        return F_loss


def pbar_desc(label, epoch, total_epochs, loss1, loss2,loss3,loss4,loss5):
    return f'{label}:{epoch:04d}/{total_epochs} | {loss1: .3f} | {loss2: .3f} | {loss3: .3f} | {loss4: .3f} | {loss5: .3f}'


def test(mr_fixed_unet,mr_moving_unet,reg_unet, test_dl,epoch,epochs,loss1,loss2,loss3,loss4,device1,device2,device3,FLAG):
    
    reg_unet.eval()
    mr_fixed_unet.eval()
    mr_moving_unet.eval()
    avg_loss1 = []
    avg_loss2 = []
    avg_loss3 = []
    sigmoid = nn.Sigmoid()
    Att_Conv1 = torch.nn.Conv2d(1, 1,1, stride=1, padding=0).to(device2)
    Att_Conv2 = torch.nn.Conv2d(1, 1,1, stride=1, padding=0).to(device3)
    t_pbar = tqdm(test_dl,desc=pbar_desc('test', epoch , epochs, 0.0,0.0,0.0,0.0,0.0))
    BLoss = SurfaceLoss()
    with torch.no_grad():
        for us_img,mr_img,name,us_mask,mr_mask,distance_map_fixed,distance_map_moving in t_pbar:
            us_img = us_img.to(device2).squeeze(1)
            us_mask = us_mask.to(device2).squeeze(1)
            distance_map_fixed = distance_map_fixed.to(device2).squeeze(1)
            mr_img = mr_img.to(device3).squeeze(1)
            mr_mask = mr_mask.to(device3).squeeze(1)
            distance_map_moving = distance_map_moving.to(device3).squeeze(1)
        
            map_fixed = sigmoid(mr_fixed_unet(us_img))
            map_moving = sigmoid(mr_moving_unet(mr_img))
            Att_moving = Att_Conv2(torch.mul(map_moving,mr_img))
            Att_fixed = Att_Conv1(torch.mul(map_fixed,us_img))
            in_imgs_mr = torch.cat((Att_fixed.to(device1),Att_moving.to(device1)),1)
            #in_imgs_mr = torch.cat((map_fixed.to(device1),map_moving.to(device1)),1)
            displacement_field = reg_unet(in_imgs_mr)
            det_jac = JacboianDet(displacement_field)
        
            warp = monai.networks.blocks.warp.Warp(mode='bilinear')
            mr_img_before_warping = mr_img.to(device1)
            mr_mask_before_warping = mr_mask.to(device1)
            warped_img2 = warp(mr_img_before_warping, displacement_field)
            warped_mr_mask = warp(mr_mask_before_warping, displacement_field)
            BCE_loss1 = loss3#FocalLoss().to(device2)#loss3
            us_mask = us_mask#.squeeze(3)
            mr_mask = mr_mask#.squeeze(3)
            map_fixed = map_fixed
            us_mask = us_mask
            Landmarks_loss1 = BCE_loss1(map_fixed,us_mask)+BLoss(map_fixed,distance_map_fixed)
            BCE_loss2 = loss4#FocalLoss().to(device3)#loss4
            map_moving = map_moving
            mr_mask = mr_mask
            Landmarks_loss2 = BCE_loss2(map_moving,mr_mask)+BLoss(map_moving,distance_map_moving)
            LNCC = LocalNormalizedCrossCorrelationLoss(spatial_dims=2).to(device1)
            Log= LOG().to(device1)
            LOG_pred,LOG_true = Log(warped_img2,us_img.to(device1))
        
            #similarity_loss =loss1(LOG_pred,LOG_true)+LNCC(LOG_pred,LOG_true)
            us_mask = us_mask.to(device1)
            similarity_loss =loss1(warped_img2,us_img.to(device1))+loss3(us_mask,warped_mr_mask)#+LNCC(warped_img2,us_img.to(device1))
            
            smoothness_loss = loss2(displacement_field) 
            fixed_unet_loss = Landmarks_loss1
            moving_unet_loss = Landmarks_loss2
            distance_map_fixed = distance_map_fixed.to(device1) 
            reg_unet_loss =smoothness_loss + similarity_loss +BLoss(warped_mr_mask,distance_map_fixed) 
            reg_loss_display = reg_unet_loss.detach().cpu().item()
            
            warped_mr_img =warped_img2.detach().cpu().squeeze(0).squeeze(0).numpy()
            us_img_new = us_img.detach().cpu().squeeze(0).squeeze(0).numpy()
            mr_img_new = mr_img.detach().cpu().squeeze(0).squeeze(0).numpy()
            df = displacement_field.detach().cpu().squeeze(0).squeeze(0).numpy()
            map_fixed_out = map_fixed.detach().cpu().squeeze(0).squeeze(0).numpy()
            map_moving_out = map_moving.detach().cpu().squeeze(0).squeeze(0).numpy()
            mr_mask_out = mr_mask.detach().cpu().squeeze(0).squeeze(0).numpy()
            us_mask_out = us_mask.detach().cpu().squeeze(0).squeeze(0).numpy()
            Att_moving_out = Att_moving.detach().cpu().squeeze(0).squeeze(0).numpy()
            Att_fixed_out = Att_fixed.detach().cpu().squeeze(0).squeeze(0).numpy()
            warped_mr_mask_out = warped_mr_mask.detach().cpu().squeeze(0).squeeze(0).numpy() 
            det_jac = det_jac.detach().cpu().numpy()
            
            n1,_= name[0].split(sep='.') 
            name = n1
            np.save('./Result_test/us_'+name+'.npy',us_img_new)
            np.save('./Result_test/mr_'+name+'.npy',mr_img_new)
            np.save('./Result_test/warped_mr_img_'+name+'.npy',warped_mr_img)
            np.save('./Result_test/df_'+name+'.npy',df)
            np.save('./Result_test/map_fixed_'+name+'.npy',map_fixed_out)
            np.save('./Result_test/map_moving_'+name+'.npy',map_moving_out)
            np.save('./Result_test/mask_moving_'+name+'.npy',mr_mask_out)
            np.save('./Result_test/mask_fixed_'+name+'.npy',us_mask_out)
            np.save('./Result_test/Att_moving_'+name+'.npy',Att_moving_out)
            np.save('./Result_test/Att_fixed_'+name+'.npy',Att_fixed_out)
            np.save('./Result_test/warped_mask_'+name+'.npy',warped_mr_mask_out)
            
            
            
        avg_loss1 = torch.mean(torch.tensor(avg_loss1))
        avg_loss2 = torch.mean(torch.tensor(avg_loss2))
        avg_loss3 = torch.mean(torch.tensor(avg_loss3))

        
        avg_loss1 = torch.mean(torch.tensor(avg_loss1))
        avg_loss2 = torch.mean(torch.tensor(avg_loss2))
        avg_loss3 = torch.mean(torch.tensor(avg_loss3))
           
        
    




 








def main(args):
    print("Welcome to Testing Phase")
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    #args = test_options.parse_arguments()
    transforms = T.Compose([T.ToTensor()])#,T.CenterCrop(256)])
    test_ds = dataloader_test.MRI_Dataset(args.TEST_US,args.TEST_MR,args.TEST_US_MASK,args.TEST_MR_MASK,args.NORM,args.FLAG,transforms)
    test_dl = DataLoader(test_ds, args.TEST_BATCH_SIZE, shuffle = True, num_workers = args.WORKERS)
    
    start_epoch = 1
    reg_unet = Reg_Net_MS_DF_Deformable(2,2,8)
    mr_fixed_unet=  U_Net(1,1,8)
    mr_moving_unet =   U_Net(1,1,8)
    
    
    #print(reg_unet)
    print('Fixed-U-Net parameters:', sum(p.numel() for p in mr_fixed_unet.parameters()))
    print('Moving-U-Net parameters:', sum(p.numel() for p in mr_moving_unet.parameters()))
    print('REG-U-Net parameters:', sum(p.numel() for p in reg_unet.parameters()))

    loss1 = GlobalMutualInformationLoss().to(args.DEVICE1)
    loss2 = BendingEnergyLoss().to(args.DEVICE1)
    loss3 = Dice()#DiceFocalLoss ().to(args.DEVICE2)#torch.nn.LogSoftmax().to(args.DEVICE2)
    loss4 =  Dice()#DiceFocalLoss ().to(args.DEVICE3)#torch.nn.LogSoftmax().to(args.DEVICE3)

    if args.LOAD_CHECKPOINT1 is not None:
        checkpoint1 = torch.load(args.LOAD_CHECKPOINT1, pickle_module = dill)
        start_epoch1 = checkpoint1['epoch']
        mr_fixed_unet.load_state_dict(checkpoint1['mr_fixed_unet_state_dict'])
        opt1 = checkpoint1['optimizer']
        sched1 = checkpoint1['lr_scheduler']
    
    if args.LOAD_CHECKPOINT2 is not None:
        checkpoint2 = torch.load(args.LOAD_CHECKPOINT2, pickle_module = dill)
        start_epoch2 = checkpoint2['epoch']
        mr_moving_unet.load_state_dict(checkpoint2['mr_moving_unet_state_dict'])
        opt2 = checkpoint2['optimizer']
        sched2 = checkpoint2['lr_scheduler']
    if args.LOAD_CHECKPOINT3 is not None:
        checkpoint3 = torch.load(args.LOAD_CHECKPOINT3, pickle_module = dill)
        start_epoch3 = checkpoint3['epoch']
        reg_unet.load_state_dict(checkpoint3['reg_unet_state_dict'])
        opt3 = checkpoint3['optimizer']
        sched3 = checkpoint3['lr_scheduler']
    mr_fixed_unet.to(args.DEVICE2)
    mr_moving_unet.to(args.DEVICE3)
    reg_unet.to(args.DEVICE1)
    
   
    start_time = time.time()
    counter = 0.0
    for epoch in range(start_epoch, args.EPOCHS+1):
        counter += 1.0
        ## testing loop
        test(mr_fixed_unet,mr_moving_unet,reg_unet, test_dl,epoch,args.EPOCHS,loss1,loss2,loss3,loss4,args.DEVICE1,args.DEVICE2,args.DEVICE3,args.FLAG)
       
    Time = time.time()-start_time
    print('The total is:', Time)
    avg_Time = Time/counter
    print('Average running time is', avg_Time)       
if __name__=='__main__':
    args = test_options.parse_arguments()
    main(args)

