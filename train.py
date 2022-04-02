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
from losses import LOG,SurfaceLoss
import nibabel as nib
from torch.cuda.amp import autocast, GradScaler
import pdb
import torch
from torch import nn



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

def train(mr_fixed_unet,mr_moving_unet,reg_unet, trn_dl,epoch,epochs,loss1,loss2,loss3,loss4,opt1,opt2,opt3,train_losses,device1,device2,device3,TENSORBOARD_LOGDIR,LOSS_WEIGHT,SAVE_EVERY,WEIGHTS_SAVE_PATH,EXP_NO,FLAG):
    
    mr_fixed_unet.train()
    mr_moving_unet.train()
    reg_unet.train()
    t_pbar = tqdm(trn_dl, desc=pbar_desc('train',epoch,epochs,0.0,0.0,0.0,0.0,0.0))
    
    avg_loss = []
    RELU = torch.nn.ReLU(inplace=False)
    Att_Conv1 = torch.nn.Conv2d(1, 1,1, stride=1, padding=0).to(device2)
    Att_Conv2 = torch.nn.Conv2d(1, 1,1, stride=1, padding=0).to(device3)
    sigmoid = nn.Sigmoid()
    BLoss = SurfaceLoss()
    for us_img,mr_img,name,us_mask,mr_mask,distance_map_fixed,distance_map_moving in t_pbar:
       
        #with autocast():
        us_img = us_img.to(device2).squeeze(1)
        us_mask = us_mask.to(device2).squeeze(1)
        distance_map_fixed = distance_map_fixed.to(device2).squeeze(1)
        mr_img = mr_img.to(device3).squeeze(1)
        mr_mask = mr_mask.to(device3).squeeze(1)
        distance_map_moving = distance_map_moving.to(device3).squeeze(1)
        #print(torch.min(mr_fixed_unet(us_img)),torch.max(mr_fixed_unet(us_img)))
        
        map_fixed = sigmoid(mr_fixed_unet(us_img))
        map_moving = sigmoid(mr_moving_unet(mr_img))
        Att_moving = Att_Conv2(torch.mul(map_moving,mr_img))
        Att_fixed = Att_Conv1(torch.mul(map_fixed,us_img))
        in_imgs_mr = torch.cat((Att_fixed.to(device1),Att_moving.to(device1)),1)
       
        #in_imgs_mr = torch.cat((map_fixed.to(device1),map_moving.to(device1)),1)
        displacement_field = reg_unet(in_imgs_mr)
        
        
        warp = monai.networks.blocks.warp.Warp(mode='bilinear')
        mr_img_before_warping = mr_img.to(device1)
        mr_mask_before_warping = mr_mask.to(device1)
        warped_img2 = warp(mr_img_before_warping, displacement_field)
        warped_mr_mask = warp(mr_mask_before_warping, displacement_field)
        
        
        BCE_loss1 =loss3#FocalLoss().to(device2)#loss3
        us_mask = us_mask#.squeeze(3)
        mr_mask = mr_mask#.squeeze(3)
        map_fixed = map_fixed
        us_mask = us_mask
        Landmarks_loss1 = BCE_loss1(map_fixed,us_mask)+BLoss(map_fixed,distance_map_fixed)
        BCE_loss2 =loss4 #FocalLoss().to(device3)#loss4
        map_moving = map_moving
        mr_mask = mr_mask
        Landmarks_loss2 = BCE_loss2(map_moving,mr_mask)+BLoss(map_moving,distance_map_moving)
        LNCC = LocalNormalizedCrossCorrelationLoss(spatial_dims=2).to(device1)
        Log = LOG().to(device1)
        LOG_pred,LOG_true = Log(warped_img2,us_img.to(device1))
        
        #similarity_loss =loss1(LOG_pred,LOG_true)+LNCC(LOG_pred,LOG_true)
        #similarity_loss =loss1(warped_img2,us_img.to(device1))+LNCC(warped_img2,us_img.to(device1))
        us_mask = us_mask.to(device1)
        similarity_loss =loss1(warped_img2,us_img.to(device1))+loss3(us_mask,warped_mr_mask)#+LNCC(warped_img2,us_img.to(device1))
        smoothness_loss = loss2(displacement_field) 
        fixed_unet_loss = Landmarks_loss1
        
        moving_unet_loss = Landmarks_loss2 
        reg_unet_loss =smoothness_loss  +BLoss(warped_mr_mask,distance_map_fixed.to(device1))#+similarity_loss 
        reg_loss_display = reg_unet_loss.detach().cpu().item()
        #print('warped_img2',torch.min(warped_img2),torch.max(warped_img2))
        #print('displacement_field',torch.min(displacement_field),torch.max(displacement_field))
        #print('map_fixed',torch.min(map_fixed),torch.max(map_fixed))
        #print('us_mask',torch.min(us_mask),torch.max(us_mask))
        #print('map_moving',torch.min(map_moving),torch.max(map_moving))
        #print('mr_mask',torch.min(mr_mask),torch.max(mr_mask))
        #print(fixed_unet_loss)
        #print(moving_unet_loss)
        #print(similarity_loss)
        #print(smoothness_loss)
        opt3.zero_grad()
        opt1.zero_grad()
        opt2.zero_grad()
        #scaler.scale(reg_unet_loss).backward(retain_graph=True)
        #scaler.step(opt3)
        reg_unet_loss.backward(retain_graph=True)
        opt3.step()
            
        #scaler.scale(fixed_unet_loss).backward(retain_graph=True)
        #scaler.step(opt1)
        
        fixed_unet_loss.backward(retain_graph=True)
        opt1.step()
            
        #scaler.scale(moving_unet_loss).backward(retain_graph=True)
        #scaler.step(opt2)
        moving_unet_loss.backward(retain_graph=True)
        opt2.step()
        
        #scaler.update()    
        

        

        t_pbar.set_description(pbar_desc('train',epoch,epochs,similarity_loss.item(),smoothness_loss.item(),reg_unet_loss.item(),fixed_unet_loss.item(),moving_unet_loss.item()))

        train_losses.update(loss1 = similarity_loss.item(), loss2 = smoothness_loss.item(),loss3=reg_unet_loss.item(),loss4 = fixed_unet_loss.item(),loss5 = moving_unet_loss.item() )
    
    avg_loss.append(reg_unet_loss.item())

def evaluate(mr_fixed_unet,mr_moving_unet,reg_unet, val_dl,epoch,epochs,loss1,loss2,loss3,loss4,val_losses,device1,device2,device3,TENSORBOARD_LOGDIR,SAVE_EVERY,WEIGHTS_SAVE_PATH,EXP_NO,FLAG,best_loss1,best_loss2,best_loss3):
    
    reg_unet.eval()
    mr_fixed_unet.eval()
    mr_moving_unet.eval()
    avg_loss1 = []
    avg_loss2 = []
    avg_loss3 = []
    sigmoid = nn.Sigmoid()
    Att_Conv1 = torch.nn.Conv2d(1, 1,1, stride=1, padding=0).to(device2)
    Att_Conv2 = torch.nn.Conv2d(1, 1,1, stride=1, padding=0).to(device3)
    v_pbar = tqdm(val_dl,desc=pbar_desc('valid', epoch , epochs, 0.0,0.0,0.0,0.0,0.0))
    BLoss = SurfaceLoss()
    with torch.no_grad():
        no = 0
        
        for us_img,mr_img,name,us_mask,mr_mask,distance_map_fixed,distance_map_moving in v_pbar:
            
            no +=1
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
            reg_unet_loss =smoothness_loss +BLoss(warped_mr_mask,distance_map_fixed)#+ similarity_loss
            reg_loss_display = reg_unet_loss.detach().cpu().item()
            
            #print(torch.min(warped_img2),torch.max(warped_img2))
            #print(torch.min(displacement_field),torch.max(displacement_field))
            #print(torch.min(map_fixed),torch.max(map_fixed))
            #print(torch.min(us_mask),torch.max(us_mask))
            #print(torch.min(map_moving),torch.max(map_moving))
            #print(torch.min(mr_mask),torch.max(mr_mask))
            #print(fixed_unet_loss)
            #print(moving_unet_loss)
            #print(similarity_loss)
            #print(smoothness_loss)
          
            v_pbar.set_description(pbar_desc('val',epoch,epochs,similarity_loss.item(),smoothness_loss.item(),reg_unet_loss.item(),fixed_unet_loss.item(),moving_unet_loss.item()))


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
            #if no % SAVE_EVERY == 0:
            if no==10 or no == 100 or no == 200:
                n1,_= name[0].split(sep='.') 
                name = n1
                #print(name)
                """
                mr_img1_new = nib.Nifti1Image(mr_img1_new, np.eye(4)) 
                mr_img1_new.header.get_xyzt_units()
                mr_img1_new.to_filename('./Result/mr_img1_'+name[0]+'.nii.gz')
            
                mr_img2_new = nib.Nifti1Image(mr_img2_new, np.eye(4)) 
                mr_img2_new.header.get_xyzt_units()
                mr_img2_new.to_filename('./Result/mr_img2_'+name[0]+'.nii.gz')
            
                warped_mr_img2 = nib.Nifti1Image(warped_mr_img2, np.eye(4)) 
                warped_mr_img2.header.get_xyzt_units()
                warped_mr_img2.to_filename('./Result/warped_mr_img2_'+name[0]+'.nii.gz')
            
                df = nib.Nifti1Image(df, np.eye(4)) 
                df.header.get_xyzt_units()
                df.to_filename('./Result/df_'+name[0]+'.nii.gz')
            
                map_fixed_out = nib.Nifti1Image(map_fixed_out, np.eye(4)) 
                map_fixed_out.header.get_xyzt_units()
                map_fixed_out.to_filename('./Result/map_fixed_'+name[0]+'.nii.gz')
            
                map_moving_out = nib.Nifti1Image(map_moving_out, np.eye(4)) 
                map_moving_out.header.get_xyzt_units()
                map_moving_out.to_filename('./Result/map_moving_'+name[0]+'.nii.gz')
                det_jac = nib.Nifti1Image(det_jac, np.eye(4)) 
                det_jac.header.get_xyzt_units()
                det_jac.to_filename('./Result/'+name[0]+'.nii.gz') 
                """
                np.save('./Result/us_'+name+'.npy',us_img_new)
                np.save('./Result/mr_'+name+'.npy',mr_img_new)
                np.save('./Result/warped_mr_img_'+name+'.npy',warped_mr_img)
                np.save('./Result/df_'+name+'.npy',df)
                np.save('./Result/map_fixed_'+name+'.npy',map_fixed_out)
                np.save('./Result/map_moving_'+name+'.npy',map_moving_out)
                np.save('./Result/mask_moving_'+name+'.npy',mr_mask_out)
                np.save('./Result/mask_fixed_'+name+'.npy',us_mask_out)
                np.save('./Result/Att_moving_'+name+'.npy',Att_moving_out)
                np.save('./Result/Att_fixed_'+name+'.npy',Att_fixed_out)
                np.save('./Result/warped_mask_'+name+'.npy',warped_mr_mask_out)
                
                (Image.fromarray(us_img_new*255).convert('L')).save('./Result/us_'+name+'.png')
                (Image.fromarray(mr_img_new*255).convert('L')).save('./Result/mr_'+name+'.png')
                (Image.fromarray(warped_mr_img*255).convert('L')).save('./Result/warped_mr_img_'+name+'.png')
                (Image.fromarray(df[0]*255).convert('L')).save('./Result/df_0_'+name+'.png')
                (Image.fromarray(df[1]*255).convert('L')).save('./Result/df_1_'+name+'.png')
                (Image.fromarray(map_fixed_out*255).convert('L')).save('./Result/map_fixed_'+name+'.png')
                (Image.fromarray(map_moving_out*255).convert('L')).save('./Result/map_moving_'+name+'.png')
                (Image.fromarray(mr_mask_out*255).convert('L')).save('./Result/mask_moving_'+name+'.png')
                (Image.fromarray(us_mask_out*255).convert('L')).save('./Result/mask_fixed_'+name+'.png')
                (Image.fromarray(Att_fixed_out*255).convert('L')).save('./Result/Att_fixed_'+name+'.png')
                (Image.fromarray(Att_moving_out*255).convert('L')).save('./Result/Att_moving_'+name+'.png')
                (Image.fromarray(warped_mr_mask_out*255).convert('L')).save('./Result/warped_mask_'+name+'.png')
            avg_loss1.append(fixed_unet_loss.item())
            avg_loss2.append(moving_unet_loss.item())
            avg_loss3.append(reg_unet_loss.item())
            
            
        avg_loss1 = torch.mean(torch.tensor(avg_loss1))
        avg_loss2 = torch.mean(torch.tensor(avg_loss2))
        avg_loss3 = torch.mean(torch.tensor(avg_loss3))

        
        avg_loss1 = torch.mean(torch.tensor(avg_loss1))
        avg_loss2 = torch.mean(torch.tensor(avg_loss2))
        avg_loss3 = torch.mean(torch.tensor(avg_loss3))
        val_losses.update(loss1 = similarity_loss.item(), loss2 = smoothness_loss.item(),loss3=reg_unet_loss.item(),loss4=fixed_unet_loss.item(), loss5 = moving_unet_loss.item())   
        if avg_loss1 < best_loss1 or epoch % SAVE_EVERY == 0:
            best_loss1 = avg_loss1
            torch.save(mr_fixed_unet.state_dict(), f'{WEIGHTS_SAVE_PATH}/{EXP_NO:02d}-mr_fixed_unet-epoch-{epoch:04d}_{best_loss1:.3f}.pth')
        if avg_loss2 < best_loss2 or epoch % SAVE_EVERY == 0:
            best_loss2 = avg_loss2
            torch.save(mr_moving_unet.state_dict(), f'{WEIGHTS_SAVE_PATH}/{EXP_NO:02d}-mr_moving_unet-epoch-{epoch:04d}_{best_loss2:.3f}.pth')
        if avg_loss3 < best_loss3 or epoch % SAVE_EVERY == 0:
            best_loss3 = avg_loss3
            torch.save(reg_unet.state_dict(), f'{WEIGHTS_SAVE_PATH}/{EXP_NO:02d}-reg_unet-epoch-{epoch:04d}_{best_loss3:.3f}.pth.tar')

    return best_loss1, best_loss2, best_loss3


