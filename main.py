import numpy as np
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import dataloader
import torch.nn.functional as F
import torchvision.transforms as T
import dill
from torchvision.utils import save_image
import random
import monai
#from monai.networks.nets import RegUNet,GlobalNet,LocalNet,UNet
from networks import init_weights, Reg_Net, U_Net, Reg_Net_MS,Reg_Net_MS_DF,Reg_Net_MS_DF_Deformable
from monai.losses import LocalNormalizedCrossCorrelationLoss,DiceFocalLoss, BendingEnergyLoss,GlobalMutualInformationLoss
from monai.networks.layers import Norm
from option import args
from train import *
from losses import *
#from torch.cuda.amp import autocast, GradScaler


TRAIN_BATCH_SIZE = args.TRAIN_BATCH_SIZE
VAL_BATCH_SIZE = args.VAL_BATCH_SIZE
LR = args.LR
WORKERS = args.WORKERS
device1 = args.DEVICE1
device2 = args.DEVICE2
device3 = args.DEVICE3
LR_DECAY = args.LR_DECAY
LR_STEP= args.LR_STEP
TRAIN_US = args.TRAIN_US
TRAIN_MR = args.TRAIN_MR
TRAIN_US_MASK = args.TRAIN_US_MASK
TRAIN_MR_MASK = args.TRAIN_MR_MASK
VAL_US = args.VAL_US
VAL_MR = args.VAL_MR
VAL_US_MASK = args.VAL_US_MASK
VAL_MR_MASK = args.VAL_MR_MASK
EXP_NO = args.EXP_NO 
LOAD_CHECKPOINT1 = args.LOAD_CHECKPOINT1
LOAD_CHECKPOINT2 = args.LOAD_CHECKPOINT2
LOAD_CHECKPOINT3 = args.LOAD_CHECKPOINT3
TENSORBOARD_LOGDIR = args.TENSORBOARD_LOGDIR 
END_EPOCH_SAVE_SAMPLES_PATH = args.END_EPOCH_SAVE_SAMPLES_PATH
WEIGHTS_SAVE_PATH = args.WEIGHTS_SAVE_PATH 
LOSS_WEIGHT = args.LOSS_WEIGHT
BATCHES_TO_SAVE = args.BATCHES_TO_SAVE 
SAVE_EVERY = args.SAVE_EVERY 
VISUALIZE_EVERY = args.VISUALIZE_EVERY 
EPOCHS = args.EPOCHS
NORM = args.NORM
FLAG = args.FLAG

class Bookkeeping:
    def __init__(self, tensorboard_log_path=None, suffix=''):
        self.loss_names = ['loss1','loss2','loss3','loss4','loss5']
        self.genesis()
        ## initialize tensorboard objects
        self.tboard = dict()
        if tensorboard_log_path is not None:
            if not os.path.exists(tensorboard_log_path):
                os.mkdir(tensorboard_log_path)
            for name in self.loss_names:
                self.tboard[name] = SummaryWriter(os.path.join(tensorboard_log_path, name + '_' + suffix))
            
    def genesis(self):
        self.losses = {key: 0 for key in self.loss_names}
        self.count = 0

    def update(self, **kwargs):
        for key in kwargs:
            self.losses[key]+=kwargs[key]
        self.count +=1

    def reset(self):
        self.genesis()

    def get_avg_losses(self):
        avg_losses = dict()
        for key in self.loss_names:
            avg_losses[key] = self.losses[key] / (self.count +1e-10)
        return avg_losses

    def update_tensorboard(self, epoch):
        avg_losses = self.get_avg_losses()
        for key in self.loss_names:
            self.tboard[key].add_scalar(key, avg_losses[key], epoch)


def save_checkpoint1(epoch, net, best_metrics, optimizer, lr_scheduler, filename='checkpoint1.pth.tar'):
    state = {'epoch': epoch, 'mr_fixed_unet_state_dict': net.state_dict(),
             'best_metrics': best_metrics, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    torch.save(state, filename, pickle_module=dill)


def save_checkpoint2(epoch, net, best_metrics, optimizer, lr_scheduler, filename='checkpoint2.pth.tar'):
    state = {'epoch': epoch, 'mr_moving_unet_state_dict': net.state_dict(),
             'best_metrics': best_metrics, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    torch.save(state, filename, pickle_module=dill)
    
def save_checkpoint3(epoch, net, best_metrics, optimizer, lr_scheduler, filename='checkpoint3.pth.tar'):
    state = {'epoch': epoch, 'reg_unet_state_dict': net.state_dict(),
             'best_metrics': best_metrics, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    torch.save(state, filename, pickle_module=dill)


def main():
    print("Welcome to MONAI")
    transforms = T.Compose([T.ToTensor()])#,T.CenterCrop(256)])
    trn_ds = dataloader.MRI_Dataset(TRAIN_US,TRAIN_MR,TRAIN_US_MASK,TRAIN_MR_MASK,NORM,FLAG,transforms)
    trn_dl = DataLoader(trn_ds, TRAIN_BATCH_SIZE, shuffle = True, num_workers = WORKERS)
    transforms = T.Compose([T.ToTensor()])#,T.CenterCrop(256)])
    val_ds = dataloader.MRI_Dataset(VAL_US,VAL_MR,VAL_US_MASK,VAL_MR_MASK,NORM,FLAG,transforms)
    val_dl = DataLoader(val_ds, VAL_BATCH_SIZE, shuffle = False, num_workers = WORKERS)
    start_epoch = 1
    #reg_unet = Reg_Net(2,2,4)
    reg_unet =Reg_Net_MS_DF_Deformable(2,2,8)
    init_weights(reg_unet)
    mr_moving_unet =  U_Net(1,1,8)
    init_weights(mr_moving_unet)
    mr_fixed_unet =  U_Net(1,1,8)
    init_weights(mr_fixed_unet)
    
    
    #print(reg_unet)
    print('Fixed-U-Net parameters:', sum(p.numel() for p in mr_fixed_unet.parameters()))
    print('Moving-U-Net parameters:', sum(p.numel() for p in mr_moving_unet.parameters()))
    print('REG-U-Net parameters:', sum(p.numel() for p in reg_unet.parameters()))

    loss1 = GlobalMutualInformationLoss().to(device1)
    loss2 = BendingEnergyLoss().to(device1)
    loss3 =Dice()# DiceFocalLoss ().to(device2)#torch.nn.LogSoftmax().to(device2)
    loss4 = Dice()# DiceFocalLoss ().to(device3)#torch.nn.LogSoftmax().to(device3)
    opt1 = torch.optim.Adam(mr_fixed_unet.parameters(), LR,weight_decay=0.0001)
    opt2 = torch.optim.Adam(mr_moving_unet.parameters(), LR,weight_decay=0.0001)
    opt3 = torch.optim.Adam(reg_unet.parameters(), LR,weight_decay=0.0001)
    #opt1 = torch.optim.SGD(mr_fixed_unet.parameters(), LR, momentum=0.9)
    #opt2 = torch.optim.SGD(mr_moving_unet.parameters(), LR, momentum=0.9)
    #opt3 = torch.optim.SGD(reg_unet.parameters(), LR, momentum=0.9)
    sched1 = optim.lr_scheduler.StepLR(opt1, LR_STEP, gamma=LR_DECAY)
    sched2 = optim.lr_scheduler.StepLR(opt2, LR_STEP, gamma=LR_DECAY)
    sched3 = optim.lr_scheduler.StepLR(opt3, LR_STEP, gamma=LR_DECAY)
    #sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, LR_STEP)
    #sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, LR_STEP)
    #sched3 = optim.lr_scheduler.CosineAnnealingLR(opt3, LR_STEP)

    if not os.path.exists(WEIGHTS_SAVE_PATH):
        os.mkdir(WEIGHTS_SAVE_PATH)

    if LOAD_CHECKPOINT1 is not None:
        checkpoint1 = torch.load(LOAD_CHECKPOINT1, pickle_module = dill,map_location=device2)
        start_epoch1 = checkpoint1['epoch']
        mr_fixed_unet.load_state_dict(checkpoint1['mr_fixed_unet_state_dict'],strict=False)
        opt1 = checkpoint1['optimizer']
        sched1 = checkpoint1['lr_scheduler']
    
    if LOAD_CHECKPOINT2 is not None:
        checkpoint2 = torch.load(LOAD_CHECKPOINT2, pickle_module = dill,map_location=device3)
        start_epoch2 = checkpoint2['epoch']
        mr_moving_unet.load_state_dict(checkpoint2['mr_moving_unet_state_dict'],strict=False)
        opt2 = checkpoint2['optimizer']
        sched2 = checkpoint2['lr_scheduler']
    if LOAD_CHECKPOINT3 is not None:
        checkpoint3 = torch.load(LOAD_CHECKPOINT3, pickle_module = dill,map_location=device1)
        start_epoch3 = checkpoint3['epoch']
        reg_unet.load_state_dict(checkpoint3['reg_unet_state_dict'],strict=False)
        opt3 = checkpoint3['optimizer']
        sched3 = checkpoint3['lr_scheduler']
    mr_fixed_unet.to(device2)
    mr_moving_unet.to(device3)
    reg_unet.to(device1)
    train_losses = Bookkeeping(TENSORBOARD_LOGDIR, suffix='trn')
    val_losses = Bookkeeping(TENSORBOARD_LOGDIR, suffix='val')
    best_loss1 = float('inf')
    best_loss2 = float('inf')
    best_loss3 = float('inf')
    #### for mixed precision ####
    #scaler = GradScaler()
    for epoch in range(start_epoch, EPOCHS+1):
        ## training loop
        
        train(mr_fixed_unet,mr_moving_unet,reg_unet, trn_dl,epoch,EPOCHS,loss1,loss2,loss3,loss4,opt1,opt2,opt3,train_losses,device1,device2,device3,TENSORBOARD_LOGDIR,LOSS_WEIGHT,SAVE_EVERY,WEIGHTS_SAVE_PATH,EXP_NO,FLAG)

        best_loss1,best_loss2,best_loss3= evaluate(mr_fixed_unet,mr_moving_unet,reg_unet, val_dl,epoch,EPOCHS,loss1,loss2,loss3,loss4,val_losses,device1,device2,device3,TENSORBOARD_LOGDIR,SAVE_EVERY,WEIGHTS_SAVE_PATH,EXP_NO,FLAG,best_loss1,best_loss2,best_loss3)
        
        sched1.step()
        sched2.step()
        sched3.step()
        save_checkpoint1(epoch, mr_fixed_unet, None, opt1, sched1)
        save_checkpoint2(epoch, mr_moving_unet, None, opt2, sched2)
        save_checkpoint3(epoch, reg_unet, None, opt3, sched3)

        train_losses.update_tensorboard(epoch)
        val_losses.update_tensorboard(epoch)

        ## Reset all losses for the new epoch
        train_losses.reset()
        val_losses.reset()
if __name__=='__main__':
    main()
