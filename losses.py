import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Functions import generate_grid_unit

device = 'cuda:1'

def Gaussian(input_t, kernel_size = 7, sigma = 5,channels = 1):
    # Create a x, y, z coordinate grid of shape (kernel_size, kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.permute(1,0)
    
    xyz_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 3-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *torch.exp(-torch.sum((xyz_grid - mean)**2., dim=-1) /(2*variance))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 3d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels,1, 1, 1)
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel.to(device)
    gaussian_filter.weight.requires_grad = False
    #print(type(input_t))
    input_t = input_t.float()
    input_t = input_t.to(device)
    output_t = gaussian_filter(input_t)
    return output_t

def Laplacian(y_pred):
    dyp = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dxp = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    
    
    dy = torch.abs(dyp[:, :, 1:, :] - dyp[:, :, :-1, :])
    dx = torch.abs(dxp[:, :, :, 1:] - dxp[:, :, :, :-1])
    
    
        
    grad = torch.mean(dx) + torch.mean(dy) 
   
    return dx, dy, grad

class LOG(torch.nn.Module):
    def __init__(self):
        super(LOG,self).__init__()

    def forward(self, y_pred, y_true):
        G1 = Gaussian(y_pred,kernel_size = 3, sigma = 0.1,channels = 1)
        G2 = Gaussian(y_true,kernel_size = 3, sigma = 0.1,channels = 1)
        dxp,dyp,Lp = Laplacian(G1)
        dxt,dyt,Lt = Laplacian(G2)
        
        Pad_func1 = nn.ConstantPad2d((2, 2, 1, 1), 0)
        dxt = Pad_func1(dxt)
        dxp = Pad_func1(dxp)
        
        Pad_func2 = nn.ConstantPad2d((1, 1,2,2), 0)
        dyt = Pad_func2(dyt)
        dyp = Pad_func2(dyp)
        
        LOG_pred = dxp+dyp
        LOG_true = dxt+dyt
        return LOG_pred,LOG_true


class Dice(nn.Module):
    def __init__(self):
        super(Dice,self).__init__()
    def forward(self, y_true, y_pred):
        smooth = 1
        iflat = y_pred.contiguous().view(-1)
        tflat = y_true.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        dice_loss = 1-((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
        return dice_loss
        
def smoothloss(y_pred):
    dy = torch.abs(y_pred[:,:,1:, :] - y_pred[:,:, :-1, :])
    dx = torch.abs(y_pred[:,:,:, 1:] - y_pred[:,:, :, :-1])
    
    return (torch.mean(dx * dx)+torch.mean(dy*dy))/2.0        
        
        
def my_distance_map(fixed_mask):
    D_map = np.zeros(fixed_mask.shape)
    center_of_mass = ndimage.measurements.center_of_mass(fixed_mask)
    for i in range(0,fixed_mask.shape[0]):
        for j in range(0,fixed_mask.shape[1]):
            if fixed_mask[i,j]!=0.0:
                D_map[i,j] =-np.sum(np.square((i - center_of_mass[0])+(j-center_of_mass[1])))
            else:
                D_map[i,j] =np.sum(np.square((i - center_of_mass[0])+(j-center_of_mass[1])))
    return D_map
    
    
class SurfaceLoss(nn.Module):
    def __init__(self):
        super(SurfaceLoss,self).__init__()
    def forward(self, y_pred,dis_map):
        L_B = torch.mean(torch.mul(y_pred,dis_map))
        return L_B  
