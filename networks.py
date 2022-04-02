import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import torchvision

## Initialization Function ##

Device1 = 'cuda:1'

def init_weights(net, init_type = 'kaiming', gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('conv') != -1 or classname.find('Linear') != -1):
            ### The find() method returns -1 if the value is not found
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
                ### notice: weight is a parameter object but weight.data is a tensor
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func) ### it is called for m iterating over every submodule of (in this case) net as well as net itself, due to the method call net.apply(…).

class SpatialTransformNearest_unit(nn.Module):
    def __init__(self):
        super(SpatialTransformNearest_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        #print(sample_grid.shape)
        #print(flow.shape)
        flow = torch.swapaxes(flow, 1,3)
        sample_grid = sample_grid + flow
        sample_grid = sample_grid.float()
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest', padding_mode="border", align_corners=True)
        return flow
        
def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    
    grid = np.rollaxis(np.array(np.meshgrid( y, x)), 0, 3)
    grid = np.swapaxes(grid, 0,1)
    grid = np.swapaxes(grid, 0,1)
    return grid
    


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.01),
                nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.01))
    def forward(self,x):
        
        x = self.conv(x)
        return x
        
class deformable_conv_block(nn.Module):
    def __init__(self):
        super(deformable_conv_block,self).__init__()
        
    def forward(self,x,kh,kw,batch_size,channel_in,channel_out,size_1,size_2):
        kh, kw = kh,kw
        weight = torch.rand(channel_out, channel_in, kh, kw).to(Device1)
        offset = torch.rand(batch_size, 2 * kh * kw, size_1, size_2).to(Device1)
        mask = torch.rand(batch_size, kh * kw, size_1, size_2).to(Device1)
        x = torchvision.ops.deform_conv2d(x,offset,weight,mask=mask)
        p1d = (1, 1, 1, 1) 
        x = F.pad(x, p1d, "constant", 0) 
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.01)) ###inplace=True means that it will modify the input directly, without allocating any additional output.
    def forward(self,x):
        x = self.up(x)
        return x


class Reg_Net(nn.Module):
    def __init__(self,img_ch = 2, output_ch = 2,initial_num_ch=64):
        super(Reg_Net,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=initial_num_ch)
        self.Conv2 = conv_block(ch_in = initial_num_ch,ch_out=2*initial_num_ch)
        self.Conv3 = conv_block(ch_in=2*initial_num_ch,ch_out=4*initial_num_ch)
        self.Conv4 = conv_block(ch_in=4*initial_num_ch,ch_out=8*initial_num_ch)
        self.Conv5 = conv_block(ch_in=8*initial_num_ch,ch_out=16*initial_num_ch)

        self.Up5 = up_conv(ch_in=16*initial_num_ch,ch_out=8*initial_num_ch)
        self.Up_conv5 = conv_block(ch_in=16*initial_num_ch, ch_out=8*initial_num_ch)

        self.Up4 = up_conv(ch_in=8*initial_num_ch,ch_out=4*initial_num_ch)
        self.Up_conv4 = conv_block(ch_in=8*initial_num_ch, ch_out=4*initial_num_ch)

        self.Up3 = up_conv(ch_in=4*initial_num_ch,ch_out=2*initial_num_ch)
        self.Up_conv3 = conv_block(ch_in=4*initial_num_ch, ch_out=2*initial_num_ch)

        self.Up2 = up_conv(ch_in=2*initial_num_ch,ch_out=initial_num_ch)
        self.Up_conv2 = conv_block(ch_in=2*initial_num_ch, ch_out=initial_num_ch)

        self.Conv_1x1 = nn.Conv2d(initial_num_ch,output_ch,kernel_size=1,padding=0)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.transformer = SpatialTransformNearest_unit()
    def forward(self,x):
        ## Encoder ##
        x1 = self.Conv1(x)
        print(x1.shape)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
       
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        ## Decoder ##

        d5 = self.Up5(x5)
        #print(d5.shape,x4.shape)
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        #d4 = self.Up4(x4)
        d4 = torch.cat((x3,d4),dim=1)

        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)

        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)

        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        #print('flow',d1.shape)
        #d1 = self.tanh(d1)
        
        
        #imshape = d1.shape[2:]
        #grid = generate_grid_unit(imshape)
        #grid = torch.tensor(grid)
        #grid = grid.unsqueeze(0).to(DEVICE)
        
        #out = self.transformer(mov_img,d1,grid)
        #out_seg = self.transformer(mov_seg,d1,grid)
        return d1
        
        
class Reg_Net_MS(nn.Module):
    def __init__(self,img_ch = 2, output_ch = 2,initial_num_ch=64):
        super(Reg_Net_MS,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=initial_num_ch)
        self.Conv2 = conv_block(ch_in = (initial_num_ch+2),ch_out=2*(initial_num_ch)+2)
        self.Conv3 = conv_block(ch_in=2*(initial_num_ch)+4,ch_out=4*(initial_num_ch)+2)
        self.Conv4 = conv_block(ch_in=4*(initial_num_ch)+4,ch_out=8*(initial_num_ch)+2)
        self.Conv5 = conv_block(ch_in=8*(initial_num_ch)+4,ch_out=16*(initial_num_ch)+2)

        self.Up5 = up_conv(ch_in=16*(initial_num_ch)+2,ch_out=8*(initial_num_ch)+4)
        self.Up_conv5 = conv_block(ch_in=16*(initial_num_ch)+2+4, ch_out=8*(initial_num_ch)+2)

        self.Up4 = up_conv(ch_in=8*(initial_num_ch)+2,ch_out=4*(initial_num_ch)+4)
        self.Up_conv4 = conv_block(ch_in=8*(initial_num_ch)+2+4, ch_out=4*(initial_num_ch)+2)

        self.Up3 = up_conv(ch_in=4*(initial_num_ch)+2,ch_out=2*(initial_num_ch)+4)
        self.Up_conv3 = conv_block(ch_in=4*(initial_num_ch)+2+4, ch_out=2*(initial_num_ch)+2)

        self.Up2 = up_conv(ch_in=2*(initial_num_ch)+2,ch_out=(initial_num_ch+2))
        self.Up_conv2 = conv_block(ch_in=2*(initial_num_ch)+2, ch_out=initial_num_ch)

        self.Conv_1x1 = nn.Conv2d(initial_num_ch,output_ch,kernel_size=1,padding=0)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.transformer = SpatialTransformNearest_unit()
    def forward(self,x):
        ## Encoder ##
        #print('input shape',x.shape)
        x1 = self.Conv1(x)
        #print('output of conv1',x1.shape)
        x2 = self.Maxpool(x1)
        x2_orig = self.Maxpool(x)
        x2 = torch.cat((x2,x2_orig),dim=1)
        #print('concatenated l2 shape', x2.shape)
        x2 = self.Conv2(x2)
        #print('output of conv2',x2.shape)
        x3 = self.Maxpool(x2)
        x3_orig = self.Maxpool(x2_orig)
        x3 = torch.cat((x3,x3_orig),dim=1)
        #print('concatenated l3 shape', x3.shape)
        x3 = self.Conv3(x3)
        #print('output of conv3',x3.shape)
        x4 = self.Maxpool(x3)
        x4_orig = self.Maxpool(x3_orig)
        x4 = torch.cat((x4,x4_orig),dim=1)
        #print('concatenated l4 shape', x4.shape)
        x4 = self.Conv4(x4)
        #print('output of conv4',x4.shape)
        x5 = self.Maxpool(x4)
        x5_orig = self.Maxpool(x4_orig)
        x5 = torch.cat((x5,x5_orig),dim=1)
        #print('concatenated l5 shape', x5.shape)
        x5 = self.Conv5(x5)
        #print('output of conv5',x5.shape)
        
        ## Decoder ##

        d5 = self.Up5(x5)
        #print('input shape to up5', d5.shape)
        #print('The shape of x4',x4.shape)
        d5 = torch.cat((x4,d5),dim=1)
        #print('input shape to up_conv5', d5.shape)

        d5 = self.Up_conv5(d5)
        #print('output shape to up_conv5', d5.shape)
        d4 = self.Up4(d5)
        #print('the shape of the Up4 output',d4.shape)
        #print('the shape of x3',x3.shape)
        #d4 = self.Up4(x4)
        d4 = torch.cat((x3,d4),dim=1)

        d4 = self.Up_conv4(d4)
        #print('The shape of Up_conv4 output', d4.shape)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)

        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)

        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        #print('flow',d1.shape)
        #d1 = self.tanh(d1)
        
        
        #imshape = d1.shape[2:]
        #grid = generate_grid_unit(imshape)
        #grid = torch.tensor(grid)
        #grid = grid.unsqueeze(0).to(DEVICE)
        
        #out = self.transformer(mov_img,d1,grid)
        #out_seg = self.transformer(mov_seg,d1,grid)
        return d1


class Reg_Net_MS_DF(nn.Module):
    def __init__(self,img_ch = 2, output_ch = 2,initial_num_ch=64):
        super(Reg_Net_MS_DF,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=initial_num_ch)
        self.Conv2 = conv_block(ch_in = (initial_num_ch+2),ch_out=2*(initial_num_ch)+2)
        self.Conv3 = conv_block(ch_in=2*(initial_num_ch)+4,ch_out=4*(initial_num_ch)+2)
        self.Conv4 = conv_block(ch_in=4*(initial_num_ch)+4,ch_out=8*(initial_num_ch)+2)
        self.Conv5 = conv_block(ch_in=8*(initial_num_ch)+4,ch_out=16*(initial_num_ch)+2)

        self.Up5 = up_conv(ch_in=16*(initial_num_ch)+2,ch_out=8*(initial_num_ch)+4)
        self.Up_conv5 = conv_block(ch_in=16*(initial_num_ch)+2+4, ch_out=8*(initial_num_ch)+2)
        self.df_conv5 = conv_block(ch_in=8*(initial_num_ch)+2,ch_out = 2)
        
        self.Up4 = up_conv(ch_in=8*(initial_num_ch)+2,ch_out=4*(initial_num_ch)+4)
        self.Up4_df = up_conv(ch_in=2,ch_out=2)
        self.Up_conv4 = conv_block(ch_in=8*(initial_num_ch)+2+4, ch_out=4*(initial_num_ch)+2)
        self.df_conv4 = conv_block(ch_in = 4*(initial_num_ch)+2,ch_out = 2)

        self.Up3 = up_conv(ch_in=4*(initial_num_ch)+2,ch_out=2*(initial_num_ch)+4)
        self.Up3_df = up_conv(ch_in=2,ch_out=2)
        self.Up_conv3 = conv_block(ch_in=4*(initial_num_ch)+2+4, ch_out=2*(initial_num_ch)+2)
        self.df_conv3 = conv_block(ch_in = 2*(initial_num_ch)+2, ch_out = 2)

        self.Up2 = up_conv(ch_in=2*(initial_num_ch)+2,ch_out=(initial_num_ch+2))
        self.Up2_df = up_conv(ch_in=2,ch_out=2)
        self.Up_conv2 = conv_block(ch_in=2*(initial_num_ch)+2, ch_out=initial_num_ch)
        self.df_conv2 = conv_block(ch_in = initial_num_ch, ch_out = 2)

        self.Conv_1x1 = nn.Conv2d(initial_num_ch,output_ch,kernel_size=1,padding=0)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.transformer = SpatialTransformNearest_unit()
    def forward(self,x):
        ## Encoder ##
        #print('input shape',x.shape)
        x1 = self.Conv1(x)
        #print('output of conv1',x1.shape)
        x2 = self.Maxpool(x1)
        x2_orig = self.Maxpool(x)
        x2 = torch.cat((x2,x2_orig),dim=1)
        #print('concatenated l2 shape', x2.shape)
        x2 = self.Conv2(x2)
        #print('output of conv2',x2.shape)
        x3 = self.Maxpool(x2)
        x3_orig = self.Maxpool(x2_orig)
        x3 = torch.cat((x3,x3_orig),dim=1)
        #print('concatenated l3 shape', x3.shape)
        x3 = self.Conv3(x3)
        #print('output of conv3',x3.shape)
        x4 = self.Maxpool(x3)
        x4_orig = self.Maxpool(x3_orig)
        x4 = torch.cat((x4,x4_orig),dim=1)
        #print('concatenated l4 shape', x4.shape)
        x4 = self.Conv4(x4)
        #print('output of conv4',x4.shape)
        x5 = self.Maxpool(x4)
        x5_orig = self.Maxpool(x4_orig)
        x5 = torch.cat((x5,x5_orig),dim=1)
        #print('concatenated l5 shape', x5.shape)
        x5 = self.Conv5(x5)
        #print('output of conv5',x5.shape)
        
        ## Decoder ##

        d5 = self.Up5(x5)
        #print('input shape to up5', d5.shape)
        #print('The shape of x4',x4.shape)
        d5 = torch.cat((x4,d5),dim=1)
        #print('input shape to up_conv5', d5.shape)

        d5 = self.Up_conv5(d5)
        df5 = self.df_conv5(d5)
        #print('output shape to up_conv5', d5.shape)
        d4 = self.Up4(d5)
        #print('the shape of the Up4 output',d4.shape)
        #print('the shape of x3',x3.shape)
        #d4 = self.Up4(x4)
        d4 = torch.cat((x3,d4),dim=1)

        d4 = self.Up_conv4(d4)
        df4 = self.df_conv4(d4)
        df4 = (df4 + self.Up4_df(df5))/2.0
        #print('The shape of Up_conv4 output', d4.shape)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)

        d3 = self.Up_conv3(d3)
        df3 = self.df_conv3(d3)
        df3 = (df3 + self.Up3_df(df4))/2.0

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)

        d2 = self.Up_conv2(d2)
        df2 = self.df_conv2(d2)
        df2 = (df2 + self.Up2_df(df3))/2.0

        d1 = self.Conv_1x1(d2)
        d1 = (d1+df2)/2.0
        #print('flow',d1.shape)
        #d1 = self.tanh(d1)
        
        
        #imshape = d1.shape[2:]
        #grid = generate_grid_unit(imshape)
        #grid = torch.tensor(grid)
        #grid = grid.unsqueeze(0).to(DEVICE)
        
        #out = self.transformer(mov_img,d1,grid)
        #out_seg = self.transformer(mov_seg,d1,grid)
        return d1
                
        
class U_Net(nn.Module):
    def __init__(self,img_ch = 1, output_ch = 1, initial_num_ch = 64):
        super(U_Net,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=initial_num_ch)
        self.Conv2 = conv_block(ch_in = initial_num_ch,ch_out=2*initial_num_ch)
        self.Conv3 = conv_block(ch_in=2*initial_num_ch,ch_out=4*initial_num_ch)
        self.Conv4 = conv_block(ch_in=4*initial_num_ch,ch_out=8*initial_num_ch)
        self.Conv5 = conv_block(ch_in=8*initial_num_ch,ch_out=16*initial_num_ch)

        self.Up5 = up_conv(ch_in=16*initial_num_ch,ch_out=8*initial_num_ch)
        self.Up_conv5 = conv_block(ch_in=16*initial_num_ch, ch_out=8*initial_num_ch)

        self.Up4 = up_conv(ch_in=8*initial_num_ch,ch_out=4*initial_num_ch)
        self.Up_conv4 = conv_block(ch_in=8*initial_num_ch, ch_out=4*initial_num_ch)

        self.Up3 = up_conv(ch_in=4*initial_num_ch,ch_out=2*initial_num_ch)
        self.Up_conv3 = conv_block(ch_in=4*initial_num_ch, ch_out=2*initial_num_ch)

        self.Up2 = up_conv(ch_in=2*initial_num_ch,ch_out=initial_num_ch)
        self.Up_conv2 = conv_block(ch_in=2*initial_num_ch, ch_out=initial_num_ch)

        self.Conv_1x1 = nn.Conv2d(initial_num_ch,output_ch,kernel_size=1,padding=0)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        ## Encoder ##
        x1 = self.Conv1(x)
        #print(x1.shape)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
       
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        ## Decoder ##

        d5 = self.Up5(x5)
        #print(d5.shape,x4.shape)
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        #d4 = self.Up4(x4)
        d4 = torch.cat((x3,d4),dim=1)

        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)

        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)

        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        #print('flow',d1.shape)
        #d1 = self.tanh(d1)
        
        #d1 =  nn.functional.softmax(d1,dim =1 )
        
        return d1





class Reg_Net_MS_DF_Deformable(nn.Module):
    def __init__(self,img_ch = 2, output_ch = 2,initial_num_ch=64):
        super(Reg_Net_MS_DF_Deformable,self).__init__()
        self.initial_num_ch = initial_num_ch
        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = deformable_conv_block()#conv_block(ch_in=img_ch,ch_out=initial_num_ch)
        self.Conv2 = deformable_conv_block()#conv_block(ch_in = (initial_num_ch+2),ch_out=2*(initial_num_ch)+2)
        self.Conv3 = deformable_conv_block()#conv_block(ch_in=2*(initial_num_ch)+4,ch_out=4*(initial_num_ch)+2)
        self.Conv4 = deformable_conv_block()#conv_block(ch_in=4*(initial_num_ch)+4,ch_out=8*(initial_num_ch)+2)
        self.Conv5 = deformable_conv_block()#conv_block(ch_in=8*(initial_num_ch)+4,ch_out=16*(initial_num_ch)+2)

        self.Up5 = up_conv(ch_in=16*(initial_num_ch)+2,ch_out=8*(initial_num_ch)+4)
        self.Up_conv5 = conv_block(ch_in=16*(initial_num_ch)+2+4, ch_out=8*(initial_num_ch)+2)
        self.df_conv5 = conv_block(ch_in=8*(initial_num_ch)+2,ch_out = 2)
        
        self.Up4 = up_conv(ch_in=8*(initial_num_ch)+2,ch_out=4*(initial_num_ch)+4)
        self.Up4_df = up_conv(ch_in=2,ch_out=2)
        self.Up_conv4 = conv_block(ch_in=8*(initial_num_ch)+2+4, ch_out=4*(initial_num_ch)+2)
        self.df_conv4 = conv_block(ch_in = 4*(initial_num_ch)+2,ch_out = 2)

        self.Up3 = up_conv(ch_in=4*(initial_num_ch)+2,ch_out=2*(initial_num_ch)+4)
        self.Up3_df = up_conv(ch_in=2,ch_out=2)
        self.Up_conv3 = conv_block(ch_in=4*(initial_num_ch)+2+4, ch_out=2*(initial_num_ch)+2)
        self.df_conv3 = conv_block(ch_in = 2*(initial_num_ch)+2, ch_out = 2)

        self.Up2 = up_conv(ch_in=2*(initial_num_ch)+2,ch_out=(initial_num_ch+2))
        self.Up2_df = up_conv(ch_in=2,ch_out=2)
        self.Up_conv2 = conv_block(ch_in=2*(initial_num_ch)+2, ch_out=initial_num_ch)
        self.df_conv2 = conv_block(ch_in = initial_num_ch, ch_out = 2)

        self.Conv_1x1 = nn.Conv2d(initial_num_ch,output_ch,kernel_size=1,padding=0)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.transformer = SpatialTransformNearest_unit()
    def forward(self,x):
        ## Encoder ##
        #print('input shape',x.shape)
        inch = self.initial_num_ch
        kh,kw = 3,3
        batch_size = x.shape[0]
        channel_in = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        x1 = self.Conv1(x,kh, kw,batch_size ,channel_in,inch,h-2,w-2)
        #print('output of conv1',x1.shape)
        #print("The shape of x",x.shape)
        #print("the shape of x1",x1.shape)
        
        x2 = self.Maxpool(x1)
        x2_orig = self.Maxpool(x)
        #print("x2 shape",x2.shape)
        #print("x2 original shape",x2_orig.shape)
        x2 = torch.cat((x2,x2_orig),dim=1)
        #print('concatenated l2 shape', x2.shape)
        h = x2.shape[2]
        w = x2.shape[3]
        x2 = self.Conv2(x2,kh,kw,batch_size,inch+2,2*(inch)+2,h-2,w-2)
        #print('output of conv2',x2.shape)
        x3 = self.Maxpool(x2)
        x3_orig = self.Maxpool(x2_orig)
        x3 = torch.cat((x3,x3_orig),dim=1)
        #print('concatenated l3 shape', x3.shape)
        h = x3.shape[2]
        w = x3.shape[3]
        x3 = self.Conv3(x3,kh,kw,batch_size,2*(inch)+4,4*(inch)+2,h-2,w-2)
        #print('output of conv3',x3.shape)
        x4 = self.Maxpool(x3)
        x4_orig = self.Maxpool(x3_orig)
        x4 = torch.cat((x4,x4_orig),dim=1)
        #print('concatenated l4 shape', x4.shape)
        h = x4.shape[2]
        w = x4.shape[3]
        x4 = self.Conv4(x4,kh,kw,batch_size,4*(inch)+4,8*(inch)+2,h-2,w-2)
        #print('output of conv4',x4.shape)
        x5 = self.Maxpool(x4)
        x5_orig = self.Maxpool(x4_orig)
        x5 = torch.cat((x5,x5_orig),dim=1)
        #print('concatenated l5 shape', x5.shape)
        h = x5.shape[2]
        w = x5.shape[3]
        x5 = self.Conv5(x5,kh,kw,batch_size,8*(inch)+4,16*(inch)+2,h-2,w-2)
        #print('output of conv5',x5.shape)
        
        ## Decoder ##

        d5 = self.Up5(x5)
        #print('input shape to up5', d5.shape)
        #print('The shape of x4',x4.shape)
        d5 = torch.cat((x4,d5),dim=1)
        #print('input shape to up_conv5', d5.shape)

        d5 = self.Up_conv5(d5)
        df5 = self.df_conv5(d5)
        #print('output shape to up_conv5', d5.shape)
        d4 = self.Up4(d5)
        #print('the shape of the Up4 output',d4.shape)
        #print('the shape of x3',x3.shape)
        #d4 = self.Up4(x4)
        d4 = torch.cat((x3,d4),dim=1)

        d4 = self.Up_conv4(d4)
        df4 = self.df_conv4(d4)
        df4 = (df4 + self.Up4_df(df5))/2.0
        #print('The shape of Up_conv4 output', d4.shape)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)

        d3 = self.Up_conv3(d3)
        df3 = self.df_conv3(d3)
        df3 = (df3 + self.Up3_df(df4))/2.0

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)

        d2 = self.Up_conv2(d2)
        df2 = self.df_conv2(d2)
        df2 = (df2 + self.Up2_df(df3))/2.0

        d1 = self.Conv_1x1(d2)
        d1 = (d1+df2)/2.0
        #print('flow',d1.shape)
        #d1 = self.tanh(d1)
        
        
        #imshape = d1.shape[2:]
        #grid = generate_grid_unit(imshape)
        #grid = torch.tensor(grid)
        #grid = grid.unsqueeze(0).to(DEVICE)
        
        #out = self.transformer(mov_img,d1,grid)
        #out_seg = self.transformer(mov_seg,d1,grid)
        return d1

