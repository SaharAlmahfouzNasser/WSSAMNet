import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import itertools
import torch.nn.functional as F
import torchvision.transforms as T

def Trans(TS):
    transforms = T.Compose([T.CenterCrop(200)])
    out = transforms(TS)
    return out

def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def transform_unit_flow_to_flow(flow):
    x, y, z, _ = flow.shape
    flow[:, :, :, 0] = flow[:, :, :, 0] * (z-1)
    flow[:, :, :, 1] = flow[:, :, :, 1] * (y-1)
    flow[:, :, :, 2] = flow[:, :, :, 2] * (x-1)

    return flow


def transform_unit_flow_to_flow_cuda(flow):
    b, x, y, z, c = flow.shape
    flow[:, :, :, :, 0] = flow[:, :, :, :, 0] * (z-1)
    flow[:, :, :, :, 1] = flow[:, :, :, :, 1] * (y-1)
    flow[:, :, :, :, 2] = flow[:, :, :, :, 2] * (x-1)

    return flow


def load_4D(name):
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + X.shape)
    return X


def load_5D(name):
    X = fixed_nii = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + (1,) + X.shape)
    return X


def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)

    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


def save_img(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


def save_img_nii(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


def save_flow(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


class Dataset(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, iterations, norm=False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.iterations = iterations

    def __len__(self):
        'Denotes the total number of samples'
        return self.iterations

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        index_pair = np.random.permutation(len(self.names))[0:2]
        img_A = load_4D(self.names[index_pair[0]])
        img_B = load_4D(self.names[index_pair[1]])
        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, norm=False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.index_pair[step][0])
        img_B = load_4D(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Predict_dataset(Data.Dataset):
    def __init__(self, fixed_list, move_list, fixed_label_list, move_label_list, norm=False):
        super(Predict_dataset, self).__init__()
        self.fixed_list = fixed_list
        self.move_list = move_list
        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.move_list)

    def __getitem__(self, index):
        fixed_img = np.load(self.fixed_list[index])
        moved_img = np.load(self.move_list[index])
        fixed_label = np.load(self.fixed_label_list[index])
        moved_label = np.load(self.move_label_list[index])

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)
        
        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)
        fixed_label = torch.from_numpy(fixed_label)
        moved_label = torch.from_numpy(moved_label)
        factor = 64
        fixed_img = fixed_img[factor:256-factor,factor:256-factor,factor:288-factor]
        moved_img = moved_img[factor:256-factor,factor:256-factor,factor:288-factor]
        fixed_label = fixed_label[factor:256-factor,factor:256-factor,factor:288-factor]
        moved_label = moved_label[factor:256-factor,factor:256-factor,factor:288-factor]
        ################
        fixed_label = fixed_label.type(torch.LongTensor)
        fixed_label = F.one_hot(fixed_label, num_classes=16)
        #print(fixed_label.shape)
        fixed_label = fixed_label.permute(3,0,1,2)
        #fixed_label = torch.squeeze(fixed_label,0)
        fixed_label = fixed_label.type(torch.FloatTensor)
        moved_label = moved_label.type(torch.LongTensor)
        moved_label = F.one_hot(moved_label, num_classes=16)
        moved_label = moved_label.permute(3,0,1,2)
        #moved_label = torch.squeeze(moved_label,0)
        moved_label = moved_label.type(torch.FloatTensor)
        ################
        fixed_img = fixed_img.unsqueeze(0)
        moved_img = moved_img.unsqueeze(0)
        #fixed_label = fixed_label.unsqueeze(0)
        #moved_label = moved_label.unsqueeze(0)
        if self.norm:
            output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                      'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
            return output
        else:
            output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                      'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
            return output
