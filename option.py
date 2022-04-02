import argparse

EXP_NO = 1
parser = argparse.ArgumentParser(description='Select the values of the hyper-parameters')
parser.add_argument('--TRAIN_BATCH_SIZE', default = 8)
parser.add_argument('--VAL_BATCH_SIZE', default = 1)
parser.add_argument('--LR', default = 0.0001)
parser.add_argument('--WORKERS', default = 8)
parser.add_argument('--DEVICE1', default = 'cuda:1')
parser.add_argument('--DEVICE2', default = 'cuda:0')
parser.add_argument('--DEVICE3', default = 'cuda:2')

parser.add_argument('--LR_DECAY', default = 0.5)
parser.add_argument('--LR_STEP', default = 10000)
"""
parser.add_argument('--TRAIN_US', default = '/home/sahar/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/fixed_data/dataset/train/us/')
parser.add_argument('--TRAIN_MR', default = '/home/sahar/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/fixed_data/dataset/train/mr/')
parser.add_argument('--TRAIN_US_MASK', default = '/home/sahar/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/fixed_data/dataset/train/us_mask/')
parser.add_argument('--TRAIN_MR_MASK', default = '/home/sahar/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/fixed_data/dataset/train/mr/')

parser.add_argument('--VAL_US', default = '/home/sahar/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/fixed_data/dataset/test/us/')
parser.add_argument('--VAL_MR', default = '/home/sahar/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/fixed_data/dataset/test/mr/')
parser.add_argument('--VAL_US_MASK', default = '/home/sahar/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/fixed_data/dataset/test/us_mask/')
parser.add_argument('--VAL_MR_MASK', default = '/home/sahar/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/fixed_data/dataset/test/mr/')
"""

parser.add_argument('--TRAIN_US', default = '/home/Drive3/sahar_datasets/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/train_new/us/')
parser.add_argument('--TRAIN_MR', default = '/home/Drive3/sahar_datasets/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/train_new/mr/')
parser.add_argument('--TRAIN_US_MASK', default = '/home/Drive3/sahar_datasets/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/train_new/us_mask/')
parser.add_argument('--TRAIN_MR_MASK', default = '/home/Drive3/sahar_datasets/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/train_new/mr_mask/')

parser.add_argument('--VAL_US', default = '/home/Drive3/sahar_datasets/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/val_new/us/')
parser.add_argument('--VAL_MR', default = '/home/Drive3/sahar_datasets/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/val_new/mr/')
parser.add_argument('--VAL_US_MASK', default = '/home/Drive3/sahar_datasets/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/val_new/us_mask/')
parser.add_argument('--VAL_MR_MASK', default = '/home/Drive3/sahar_datasets/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/val_new/mr_mask/')


#parser.add_argument('--TEST', default = '/home/Drive3/sahar_datasets/BraTSReg_Training_Data_v2/TEST')


parser.add_argument('--EXP_NO', default = EXP_NO)
parser.add_argument('--LOAD_CHECKPOINT1', default =None)#'checkpoint1.pth.tar')#None)
parser.add_argument('--LOAD_CHECKPOINT2', default =None)#'checkpoint2.pth.tar')
parser.add_argument('--LOAD_CHECKPOINT3', default = None)#'checkpoint3.pth.tar')
parser.add_argument('--TENSORBOARD_LOGDIR', default = f'{EXP_NO:02d}-tboard')
parser.add_argument('--END_EPOCH_SAVE_SAMPLES_PATH', default = f'{EXP_NO:02d}-epoch_end_samples')
parser.add_argument('--WEIGHTS_SAVE_PATH', default = f'{EXP_NO:02d}-weights')
parser.add_argument('--LOSS_WEIGHT', default = 0.5)
parser.add_argument('--BATCHES_TO_SAVE', default = 1)
parser.add_argument('--SAVE_EVERY', default = 1)
parser.add_argument('--VISUALIZE_EVERY', default = 10)
parser.add_argument('--EPOCHS', default = 500)
parser.add_argument('--NORM', default = True)
parser.add_argument('--FLAG', default = False)
args = parser.parse_args()

