import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Select the values of the hyper-parameters')
    parser.add_argument('--TEST_BATCH_SIZE', default = 1)
    parser.add_argument('--WORKERS', default = 8)
    parser.add_argument('--DEVICE1', default = 'cuda:1')
    parser.add_argument('--DEVICE2', default = 'cuda:0')
    parser.add_argument('--DEVICE3', default = 'cuda:2')

    
    parser.add_argument('--TEST_US', default = '/home/Drive3/sahar_datasets/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/test_new/us/')
    parser.add_argument('--TEST_MR', default = '/home/Drive3/sahar_datasets/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/test_new/mr/')
    parser.add_argument('--TEST_US_MASK', default = '/home/Drive3/sahar_datasets/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/test_new/us_mask/')
    parser.add_argument('--TEST_MR_MASK', default = '/home/Drive3/sahar_datasets/Registration/3DRegST/SupervisedRegST/MuliScaleReg/LapIRN/2D/Data/test_new/mr_mask/')
    
    parser.add_argument('--LOAD_CHECKPOINT1', default ='checkpoint1.pth.tar')
    parser.add_argument('--LOAD_CHECKPOINT2', default ='checkpoint2.pth.tar')
    parser.add_argument('--LOAD_CHECKPOINT3', default = 'checkpoint3.pth.tar')
    parser.add_argument('--EPOCHS', default = 1)
    parser.add_argument('--NORM', default = True)
    parser.add_argument('--FLAG', default = False)
    #args = parser.parse_args()
    return parser.parse_args()
