# EXAMPLE_PROCESS_IMAGES  Code to read and process images for ROxford and RParis datasets.
# Revisited protocol requires query images to be removed from the database, and cropped prior to any processing.
# This code makes sure the protocol is strictly followed.
#
# More details about the revisited annotation and evaluation can be found in:
# Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking, CVPR 2018
#
# Authors: Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., 2018

import os, sys
import numpy as np

from PIL import Image, ImageFile

from dataset import configdataset
from download import download_datasets

import torch
from torch.autograd import Variable
sys.path.insert(0, '/home.guest/zakhairy/code/our_TextTopicNet/CNN/PyTorch')
import AlexNet_pool_norm
from tempfile import TemporaryFile
from torchvision import transforms

outfile_X = TemporaryFile()
outfile_Y = TemporaryFile()

#---------------------------------------------------------------------
# Set data folder and testing parameters
#---------------------------------------------------------------------
# Set data folder, change if you have downloaded the data somewhere else
data_root = os.path.join("../../data/roxford")
# Check, and, if necessary, download test data (Oxford and Pairs) and revisited annotation
download_datasets(data_root)

# Set test dataset: roxford5k | rparis6k
test_dataset = 'roxford5k'
IMG_SIZE = 256
MEAN = np.array([0,0,0])
#---------------------------------------------------------------------
# Read images
#---------------------------------------------------------------------

def pil_loader(path):
    # to avoid crashing for truncated (corrupted images)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning 
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def transformation(im):
    im = im.resize((IMG_SIZE,IMG_SIZE)) # resize to IMG_SIZExIMG_SIZE
    im = im.crop((14,14,241,241)) # central crop of 227x227
    if len(np.array(im).shape) < 3:
            im = im.convert('RGB') # grayscale to RGB
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1] # switch channels RGB -> BGR
    in_ -= MEAN # subtract mean
    in_ = in_.transpose((2,0,1)) # transpose to channel x height x width order
    t_img = Variable( torch.from_numpy(np.flip(in_[np.newaxis,:,:,:] ,axis=0).copy()))
    return t_img



print ('>> {}: Processing test dataset...'.format(test_dataset)) 
# config file for the dataset
# separates query image list from database image list, if revisited protocol used
cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets'))


PATH = "/home.guest/zakhairy/code/our_TextTopicNet/CNN/PyTorch/model"
print ('Model weights are loaded from : ' + PATH)

# Initialize pytorch model instnce with given weights and model prototxt
# 'torch.nn.parallel.data_parallel.DataParallel' has changed.
# torch.nn.Module.dump_patches = True
# net = AlexNet_pool_norm.AlexnetPoolNorm()
net = torch.load(PATH)
# net.load_state_dict(torch.load(PATH))
for param in net.parameters():
    param.requires_grad = False

# query images
X = np.empty((40,cfg['nq']))

for i in np.arange(cfg['nq']):
    qim = pil_loader(cfg['qim_fname'](cfg, i))
    t_im = transformation(qim)
    output = net.forward(t_im)
    output.t()
    print("{} {}".format(X[:,i].shape, output.size()))
    X[:,i] = output
    print('>> {}: Processing query image {}'.format(test_dataset, i+1))
np.save(outfile_X, X)

Y = np.empty((1000,cfg['nq']))

for i in np.arange(cfg['n']):
    im = pil_loader(cfg['im_fname'](cfg, i))
    ##------------------------------------------------------
    ## Perform image processing here, eg, feature extraction
    ##------------------------------------------------------
    output = net.forward(qim)
    Y[:,i] = output 
    print('>> {}: Processing database image {}'.format(test_dataset, i+1))
np.save(outfile_Y, Y)
