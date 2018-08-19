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
import torch.nn as nn
sys.path.insert(0, '/home.guest/zakhairy/code/our_TextTopicNet/CNN/PyTorch')
import AlexNet_pool_norm
from torchvision import transforms

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="6"


#---------------------------------------------------------------------
# Set data folder and testing parameters
#---------------------------------------------------------------------
# Set data folder, change if you have downloaded the data somewhere else
data_root = os.path.join("../../data/roxford")
# Check, and, if necessary, download test data (Oxford and Pairs) and revisited annotation
download_datasets(data_root)

# Set test dataset: roxford5k | rparis6k
test_dataset = 'rparis6k'
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


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

print ('>> {}: Processing test dataset...'.format(test_dataset)) 
# config file for the dataset   
# separates query image list from database image list, if revisited protocol used
layers = [["pool5",  0, 256 * 6 * 6], ["fc6", 2, 4096], ["fc7", -2, 4096]]
our_layer = 0

cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets'))


PATH = "/home.guest/zakhairy/code/our_TextTopicNet/CNN/PyTorch/model"
print ('Model weights are loaded from : ' + PATH)

net = torch.load(PATH)

if layers[our_layer][0] == "pool5":
    net.classifier = nn.Sequential()
else:
    net.classifier = nn.Sequential(
                    *list(net.classifier.children())[:layers[our_layer][1]]
                )
# net.load_state_dict(torch.load(PATH))
for param in net.parameters():
    param.requires_grad = False

# query images
X = np.empty(( layers[our_layer][2], cfg['nq']))

for i in np.arange(cfg['nq']):
    
    qim = pil_loader(cfg['qim_fname'](cfg, i)).crop(cfg['gnd'][i]['bbx'])
    t_im = transformation(qim)
    output = net.forward(t_im)

    X[:,i] = output.data.cpu().numpy()

    print('>> {}: Processing query image {}'.format(test_dataset, i+1))
X = normalized(X,0)
 
# save X to file
np.save("X_"+layers[our_layer][0] + test_dataset +".npy", X)

# 5000
Q = np.empty(( layers[our_layer][2], cfg['n']))

for i in np.arange(cfg['n']):
    im = pil_loader(cfg['im_fname'](cfg, i))
    t_im = transformation(im)
    output = net.forward(t_im)
    # print("{}".format(np.array(output[0,0])))
    # print("{}".format(np.squeeze(output).size()))
    Q[:,i] = output.data.cpu().numpy()
    print('>> {}: Processing database image {}'.format(test_dataset, i+1))
Q = normalized(Q,0)
# save Q to file
np.save("Q_"+layers[our_layer][0] + test_dataset +".npy", Q)
