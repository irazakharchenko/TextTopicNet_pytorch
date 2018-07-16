from __future__ import print_function, division
import os, sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from ast import literal_eval
from PIL import Image
from random import randint
from shutil import copyfile
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file, root_dir, transform=None):
        """
        Args:
            file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.topics = []
        self.images = []
        len = 20
        with open(file, "r") as inp:
            li = inp.readline()
            while li :
                key, val = li.split("\t")
                
                self.images.append(key)
                self.topics.append(literal_eval("[" + val + "]"))
                if  len > 0 and randint(0,2) == 1:
                    copyfile(root_dir + key, "/home.guest/zakhairy/code/our_TextTopicNet/data/some_ph/" + key.split("/")[-1])
                    len -= 1
                li = inp.readline()
        #print(self.landmarks_frame[20])
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.root_dir,
                                self.images[idx])
        image = Image.open(img_name)
        
        if randint(0,1) == 1:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if len(np.array(image).shape) < 3:
            image = image.convert('RGB')
        image = np.asarray(image)
        topics = self.topics[idx]
        if self.transform:
            image, topics = self.transform([image,  topics])
        
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        topics = torch.from_numpy(np.asarray(topics)).long()
        
        return [image,  topics]

class Rescale(object):
    """Rescale the image in a sample to a given size.   

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample[0], sample[1]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return [img, landmarks]


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        

        return [ image,  landmarks]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if ( len(image.shape) < 3 ):
            image = to_rgb2(image)
        
        image = image.transpose((2, 0, 1))
        print( "image shape ", image.shape)
        return [ torch.from_numpy(image).float(),
                 torch.from_numpy(landmarks).long()]


class Normalize(object):
    "Normalize image"
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, sample):
        image, landmarks = sample
        image = np.divide(np.subtract(image, self.mean.expand_as(image)) , self.std.expand_as(image))
        return [image, landmarks]


def to_rgb2(im):
    # as 1, but we use broadcasting in one line
    w, h = im.size
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, :] = im[:, :, np.newaxis]
    return ret