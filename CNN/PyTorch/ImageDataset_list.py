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
import json

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
        with open(file, "r") as inp:
            if file.endswith("json"):
                data = json.load(inp)
                for key in data.keys():
                    if "/25/" in key or "/26/" in key:
                        # these folders are missed
                        continue
                    self.images.append(key)
                    self.topics.append(data[key])
            else: 
            
                li = inp.readline()
                while li :
                    key, val = li.split("\t")
                    
                    self.images.append(key)
                    self.topics.append(literal_eval("[" + val + "]"))
                    li = inp.readline()
            
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
        if self.transform :
            image, topics = self.transform([image,  topics])

        image = image.transpose((2, 0, 1))  
        # print([ x for x in topics if x != 0])
        # print("image size {}".format(image.shape))
        image = torch.from_numpy(np.flip(image,axis=0).copy()).float()
        topics = torch.from_numpy(np.asarray(topics)).float()
        
        return [image,  topics]

class Rescale(object):
    """Rescale the image in a sample to a given size.   

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample[0], sample[1]

        img = transform.resize(image, self.output_size)
        
        return [img, landmarks]


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        
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

        # print("shape img in random crop {}".format(image.shape))
        return [ image,  landmarks]


class Normalize(object):
    "Normalize image"
    def __init__(self, mean):
        self.mean = np.array(mean, dtype='f')
      

    def __call__(self, sample):
        image, landmarks = sample
        image = image[:,:,::-1] # switch channels RGB -> BGR0
        
        # print("image {}, image after substraction {}".format(image[0][:10], (image - self.mean)[0][:10]))
        # print("image size {}".format(image.shape))  
        image = image - self.mean
        # print("image size {}".format(image.shape))
        return [image, landmarks]

class Mirroring(object):
    "Mirror image"
    def __init__(self,):
        pass
    def __call__(self, sample):
        image, landmarks = sample
        im = image
        if randint(0,1) == 1:
            im = im[::-1,::-1,:]

        return [im, landmarks]
        


def to_rgb2(im):
    # as 1, but we use broadcasting in one line
    w, h = im.size
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, :] = im[:, :, np.newaxis]
    return ret 