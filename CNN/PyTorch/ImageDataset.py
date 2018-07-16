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
        len = 1000
        with open(file, "r") as inp:
            li = inp.readline()
            while li :
                key, val = li.split("\t")
                
                self.images.append(key)
                self.topics.append(literal_eval("[" + val + "]"))
                
                li = inp.readline()
                len -= 1
                if len < 0:
                    break
        #print(self.landmarks_frame[20])
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx][0])
        image = Image.open(img_name)
        image = image.convert('RGB')
        #print(image)

        if self.transform:
            image = self.transform(image)
        landmarks = self.landmarks_frame.iloc[idx][1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 1)
        
        sample = {'image': image, 'landmarks': landmarks}
        
        

        return sample

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
        image, landmarks = sample['image'], sample['landmarks']

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

        return {'image': img, 'landmarks': landmarks}


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
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if ( len(image.shape) < 3 ):
            image = to_rgb2(image)
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image).double(),
                'landmarks': torch.from_numpy(landmarks).double()}


class Normalize(object):
    "Normalize image"
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = np.divide(np.subtract(image, self.mean.expand_as(image)) , self.std.expand_as(image))
        return {'image': image,
                'landmarks':landmarks}


def to_rgb2(im):
    # as 1, but we use broadcasting in one line
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, :] = im[:, :, np.newaxis]
    return ret