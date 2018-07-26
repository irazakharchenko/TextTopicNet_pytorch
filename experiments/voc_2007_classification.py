import sys,os
import random
import json
import numpy as np

from termcolor import colored
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim

from torch.autograd import Variable
sys.path.insert(0, '/home.guest/zakhairy/code/our_TextTopicNet/CNN/PyTorch')

import AlexNet_pool_norm


def get_vector(target_layer, t_img):
    
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    # 5. Attach that function to our selected layer
    h = target_layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding
  
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="5"

### Start : Extract the representation from specified layer and save in generated_data direcroty ###
if len(sys.argv)<2:
  print colored('You must provide the layer from wich to extract features. e.g. fc7, fc6, pool5, ...', 'red')
  quit()

layer = sys.argv[1]

# Specify paths to model prototxt and model weights
PATH = "/home.guest/zakhairy/code/our_TextTopicNet/CNN/PyTorch/model"
# Initialize pytorch model instnce with given weights and model prototxt
net = torch.load(PATH)
new_classifier = nn.Sequential(*list(net.children())[:-1])
model = new_classifier
IMG_SIZE = 256
MODEL_INPUT_SIZE = 227
MEAN = np.array([104.00698793, 116.66876762, 122.67891434])

# Specify path to directory containing PASCAL VOC2007 images
img_root = '/mnt/lascar/qqiscen/src/TextTopicNet/data/VOC2007/VOCdevkit/VOC2007/JPEGImages/'
out_root = '/home.guest/zakhairy/code/our_TextTopicNet/CNN/PyTorch/SVMs/VOC2007/generated_data/voc_2007_classification/features_'+layer+'/'
if not os.path.exists(out_root):
  os.makedirs(out_root)

# Get list of all file (image) names for VOC2007
onlyfiles = [f for f in os.listdir(img_root) if os.path.isfile(os.path.join(img_root, f))]

print colored('Starting image representation generation', 'green')
# For given layer and each given input image, generate corresponding representation
net = model
for param in net.parameters():
    param.requires_grad = False
net.eval()
for sample in onlyfiles[:10]:
  im_filename = img_root+sample
  

  im = Image.open(im_filename)
  im = im.resize((IMG_SIZE,IMG_SIZE)) # resize to IMG_SIZExIMG_SIZE
  im = im.crop((14,14,241,241)) # central crop of 227x227
  if len(np.array(im).shape) < 3:
    im = im.convert('RGB') # grayscale to RGB
  in_ = np.array(im, dtype=np.float32)
  in_ = in_[:,:,::-1] # switch channels RGB -> BGR
  in_ -= MEAN # subtract mean
  in_ = in_.transpose((2,0,1)) # transpose to channel x height x width order
  t_img = Variable( torch.from_numpy(np.flip(in_[np.newaxis,:,:,:] ,axis=0).copy()))
  
  output = net.forward(t_img) 
  # output_prob = get_vector(target_layer,t_img) # the output feature vector for the first image in the batch
  # print out_root+sample
  f = open(out_root+sample, 'w+')
  np.save(f, output)
  f.close()

print colored('Completed image representation generation.', 'green')
### End : Generating image representations for all images ###

### Start : Learn one vs all SVMs for each target class ###
features_root = out_root
svm_out_path = '/home.guest/zakhairy/code/our_TextTopicNet/CNN/PyTorch/SVMs/VOC2007/generated_data/voc_2007_classification/'+ layer + '_SVM'
if not os.path.exists(svm_out_path):
  os.makedirs(svm_out_path)
classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'] # List of classes in PASCAL VOC2007
cs = [13,14,15,16,17,18] # List of margins for SVM

# Specify ground truth paths for PASCAL VOC2007 dataset
gt_root = '/home.guest/zakhairy/code/our_TextTopicNet/CNN/PyTorch/SVMs/VOC2007/GT_labels/'
gt_train_sufix = '_train.txt'
gt_val_sufix = '_val.tx
gt_test_sufix = '_test.txt'

mAP2 = 0

for cl in classes:

  print colored("Do grid search for class "+cl, 'green')
  with open(gt_root+cl+gt_train_sufix) as f:
    content = f.readlines()
  aux = np.load(features_root+content[0].split(' ')[0]+'.jpg')
  X = np.zeros((len(content),(aux.flatten()).shape[0]), dtype=np.float32)
  y = np.zeros(len(content))
  idx = 0
  for sample in content:
    data = sample.split(' ')
    if data[1] == '': data[1] = '1'
    X[idx,:] = np.load(features_root+data[0]+'.jpg').flatten()
    y[idx]   = max(0,int(data[1]))
    idx = idx+1

  with open(gt_root+cl+gt_val_sufix) as f:
    content = f.readlines()
  XX = np.zeros((len(content),(aux.flatten()).shape[0]), dtype=np.float32)
  yy = np.zeros(len(content))
  idx = 0
  for sample in content:
    data = sample.split(' ')
    if data[1] == '': data[1] = '1'
    XX[idx,:] = np.load(features_root+data[0]+'.jpg').flatten()
    yy[idx]   = max(0,int(data[1]))
    idx = idx+1

  bestAP=0
  bestC=-1

  scaler = preprocessing.StandardScaler().fit(X)
  joblib.dump(scaler, './generated_data/voc_2007_classification/features_'+layer+'/scaler.pkl')
  X_scaled = scaler.transform(X)
  XX_scaled = scaler.transform(XX)

  for c in cs:
    clf = svm.LinearSVC(C=pow(0.5,c))
    clf.fit(X_scaled, y)
    #yy_ = clf.predict(XX)
    yy_ = clf.decision_function(XX_scaled)
    AP = average_precision_score(yy, yy_)
    if AP > bestAP:
      bestAP = AP
      bestC=pow(0.5,c)
  print " Best validation AP :"+str(bestAP)+" found for C="+str(bestC)
  mAP2=mAP2+bestAP
  X_all = np.concatenate((X, XX), axis=0)
  scaler = preprocessing.StandardScaler().fit(X_all)
  X_all = scaler.transform(X_all)
  joblib.dump(scaler, './generated_data/voc_2007_classification/features_'+layer+'/scaler.pkl')
  print X.shape, XX.shape, X_all.shape
  y_all = np.concatenate((y, yy))
  clf = svm.LinearSVC(C=bestC)
  clf.fit(X_all, y_all)
  joblib.dump(clf, svm_out_path + '/clf_'+cl+'_'+layer+'.pkl')
  print "  ... model saved as "+svm_out_path+'/clf_'+cl+'_'+layer+'.pkl'

print "\nValidation mAP: "+str(mAP2/float(len(classes)))+" (this is an underestimate, you must run VOC_eval.m for mAP taking into account don't care objects)"

### End : Learn one vs all SVMs for PASCAL VOC 2007 ###

### Start : Testing of learned SVMs ###
res_root = './generated_data/voc_2007_classification/'+layer+'_SVM/RES_labels/'
if not os.path.exists(res_root):
  os.makedirs(res_root)

mAP2=0

for cl in classes:

  with open(gt_root+cl+gt_test_sufix) as f:
    content = f.readlines()
  print "Testing one vs. rest SVC for class "+cl+" for "+str(len(content))+" test samples"
  aux = np.load(features_root+content[0].split(' ')[0]+'.jpg')
  X = np.zeros((len(content),(aux.flatten()).shape[0]), dtype=np.float32)
  y = np.zeros(len(content))
  idx = 0
  for sample in content:
    data = sample.split(' ')
    if data[1] == '': data[1] = '1'
    X[idx,:] = np.load(features_root+data[0]+'.jpg').flatten()
    y[idx]   = max(0,int(data[1]))
    idx = idx+1

  print "  ... loading model from "+svm_out_path+'clf_'+cl+'_'+layer+'.pkl'
  clf = joblib.load(svm_out_path+'/clf_'+cl+'_'+layer+'.pkl')
  scaler = joblib.load('./generated_data/voc_2007_classification/features_'+layer+'/scaler.pkl')
  X = scaler.transform(X)

#  y_ = clf.predict(X)
  y_ = clf.decision_function(X)
  AP = average_precision_score(y, y_)
  print "  ... Test AP: "+str(AP)
  mAP2 = mAP2+AP

  fr = open(res_root+'RES_cls_test_'+cl+'.txt','w+')
  idx = 0
  for sample in content:
    data = sample.split(' ')
    fr.write(str(data[0])+' '+str(y_[idx])+'\n')
    idx = idx+1
  fr.close()

print colored("\nTest mAP: "+str(mAP2/float(len(classes)))+" (this is an underestimate, you must run VOC_eval.m for mAP taking into account don't care objects)", 'green')
### End : Testing of learned SVMs ###
