from __future__ import division

import sys,os
import random
import json
import numpy as np

from termcolor import colored
from PIL import Image
import gensim
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score
from sklearn import preprocessing
import scipy.stats as sp
from preprocess_text import preprocess_imageclef

import torch

from torch.autograd import Variable
sys.path.insert(0, '/home.guest/zakhairy/code/our_TextTopicNet/CNN/PyTorch')

import AlexNet_pool_norm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="6"

num_topics = 40
layer = 'prob'
type_data_list = ['test']

# Function to compute average precision for text retrieval given image as input
def get_AP_img2txt(sorted_scores, given_image, top_k):
        consider_top = sorted_scores[:top_k]
        top_text_classes = [GT_txt2img[i[0]][1] for i in consider_top]
        class_of_image = GT_img2txt[given_image][1]
        T = top_text_classes.count(class_of_image)
        R = top_k
        sum_term = 0
        for i in range(0,R):
                if top_text_classes[i] != class_of_image:
                        pass
                else:
                        p_r = top_text_classes[:i+1].count(class_of_image)
                        sum_term = sum_term + float(p_r/len(top_text_classes[:i+1]))
        if T == 0:
                return 0
        else:
                return float(sum_term/T)

# Function to compute average precision for image retrieval given text as input
def get_AP_txt2img(sorted_scores, given_text, top_k):
        consider_top = sorted_scores[:top_k]
        top_image_classes = [GT_img2txt[i[0]][1] for i in consider_top]
        class_of_text = GT_txt2img[given_text][1]
        T = top_image_classes.count(class_of_text)
        R = top_k
        sum_term = 0
        for i in range(0,R):
                if top_image_classes[i] != class_of_text:
                        pass
                else:
                        p_r = top_image_classes[:i+1].count(class_of_text)
                        sum_term = sum_term + float(p_r/len(top_image_classes[:i+1]))
        if T == 0:
                return 0
        else:
                return float(sum_term/T)


if len(sys.argv) < 2:
        print 'Please enter the type of query. Eg txt2img, img2txt'
        quit()
query_type=sys.argv[1]
# query_type="img2txt"
### Start : Generating image representations of wikipedia dataset for performing multi modal retrieval

layer = 'prob' # Note : Since image and text has to be in same space for retieval. CNN layer has to be 'prob'
num_topics = 40 # Number of topics for the corresponding LDA model


# Specify path to model prototxt and model weights
PATH = "/home.guest/zakhairy/code/our_TextTopicNet/CNN/PyTorch/model"
print colored('Model weights are loaded from : ' + PATH	, 'green')


# Initialize pytorch model instnce with given weights and model prototxt
net = torch.load(PATH)
for param in net.parameters():
	param.requires_grad = False
IMG_SIZE = 256
MODEL_INPUT_SIZE = 227
MEAN = np.array([104.00698793, 116.66876762, 122.67891434])

text_dir_wd = '../data/Wikipedia/wikipedia_dataset/texts/' # Path to wikipedia dataset text files
img_root = "../data/Wikipedia/wikipedia_dataset/images_wd_256/" # Path to wikipedia dataset image files
image_categories = ['art', 'geography', 'literature', 'music', 'sport', 'biology', 'history', 'media', 'royalty', 'warfare'] # List of document (image-text) categories in wikipedia dataset

# Generate representation for each image in wikipedia dataset
for type_data in type_data_list:
	# Specify path to wikipedia dataset image folder and output folder to store features
	out_root = '/home.guest/zakhairy/code/our_TextTopicNet/generated_data/multi_modal_retrieval/image/' + str(layer) + '/' + str(num_topics) + '/' + str(type_data) + '/'
	if not os.path.exists(out_root):	
		os.makedirs(out_root)
	im_txt_pair_wd = open('/mnt/lascar/qqiscen/src/TextTopicNet/data/Wikipedia/'+str(type_data)+'set_txt_img_cat.list', 'r').readlines() # Image-text pairs
	img_files = [i.split('\t')[1] + '.jpg' for i in im_txt_pair_wd] # List of image files in wikipedia dataset
	for sample in img_files:
			im_filename = img_root+sample
			print colored('Generating image representation for : ' + im_filename, 'green')
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
			# output_prob = net.blobs[layer].data[0] # the output feature vector for the first image in the batch
			# print(output)	
			f = open(out_root+sample, "w+")
			np.save(f, output)
			f.close()
print 'Finished generating representation for wikipedia dataset images'
### End : Generating image representations of wikipedia dataset for performing multi modal retrieval

### Start : Generating text representation of wikipedia dataset for performing multi modal retrieval

choose_set_list = type_data_list

# IMPORTANT ! Specify minimum probability for LDA
min_prob_LDA = None

# load id <-> term dictionary
dictionary = gensim.corpora.Dictionary.load('../LDA/dictionary.dict')

# load LDA model
ldamodel = gensim.models.ldamulticore.LdaMulticore.load('../LDA/ldamodel'+str(num_topics)+'.lda', mmap='r')
for choose_set in choose_set_list:
	# Read image-text document pair ids
	im_txt_pair_wd = open('../data/Wikipedia/'+str(choose_set)+'set_txt_img_cat.list', 'r').readlines()
	text_files_wd = [text_dir_wd + i.split('\t')[0] + '.xml' for i in im_txt_pair_wd]
	output_path_root = './generated_data/multi_modal_retrieval/text/'
	if not os.path.exists(output_path_root):
		os.makedirs(output_path_root)
	output_file_path = 'wd_txt_' + str(num_topics) + '_' + str(type_data) + '.json'
	
	# transform ALL documents into LDA space
	output_path = output_path_root + output_file_path
	TARGET_LABELS = {}
	for i in text_files_wd:
        	print colored('Generating text representation for document number : ' + str(len(TARGET_LABELS.keys())), 'green')
        	raw = open(i,'r').read()
        	process = preprocess_imageclef(raw)
        	# print "len raw / len process[0]  " + str(len(raw)) +" / " + str(len(process[0]))
        	if process[1] != '':
                	tokens = process[0]
                	# print tokens
                	bow_vector = dictionary.doc2bow(tokens)
                	# print "len bow_ve	ctor "+ str(len(bow_vector))
                	
                	while bow_vector[-1][0] > 100000:
                	    bow_vector = bow_vector[:int(-0.05*len(bow_vector))]
                	lda_vector = ldamodel.get_document_topics(bow_vector[:], minimum_probability=None)
                	#lda_vector = ldamodel[bow_vector]
                	lda_vector = sorted(lda_vector,key=lambda x:x[1],reverse=True)
                	topic_prob = {}
                	for instance in lda_vector:
                        	topic_prob[instance[0]] = instance[1]
                	labels = []
                	for topic_num in range(0,num_topics):
                        	if topic_num in topic_prob.keys():
                                	labels.append(topic_prob[topic_num])
                        	else:
                                	labels.append(0)
                	list_name =  i.split('/')
                	TARGET_LABELS[list_name[len(list_name) -1 ].split('.xml')[0]] = labels
	print TARGET_LABELS
	# Save thi as json.
	with open(output_path,'w') as fp:
		for key in TARGET_LABELS.keys():
			s = key + "\t"
			for el in TARGET_LABELS[key]:
				s += str(el) + ","
			s = s[:-1]
			s += "\n"
			fp.write(s)


### End : Generating text representation of wikipedia dataset for performing multi modal retrieval

### Start : Perform multi-modal retrieval on wikipedia dataset.

for type_data in type_data_list:
	# Wikipedia data paths
	im_txt_pair_wd = open('../data/Wikipedia/'+str(type_data)+'set_txt_img_cat.list', 'r').readlines()
	image_files_wd = [i.split('\t')[1] + '.jpg' for i in im_txt_pair_wd]
	print "len im_txt_pair_wd "+ str(len(im_txt_pair_wd))
	# Read the required Grount Truth for the task.
	GT_img2txt = {} # While retrieving text, you need image as key.
	GT_txt2img = {} # While retrieving image, you need text as key.
	for i in im_txt_pair_wd:
        	GT_img2txt[i.split('\t')[1]] = (i.split('\t')[0], i.split('\t')[2]) # (Corresponding text, class)
        	GT_txt2img[i.split('\t')[0]] = (i.split('\t')[1], i.split('\t')[2]) # (Corresponding image, class)

	# Load image representation
	image_rep = './generated_data/multi_modal_retrieval/image/' + str(layer) + '/' + str(num_topics) + '/' + str(type_data) + '/'

	# Load text representation
	# data_text = json.load(open('./generated_data/multi_modal_retrieval/text/wd_txt_' + str(num_topics) + '_' + str(type_data) + '.json','r'))
	data_text = {}
	with open('./generated_data/multi_modal_retrieval/text/wd_txt_' + str(num_topics) + '_' + str(type_data) + '.txt', "r") as inp:
        	li = inp.readline()
        	while li :
        		key, val = li.split("\t")
        		data_text[key] = eval("["+val+"]")
        		
        		li = inp.readline()
                


	image_ttp = {}
	for i in GT_img2txt.keys():
        	sample = i
        	value = np.load(image_rep + i + '.jpg')
        	image_ttp[sample] = value

	# Convert text_rep to numpy format
	text_ttp = {}
	for i in data_text.keys():
        	text_ttp[i] = np.asarray(data_text[i], dtype='f')
	# IMPORTANT NOTE : always sort the images and text in lexi order !!
	# If Query type is input=image, output=text
	mAP = 0
	if query_type == 'img2txt':
	        counter = 0
        	order_of_images = sorted(image_ttp.keys())
        	order_of_texts = sorted(text_ttp.keys())
        	print order_of_images
        	for given_image in order_of_images:
                	print colored('Performing retrieval for document number : ' + str(counter), 'green')
                	score_texts = []
                	image_reps = image_ttp[given_image]
                	for given_text in order_of_texts:
                        	text_reps = text_ttp[given_text]
                        	given_score = sp.entropy(text_reps, image_reps)
                        	score_texts.append((given_text, given_score))
                	sorted_scores = sorted(score_texts, key=lambda x:x[1],reverse=False)
                	mAP = mAP + get_AP_img2txt(sorted_scores, given_image, top_k=len(order_of_texts))
                	counter += 1
        	print image_ttp
        	print colored('MAP img2txt : ' + str(float(mAP/len(image_ttp.keys()))), 'red')
	if query_type == 'txt2img' :
        	counter = 0
        	order_of_images = sorted(image_ttp.keys())
        	order_of_texts = sorted(text_ttp.keys())
        	for given_text in order_of_texts:
                	print colored('Performing retrieval for document number : ' + str(counter), 'green')
                	score_images = []
                	text_reps = text_ttp[given_text]
                	for given_image in order_of_images:
                        	image_reps = image_ttp[given_image]
                        	given_score = sp.entropy(text_reps, image_reps)
                        	score_images.append((given_image, given_score))
                	sorted_scores = sorted(score_images, key=lambda x:x[1],reverse=False)
                	mAP = mAP + get_AP_txt2img(sorted_scores, given_text, top_k=len(order_of_images))
                	counter += 1
        	print colored('MAP txt2img : ' + str(float(mAP/len(text_ttp.keys()))), 'red')

### End : Perform multi modal retrieval on wikipedia dataset
