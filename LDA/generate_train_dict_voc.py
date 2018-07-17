import sys,os
from PIL import Image
import xml.etree.ElementTree as ET
import json

print '  '+sys.argv[0]
print '  Traverses the dataset images directory'
print '  builds a dictionary with images paths as keys and text articles paths as values'
print '  (...)'

db_dir  = '/mnt/lascar/qqiscen/src/TextTopicNet/data/VOC2007/VOCdevkit/VOC2007/'

img_dir = db_dir+'JPEGImages/'
xml_dir = db_dir+'Annotations/'

if not os.path.isdir(db_dir):
    sys.exit('ERR: Dataset folder '+db_dir+' not found!')

if not os.path.isdir(img_dir):
    sys.exit('ERR: Dataset images folder '+img_dir+' not found!')

if not os.path.isdir(xml_dir):
    sys.exit('ERR: Dataset metadata folder '+xml_dir+' not found!')



def get_article(xml_file):
    tree = ET.parse(xml_file)
  
    for obj in tree.findall('object'):                                                       
        obj_struct = {}
        obj_struct['article'] = obj.find('name').text
        return obj_struct

 

# Traverse dataset images directory, and list directories as dirs and files as files
# Build a dictionary with images paths as keys and text paths as values
# Discard non JPEG files, images from non english articles, and very small images (< 256 pixels)
train_dict = {}

for root, dirs, files in os.walk(img_dir):
    path = root.split('/')
    for file in files:
        ext = file.split('.')
        if ext[1] == 'jpg' or ext[1] == 'jpeg': # discard ~30k png files (usually diagrams, drawings, etc...)
            if not os.path.isfile(xml_dir+os.path.basename(root)+'/'+ext[0]+'.xml'):
                continue
            article = get_article(xml_dir+os.path.basename(root)+'/'+ext[0]+'.xml')
            # print(xml_dir+os.path.basename(root)+'/'+ext[0]+'.xml')
            if article == {}: # discard images from non english articles
                continue
            # print("len ",len(article))
            
            if article['article'] == '': # discard images from non english articles
                continue
            im = Image.open(root+'/'+file)
            width, height = im.size
            if width < 256 or height < 256: # discard small images
                continue
            
            img_path = path[len(path)-2]+'/'+file
            train_dict[img_path] = article['article']


with open('train_dict_VOC.json', 'w') as fp:
    json.dump(train_dict, fp)
