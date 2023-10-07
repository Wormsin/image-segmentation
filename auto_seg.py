import cv2
import numpy as np
import os
from skimage.io import imread_collection
from clustering_alg import act_contour, agg_cluster, mean_shift, k_means
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', metavar='source', type=str, help='enter the images folder')
args = parser.parse_args()

WIDTH = 640
HIGHT = 640
IMG_PATH = 'seg-dataset/images/train'
LABELS_PATH = 'seg-dataset/labels/train'

path  = args.source

if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)
    os.makedirs(LABELS_PATH)

for num, filename in enumerate(os.listdir(path)):
    name = str(format((num+1)/1000000, '6f'))
    os.rename(os.path.join(path, filename), os.path.join(IMG_PATH, f'{name[2:]}.jpg'))


images_coll = imread_collection(IMG_PATH +'/*jpg*')
images = []
for img in images_coll:
    images.append(cv2.resize(img, (WIDTH, HIGHT)))


cls = 0
for num, image in enumerate(images):
    img_contours = agg_cluster(image)
    name = str(format((num+1)/1000000, '6f'))
    txt_path = LABELS_PATH + f'/{name[2:]}'
    for cont in img_contours:
        seg = cont.reshape(-1)
        line = (cls, *seg)  # label format
        with open(f'{txt_path}.txt', 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

