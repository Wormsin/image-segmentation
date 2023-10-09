import cv2
import numpy as np
import os
from shutil import copy
import argparse
import matplotlib.pyplot as plt 
from skimage.io import imread_collection
from clustering_alg import act_contour, agg_cluster, mean_shift, k_means



def contours2mask(contours, path):
    masks = np.zeros((640, 640), dtype=np.uint8)
    for i in range(len(contours)):
        cont = contours[i]
        cont[:,0] = cont[:, 0]*WIDTH
        cont[:, 1] = cont[:, 1]*HEIGHT
        mask = np.zeros((WIDTH, HEIGHT), dtype=np.uint8)
        cont = cont.astype(np.int32)
        polygons = cont.reshape(1, -1, 2)
        mask = cv2.fillPoly(mask, polygons, color=1)
        masks+=mask
    plt.imsave(path, masks)


parser = argparse.ArgumentParser()
parser.add_argument('--source', metavar='source', type=str, help='enter the images folder')
args = parser.parse_args()

print('\n1--active contours, 2--agglomerative clustering, 3--mean shift, 4--k_means\n')
ind_alg=int(input('input the number of segmentation alg: '))-1
algs = [act_contour, agg_cluster, mean_shift, k_means]
alg = algs[ind_alg]
alg_name=alg.__name__

WIDTH = 640
HEIGHT = 640
IMG_PATH = alg_name+'-seg/images/train'
LABELS_PATH = alg_name+'-seg/labels/train'
MASK_PATH=alg_name+'-seg/masks'

path  = args.source

if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)
    os.makedirs(LABELS_PATH)
    os.makedirs(MASK_PATH)


for num, filename in enumerate(os.listdir(path)):
    name = str(format((num+1)/1000000, '6f'))
    copy(os.path.join(path, filename), os.path.join(IMG_PATH, f'{name[2:]}.jpg'))


images_coll = imread_collection(IMG_PATH +'/*jpg*')
images = []
for img in images_coll:
    images.append(cv2.resize(img, (WIDTH, HEIGHT)))


cls = 0
for num, image in enumerate(images):
    name = str(format((num+1)/1000000, '6f'))
    txt_path = LABELS_PATH + f'/{name[2:]}'
    if not os.path.exists(f'{txt_path}.txt'):
        img_contours = alg(image)
        contours2mask(img_contours, MASK_PATH+f'/{name[2:]}.jpg')
        for cont in img_contours:
            seg = cont.reshape(-1)
            line = (cls, *seg)  # label format
            with open(f'{txt_path}.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

