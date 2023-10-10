import cv2
import numpy as np
import os
import time 
from shutil import copy
import argparse
import matplotlib.pyplot as plt 
import albumentations as A
from skimage.io import imread_collection
from clustering_alg import act_contour, agg_cluster, mean_shift, k_means, mask2contours, contours2mask 


def FrameCapture(path, img_path, fps):   
    vidObj = cv2.VideoCapture(path) 
    count = 0
    t = time.time()
    start = time.time()
    success, image = vidObj.read()
    while success: 
        if time.time()-t >= 1/(fps*20) or count==0:
            cv2.imwrite(img_path+f"/frame{count}.jpg", image) 
            t = time.time()
            count += 1

        if cv2.waitKey(1) == ord('q'):
            break 
        success, image = vidObj.read()
    vidObj.release()
    cv2.destroyAllWindows()

def augment(image, mask, num_name, num=5):
    transform = A.Compose([
    A.RandomCrop(p=0.7, width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.8),
    ])
    for i in range(num):
        name = str(format((num_name+i)/1000000, '6f'))
        name = f'/{name[2:]}.jpg'
        transformed = transform(image=image, mask = mask)
        trans_img = cv2.resize(transformed['image'], (WIDTH, HEIGHT))
        trans_mask = cv2.resize(transformed['mask'], (WIDTH, HEIGHT))
        plt.imsave(IMG_PATH+name, trans_img)
        plt.imsave(MASK_PATH+name, trans_mask)
    global num_imgs
    num_imgs+=num

def make_labels(cls=0):
    
    for filename in os.listdir(MASK_PATH):
        mask = cv2.imread(os.path.join(MASK_PATH, filename))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        trash, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_contours = mask2contours(mask, height=HEIGHT, width=WIDTH)
        txt_name = filename.split('.')[0]
        txt_path = os.path.join(LABELS_PATH, f'{txt_name}.txt')
        for cont in img_contours:
            seg = cont.reshape(-1)
            line = (cls, *seg)  # label format
            with open(txt_path, 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')


parser = argparse.ArgumentParser()
parser.add_argument('--source', metavar='source', default='images', type=str, help='enter the images folder')
parser.add_argument('--video', metavar='video', type=str, default='', help='enter the path to the video')
parser.add_argument('--fps', metavar='video', type=int, default=10, help='enter the fps for a video')
parser.add_argument('--aug', action='store_true', help='turn on the augmentation')
args = parser.parse_args()

'''
print('\n1--active contours, 2--agglomerative clustering, 3--mean shift, 4--k_means\n')
ind_alg=int(input('input the number of segmentation alg: '))-1
'''
ind_alg=3
algs = [act_contour, agg_cluster, mean_shift, k_means]
alg = algs[ind_alg]
alg_name=alg.__name__


WIDTH = 640
HEIGHT = 640
IMG_PATH = alg_name+'/seg-dataset/images/train'
LABELS_PATH = alg_name+'/seg-dataset/labels/train'
MASK_PATH=alg_name+'/masks'

path_imgs  = args.source
video_path = args.video
fps = args.fps
aug = args.aug

if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)
    os.makedirs(LABELS_PATH)
    os.makedirs(MASK_PATH)

if not os.path.exists(path_imgs):
    os.makedirs(path_imgs)

if video_path!='':
    FrameCapture(video_path, path_imgs, fps)

num_imgs = 0
for num, filename in enumerate(os.listdir(path_imgs)):
    name = str(format((num+1)/1000000, '6f'))
    copy(os.path.join(path_imgs, filename), os.path.join(IMG_PATH, f'{name[2:]}.jpg'))
    num_imgs = num+1

for num, filename in enumerate(os.listdir(IMG_PATH)):
        image = cv2.imread(os.path.join(IMG_PATH, filename))
        image = cv2.resize(image, (WIDTH, HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imsave(os.path.join(IMG_PATH, filename), image)
        img_contours = alg(image)
        mask = contours2mask(img_contours, os.path.join(MASK_PATH,filename), WIDTH, HEIGHT)
        if aug:
            augment(image, mask, 1+num_imgs)

make_labels()
