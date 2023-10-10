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
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.8),
    ])
    for i in range(num):
        name = f'/{num_name+i}.jpg'
        transformed = transform(image=image, mask = mask)
        plt.imsave(IMG_PATH+name, transformed['image'])
        plt.imsave(MASK_PATH+name, transformed['mask'])
    global num_imgs
    num_imgs+=num



parser = argparse.ArgumentParser()
parser.add_argument('--source', metavar='source', default='images', type=str, help='enter the images folder')
parser.add_argument('--video', metavar='video', type=str, default='', help='enter the path to the video')
parser.add_argument('--fps', metavar='video', type=int, default=10, help='enter the fps for a video')
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

path_imgs  = args.source
video_path = args.video
fps = args.fps

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
        mask = contours2mask(img_contours, MASK_PATH+f'/{name[2:]}.jpg', WIDTH, HEIGHT)
        augment(image, mask, 1+num_imgs, 7)


        for cont in img_contours:
            seg = cont.reshape(-1)
            line = (cls, *seg)  # label format
            with open(f'{txt_path}.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

