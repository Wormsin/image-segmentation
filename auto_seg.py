import cv2
import numpy as np
import os
import time 
from shutil import copy, rmtree
import argparse
import matplotlib.pyplot as plt 
import albumentations as A
from clustering_alg import act_contour, agg_cluster, mean_shift, k_means, mask2contours, contours2mask

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
IMG_PATH = 'seg-dataset/images/train'
LABELS_PATH = 'seg-dataset/labels/train'
MASK_PATH='masks'


def FrameCapture(path, img_path, fps):   
    vidObj = cv2.VideoCapture(path) 
    count = 0
    t = time.time()
    success, image = vidObj.read()
    dt = 0.001*(30/fps-1)
    while success: 
        if time.time()-t >= dt or count==0:
            cv2.imwrite(img_path+f"/frame{count}.jpg", image) 
            t = time.time()
            count += 1

        #if cv2.waitKey(1) == ord('q'):
         #   break 
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
    A.Transpose(p=1)
    ])
    for i in range(num):
        name = str(format((num_name+i)/1000000, '6f'))
        name = f'/{name[2:]}.jpg'
        impath = IMG_PATH+name
        mask_path = MASK_PATH+name
        if not os.path.exists(mask_path):
            transformed = transform(image=image, mask = mask)
            trans_img = transformed['image']
            trans_mask = transformed['mask']
            plt.imsave(impath, trans_img)
            plt.imsave(mask_path, trans_mask)
    global num_imgs
    num_imgs+=num

def make_labels(cls=0):
    for filename in os.listdir(MASK_PATH):
        mask = cv2.imread(os.path.join(MASK_PATH, filename))
        txt_name = filename.split('.')[0]
        txt_path = os.path.join(LABELS_PATH, f'{txt_name}.txt')
        if not os.path.exists(txt_path):
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            trash, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            h_mask, w_mask = mask.shape
            img_contours = mask2contours(mask, height=h_mask, width=w_mask)
            for cont in img_contours:
                seg = cont.reshape(-1)
                line = (cls, *seg)  # label format
                with open(txt_path, 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')


def aspect_ratio(img):
    h, w, ch = img.shape
    r = max(np.round(w/WIDTH), np.round(h/HEIGHT))
    wn, hn  = (int(w//r), int(h//r))
    return (hn, wn)
def img_resize(img, path):
    hn, wn  = aspect_ratio(img)
    img = cv2.resize(img, (wn, hn))
    canvas = np.ones((HEIGHT, WIDTH, 3), dtype='uint8')*255
    x_offset = abs(wn - WIDTH)
    y_offset = abs(hn - HEIGHT)
    canvas[0:HEIGHT-y_offset, 0:WIDTH-x_offset] = img
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    plt.imsave(path, canvas)
    return canvas

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', metavar='source', default='images', type=str, help='enter the images folder')
    parser.add_argument('--video', metavar='video', type=str, default='', help='enter the path to the video')
    parser.add_argument('--fps', metavar='video', type=int, default=10, help='enter the fps for a video')
    parser.add_argument('--aug', action='store_true', help='turn on the augmentation')
    args = parser.parse_args()
    return args

def get_class_num(data):
    with open(data, "r+b") as file:
        file.seek(-5, os.SEEK_END)
        
        while file.read(1) != b'\n':
            file.seek(-2, os.SEEK_CUR)
        last_line = file.readline().decode()
    if last_line == "names:\n":
        return 0
    else:
        return int(last_line[3])+1


def run(args):

    path_imgs  = args.source
    video_path = args.video
    fps = args.fps
    aug = args.aug

    if os.path.exists(IMG_PATH):
        rmtree("seg-dataset")
        rmtree(MASK_PATH)
    os.makedirs(IMG_PATH)
    os.makedirs(LABELS_PATH)
    os.makedirs(MASK_PATH)

    if video_path!='':
        if os.path.exists(path_imgs):
            rmtree(path_imgs)
        os.makedirs(path_imgs)
        FrameCapture(video_path, path_imgs, fps) 
    elif not os.path.exists(path_imgs):
        print("There's no any image for segmentation, input the --source argument")
        return

    global num_imgs
    num_imgs = 0
    for num, filename in enumerate(os.listdir(path_imgs)):
        name = str(format((num+1)/1000000, '6f'))
        copy(os.path.join(path_imgs, filename), os.path.join(IMG_PATH, f'{name[2:]}.jpg'))
        num_imgs = num+1

    
    ORIGINAL_NUM = num_imgs
    for num, filename in enumerate(os.listdir(IMG_PATH)):
            image = cv2.imread(os.path.join(IMG_PATH, filename))
            height_new, width_new = aspect_ratio(image)
            image = cv2.resize(image, (width_new, height_new))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #plt.imsave(os.path.join(IMG_PATH, filename), image)
            img_contours = alg(image)
            mask = contours2mask(img_contours, os.path.join(MASK_PATH,filename), height_new, width_new)
            if aug and num<ORIGINAL_NUM:
                augment(image, mask, 1+num_imgs)

    make_labels(get_class_num("dataset-seg.yaml"))

def main(args):
    run(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)