from skimage.filters import gaussian
from skimage.segmentation import active_contour
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift, estimate_bandwidth
import cv2
import numpy as np
import matplotlib.pyplot as plt

def contours2mask(contours, path, width, height, save=True):
    masks = np.zeros((640, 640), dtype=np.uint8)
    for i in range(len(contours)):
        cont = contours[i]
        cont[:,0] = cont[:, 0]*width
        cont[:, 1] = cont[:, 1]*height
        mask = np.zeros((width, height), dtype=np.uint8)
        cont = cont.astype(np.int32)
        polygons = cont.reshape(1, -1, 2)
        mask = cv2.fillPoly(mask, polygons, color=1)
        masks+=mask
    if save:
        plt.imsave(path, masks)
    return masks

def mask2contours(mask, rows, cols):
    mask = mask.astype('uint8')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [x for x in contours]
    for i in range(len(contours)):
        contours[i] = contours[i].reshape(-1, 2)/[cols, rows]
    return contours

def res_down(n, img):
    while(n>0):
        img = cv2.pyrDown(img)
        n = n-1
    img = np.float32(img)
    return img


def act_contour(img):
    snakes = []
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rows, cols, chs = img.shape
    c0 = cols/2
    r0 = rows/2
    s = np.linspace(0, 2*np.pi, 200)
    r = r0 + 250*np.sin(s)
    c = c0 + 280*np.cos(s)
    init = np.array([r, c]).T

    snake = active_contour(gaussian(img, 0, preserve_range=False), init, alpha=0.15, beta=0.01, gamma=0.05)

    snake = snake[:, [1, 0]]
    snake = snake/[cols, rows]
    snakes.append(snake[1::4])
    
    return snakes

def agg_cluster(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = res_down(3, img)
    img = cv2.GaussianBlur(img, (5, 5), 1)
        
    rows, cols, chs = img.shape
        
    indices = np.dstack(np.indices(img.shape[:2]))
    xycolors = np.concatenate((img, indices), axis=-1) 
    feature_image = np.reshape(xycolors, [-1,5])
    #feature_image=np.reshape(img, [-1, 3])

    agglo = AgglomerativeClustering(n_clusters=2)
    agglo.fit(feature_image)

    mask = np.reshape(agglo.labels_, [rows, cols])
    contours = mask2contours(mask, rows, cols)

    return contours

def mean_shift(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = res_down(3, img)

    rows, cols, chs = img.shape

    indices = np.dstack(np.indices(img.shape[:2]))
    xycolors = np.concatenate((img, indices), axis=-1) 
    feature_image = np.reshape(xycolors, [-1,5])
    #feature_image = np.reshape(img, [-1,3])
    
    bandwidth = estimate_bandwidth(feature_image, quantile=0.45, n_samples=200, random_state=0)

    msh = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    msh.fit(feature_image)
    mask = np.reshape(msh.labels_, [rows, cols])
    contours = mask2contours(mask, rows, cols)
        
    return contours

def k_means(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = res_down(3, img)
    rows, cols, chs = img.shape

    feature_img = img.reshape((-1,3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    k = 2
    retval, labels, centers = cv2.kmeans(feature_img, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    color_mask = centers[labels.flatten()]
    color_mask = color_mask.reshape((rows, cols, chs))
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)
    (thresh, mask) = cv2.threshold(color_mask, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask[mask==0] =1
    mask[mask==255] = 0
    contours = mask2contours(mask, rows, cols)
    
    return contours
    
