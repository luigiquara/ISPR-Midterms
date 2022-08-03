import os
from datetime import datetime
from skimage import io
from kmeans import compute_kmeans
from ncuts import plot
import cv2
import numpy as np
from matplotlib import pyplot as plt

data_dir = 'MSRC_ObjCategImageDatabase_v1'
images = {}
centers = {}
labels = {}
out_imgs = {}

k = 3
attempts = 10
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flag = cv2.KMEANS_RANDOM_CENTERS

# Load dataset
for filename in os.listdir(data_dir):
    if(filename.startswith('1_21') and filename.endswith('_s.bmp')):
        name = os.path.join(data_dir, filename)
        images[name] = io.imread(name)

labels, centers = compute_kmeans(images, k, criteria, attempts, flag)

for key in centers:
    centers[key] = np.uint8(centers[key])
    out_imgs[key] = centers[key][labels[key].flatten()] 
    out_imgs[key] = out_imgs[key].reshape((images[key].shape))

filename = 'results/kmeans/'+str(k)+'-means '+str(datetime.now())+'.png'
plot(out_imgs, rows=5, columns=6, title='K-means Segmentation', filename=filename)