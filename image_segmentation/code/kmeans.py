import cv2
import numpy as np

#Compute image segmentation using k-means algorithm
def compute_kmeans(images, k=3, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                   attempts=10, flag=cv2.KMEANS_RANDOM_CENTERS):
    centers = {}
    labels = {}
    for key in images:
        img = images[key].reshape((-1,3))
        img = np.float32(img)

        _,labels[key],centers[key] = cv2.kmeans(img, k, None, criteria, attempts, flag)
    
    return labels, centers




    