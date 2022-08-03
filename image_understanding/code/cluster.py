import pickle
import numpy as np
from sklearn.cluster import KMeans
from main import extract_sift_descriptor, load_sets, build_histogram
from lda import lda

kmeans = pickle.load(open("kmeans.pkl", "rb"))
trainset, testset = load_sets()

kp_train, des_train= extract_sift_descriptor(trainset)
kp_test, des_test = extract_sift_descriptor(testset)

train_histograms = []
test_histograms = []
for value in des_train.values():
    train_histograms.append(build_histogram(value, kmeans))
#for key, value in des_test.items():
#    test_histograms[key] = build_histogram(value, kmeans)

lda = lda(train_histograms)

