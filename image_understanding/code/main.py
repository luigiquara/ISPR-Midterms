import os
import time
import cv2
import pickle
from sklearn.cluster import KMeans
import numpy as np
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt

def load_sets():
    data_dir = 'MSRC_ObjCategImageDatabase_v1'
    test_img = '15_s'
    images = {}
    trainset = {}
    testset= {}

    for filename in os.listdir(data_dir):
        if(filename.endswith('_s.bmp')):
            name = os.path.join(data_dir, filename)
            images[filename] = cv2.imread(name)

    for filename in images:
        if (test_img in filename): testset[filename] = images[filename]
        else: trainset[filename] = images[filename]
    
    return trainset, testset

#extract keypoints and descriptors for each image
def extract_sift_descriptor(set):
    kp = {}
    des = {}
    sift = cv2.xfeatures2d.SIFT_create()
    for key,img in set.items():
        kp[key], des[key] = sift.detectAndCompute(img, None)

    return kp, des

#learn the 500-dimensional codebook using k-means clustering
def k_means(samples, k = 500, att = 20):
    kmeans = KMeans(k, verbose = 1, n_init = att)
    kmeans.fit(samples)
    pickle.dump(kmeans, open('kmeans.pkl', 'wb'))

    return kmeans

#build histogram using the codebook
def build_histogram(descriptors, kmeans):
    hist = np.zeros(shape=len(kmeans.cluster_centers_))
    predictions = kmeans.predict(descriptors)
    for i in predictions:
        hist[i] += 1
    
    return hist, predictions

#get the set in the right format for gensim lda:

#Bag of Words representation
#list of lists - each list contains (word_id, word_freq) pairs for each visual word from the histograms
def format_corpus(set):
    total_bow = []
    for doc in set:
        single_doc_bow = []
        for id, freq in  enumerate(doc): single_doc_bow.append((id, freq))
        total_bow.append(single_doc_bow)
    
    return total_bow

#get the most relevant topic for each word
def topic_per_word(set_clustered):
    doc_topics = {}
    for key, img in set_clustered.items():
        doc_topics[key] = []
        for id, word in enumerate(img):
            r = lda.get_term_topics(word, minimum_probability = 0)
            max_topic = max(r, key = lambda k: k[1])
            doc_topics[key].append(max_topic[0])
    
    return doc_topics

def print_results(kp_set, doc_topics, set, topics_colors, check = True, path = './results_train'):
    for key in kp_set.keys():
    #using 2, 4, 8 subsets
        if(check):
            #if not (key.startswith('2_') or key.startswith('4_') or key.startswith('8_')):
                #continue
            if ('_5' not in key) and ('_13' not in key) and ('_17' not in key) and ('_28' not in key):
                continue
        
        for id, kp in enumerate(kp_set[key]):
            x = np.int(kp.pt[0])
            y = np.int(kp.pt[1])
            size = np.int(kp.size)
            color = topics_colors[doc_topics[key][id]]
            cv2.circle(set[key], (x,y), size, color, thickness = -1)  #with size parameter
            #cv2.circle(set[key], (x,y), radius = 5, color = color, thickness = -1)

        cv2.imwrite(path + '/' + key + '.png', set[key])

train_histograms = {}
train_clustered = {}
test_histograms = {}
test_clustered = {}
topics_colors =[]

#initialize colours from the MS dataset ClickMe.html
#topics_colors.append((0,0,0))
topics_colors.append((128,0,0))
topics_colors.append((0,128,0))
topics_colors.append((128,128,0))
topics_colors.append((0,0,128))
topics_colors.append((128,0,128))
topics_colors.append((0,128,128))
topics_colors.append((128,128,128))
topics_colors.append((64,0,0))
topics_colors.append((192,0,0))
topics_colors.append((64,128,0))
topics_colors.append((192,128,0))
topics_colors.append((64,0,128))
topics_colors.append((192,0,128))

trainset, testset = load_sets()

#extract keypoints and descriptors with SIFT
kp_train, des_train= extract_sift_descriptor(trainset)
kp_test, des_test = extract_sift_descriptor(testset)

#load the trained kmeans model
kmeans = pickle.load(open("kmeans.pkl", "rb"))

#train the kmeans model
#concatenated_des_train = np.vstack(value for value in des_train.values())
#kmeans = k_means(concatenated_des_train)

#build Bag Of Words model - each element is the frequency of the corrispondent cluster in the image
for key, value in des_train.items():
    hist, pred = build_histogram(value, kmeans)
    train_histograms[key] = hist
    train_clustered[key] = pred
train_bow = format_corpus(train_histograms.values())

for key, value in des_test.items():
    hist, pred = build_histogram(value, kmeans)
    test_histograms[key] = hist
    test_clustered[key] = pred
test_bow = format_corpus(test_histograms.values())

alpha = []
for i in range(9): alpha.append(0.001)
start = time.time()
iterations = 2000
passes = 2000
print(iterations)
print(passes)
#lda = LdaModel(train_bow, num_topics = 9, alpha = 'auto', eta = 'auto', iterations = iterations, passes = passes) 
print('elapsed:')
print(time.time() - start)
#lda.save('./9_auto_auto_2000iter/lda_model')
lda = LdaModel.load('./9_auto_auto_1000iter/lda_model')

#get the top scoring topic for each word
doc_topics_train = topic_per_word(train_clustered)
doc_topics_test = topic_per_word(test_clustered)

print_results(kp_train, doc_topics_train, trainset, topics_colors, check = False)
print_results(kp_test, doc_topics_test, testset, topics_colors, check = False, path = './results')