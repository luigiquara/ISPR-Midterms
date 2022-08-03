from skimage import segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt

def compute_slic(images, n_segments, compactness):
    labels_slic = {}

    for key in images:
        labels_slic[key] = segmentation.slic(images[key], n_segments = n_segments, compactness = compactness)
    
    return labels_slic


def compute_ncuts(images, labels_slic):
    labels_ncuts = {}

    for key in images:
        rag = graph.rag_mean_color(images[key], labels_slic[key], mode='similarity')
        labels_ncuts[key] = graph.cut_normalized(labels_slic[key], rag)

    return labels_ncuts


def plot(images, rows, columns, title):
    fig = plt.figure(figsize=(20,20))
    plt.suptitle(title)

    i=1

    for key in sorted(images):
        if i<columns*rows+1:
            print(key)
            fig.add_subplot(rows, columns, i)
            plt.imshow(images[key])
            plt.axis('off')
            i += 1
        else:
            exit