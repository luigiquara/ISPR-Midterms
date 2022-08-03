import os
from ncut import compute_slic, compute_ncuts, plot
from matplotlib import pyplot as plt
from skimage import io, color

data_dir = 'MSRC_ObjCategImageDatabase_v1'
images = {}
img_ncuts = {}

#DO NOT CHANGE!!
n_segments = 100
compactness = 40

# Load dataset
for filename in os.listdir(data_dir):
    if(filename.startswith('1') and filename.endswith('_s.bmp')):
        name = os.path.join(data_dir, filename)
        images[name] = io.imread(name)

labels_slic = compute_slic(images, n_segments, compactness)
labels_ncuts = compute_ncuts(images, labels_slic)

for key in labels_ncuts:
    img_ncuts[key] = color.label2rgb(labels_ncuts[key], images[key], kind = 'avg')

plot(img_ncuts, rows=5, columns=6)
plt.show()
