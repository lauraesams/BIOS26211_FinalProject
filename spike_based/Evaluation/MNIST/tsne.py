import matplotlib as mp
mp.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from sklearn import manifold
from time import time 
from mnist import MNIST


mndata = MNIST('samples')
train_img, train_labl = mndata.load_training()
train_labl = np.asarray(train_labl)

frV1_part = np.load('./output/frV1_trains.npy')

nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_part))
frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
for i in range(nbr_elem):
    frV1[i,:] = np.reshape(frV1_part[:,i,:],(nbr_cells*nbr_patches))
print(np.shape(frV1))

n_stim,n_activity = np.shape(frV1)
n_testStim = 1000
perplexity = 200

tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity = perplexity,method='exact')

X = frV1[0:n_testStim,:]
y = train_labl[0:n_testStim]
print(np.shape(y))
fitt = tsne.fit_transform(X)
print(np.shape(fitt))
target_ids = range(0,10)

plt.figure(figsize=(8, 8))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
labels = '0','1','2','3','4','5','6','7','8','9'

for i, c, label in zip(target_ids, colors, labels):
    plt.scatter(fitt[y == i, 0], fitt[y == i, 1], c=c, label=label)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.savefig('TSNE_scatter.png',bbox_inches='tight')
