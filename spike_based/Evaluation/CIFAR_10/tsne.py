import matplotlib as mp
mp.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from sklearn import manifold
from time import time 

#------------------------------------------------------------
# load and unpickle the CIFAR_10 Dataset as mention on the
# webpage
def unpickle(file):
    import pickle as cPickle
    with open(file,'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict
#----------------------------------------------------------
def loadInput():

    test_batch = unpickle('./samples/test_batch')
    test_data = test_batch['data']
    test_labels = test_batch['labels']

    return(test_labels)
#----------------------------------------------------------

def calcTSNE(fr,labels,layer_name):
    print('Start T-SNE for %s'%layer_name)

    n_classes = np.max(labels)+1

    n_stim,n_activity = np.shape(fr)
    n_testStim = 5000
    perplexity = 50#200

    tsne = manifold.TSNE(n_components=2, init='random',learning_rate = 200, random_state=0, perplexity = perplexity,method='barnes_hut')

    X = fr[0:n_testStim,:]
    y = labels[0:n_testStim]
    print(np.shape(y))
    fitt = tsne.fit_transform(X)
    print(np.shape(fitt))
    target_ids = range(0,n_classes)

    plt.figure(figsize=(8, 8))
    colors = 'tomato', 'seagreen', 'steelblue', 'cyan', 'saddlebrown', 'y', 'k', 'slategray', 'orange', 'purple'
    cmap = mp.cm.get_cmap('gist_rainbow')
    norm =mp.colors.Normalize(vmin = 0, vmax = n_classes)
    #colors = cmap(norm(target_ids))
    labels = ['%i'%i for i in range(0,n_classes) ]

    for i, c, label in zip(target_ids, colors, labels):
        plt.scatter(fitt[y == i, 0], fitt[y == i, 1], color=c, label=label)
    plt.axis('off')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5,fontsize=15)
    plt.savefig('TSNE_scatter%s.png'%(layer_name),bbox_inches='tight')


def startTSNE():

    test_labels = loadInput()
    test_labels = np.asarray(test_labels)

    ## L4 pyr ##    
    frV1_part = np.load('./output/frE1_test.npy') # load the activity vectors on the test-set
    # recreate the complete response vector
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(frV1_part[:,i,:],(nbr_cells*nbr_patches))
    frV1 /= np.max(frV1)

    calcTSNE(frV1,test_labels,'E1')


    ##L4 inhib ##
    frV1_part = np.load('./output/frIL1_test.npy') # load the activity vectors on the test-set
    # recreate the complete response vector
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(frV1_part[:,i,:],(nbr_cells*nbr_patches))
    frV1 /= np.max(frV1)

    calcTSNE(frV1,test_labels,'I1')


    ##L2/3 Pyr ##
    frV1_part = np.load('./output/frE2_test.npy') # load the activity vectors on the test-set
    # recreate the complete response vector
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(frV1_part[:,i,:],(nbr_cells*nbr_patches))
    frV1 /= np.max(frV1)

    calcTSNE(frV1,test_labels,'E2')

    ##L2/3 inib ##
    frV1_part = np.load('./output/frIL2_test.npy') # load the activity vectors on the test-set
    # recreate the complete response vector
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(frV1_part[:,i,:],(nbr_cells*nbr_patches))
    frV1 /= np.max(frV1)

    calcTSNE(frV1,test_labels,'I2')

if __name__ == "__main__":

    startTSNE()

