import matplotlib as mp
mp.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from sklearn import manifold
from time import time 
from mnist import MNIST

def createLables(idx_list):
    label_list = ['0','1','2','3','4','5','6','7','8','9',
                  'A','B','C','D','E','F','G','H','I','J',
                  'K','L','M','N','O','P','Q','R','S','T',
                  'U','V','W','X','Y','Z','a','b','d','e',
                  'f','g','h','n','q','r','t' ]
    label_list = np.asarray(label_list)
    return(label_list[idx_list])

def calcTSNE(fr,labels,idx_list,layer_name):
    print('Start T-SNE for %s'%layer_name)

    n_classes = np.max(labels)+1

    n_stim,n_activity = np.shape(fr)
    n_testStim = 5000
    n_stimClass = 400#int(n_testStim/10)
    perplexity = 50#200

    tsne = manifold.TSNE(n_components=2, init='random',learning_rate = 200, random_state=0, perplexity = perplexity,method='barnes_hut')

    # get only examples from the ten random chosen classes:

    X = np.reshape([fr[ np.where(labels == idx_list[c])[0] ] for c in range(len(idx_list)) ], (len(idx_list)*n_stimClass,n_activity))
    y = np.reshape([labels[ np.where(labels == idx_list[c])[0] ] for c in range(len(idx_list)) ], (len(idx_list)*n_stimClass))    

    print(np.shape(X))
    print('-------------')
    print(np.shape(y))
    fitt = tsne.fit_transform(X)
    print(np.shape(fitt))
    target_ids = idx_list #range(0,n_classes)

    plt.figure(figsize=(8, 8))
    colors = 'tomato', 'seagreen', 'steelblue', 'cyan', 'saddlebrown', 'y', 'k', 'slategray', 'orange', 'purple'
    cmap = mp.cm.get_cmap('gist_rainbow')
    norm =mp.colors.Normalize(vmin = 0, vmax = n_classes)
    #colors = cmap(norm(target_ids))
    labels = createLables(idx_list) #['%i'%i for i in range(0,n_classes) ]

    for i, c, label in zip(target_ids, colors, labels):
        plt.scatter(fitt[y == i, 0], fitt[y == i, 1], color=c, label=label)
    plt.axis('off')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5,fontsize=15)
    plt.savefig('TSNE_scatter%s.png'%(layer_name),bbox_inches='tight')


def startTSNE():

    mndata = MNIST('emnist-balance')
    test_img, test_labl = mndata.load_testing()
    test_labl = np.asarray(test_labl)

        
    n_classes = np.max(test_labl)+1
    # choose randomly 10 classes for the tsne
    idx_list = np.random.choice(n_classes,10,replace=False)    
    #print(idx_list)
    idx_list = np.asarray([15, 19, 33, 31, 43, 26,  3, 14,  8,  5])

    ## L4 Pyr ##    
    fr_part = np.load('./output/frV1_test.npy') # load the activity vectors on the test-set
    # recreate the complete response vector
    nbr_cells,nbr_elem,nbr_patches = (np.shape(fr_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(fr_part[:,i,:],(nbr_cells*nbr_patches))
    frV1 /= np.max(frV1)

    calcTSNE(frV1,test_labl,idx_list,'E1')

    return -1

    ##L4 inhib ##
    fr_part = np.load('./output/frIL1_test.npy') # load the activity vectors on the test-set
    # recreate the complete response vector
    nbr_cells,nbr_elem,nbr_patches = (np.shape(fr_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(fr_part[:,i,:],(nbr_cells*nbr_patches))
    frV1 /= np.max(frV1)

    calcTSNE(frV1,test_labl,idx_list,'I1')

    ##L2/3 Pyr ##
    fr_part = np.load('./output/frV2_test.npy') # load the activity vectors on the test-set
    # recreate the complete response vector
    nbr_cells,nbr_elem,nbr_patches = (np.shape(fr_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(fr_part[:,i,:],(nbr_cells*nbr_patches))
    frV1 /= np.max(frV1)

    calcTSNE(frV1,test_labl,idx_list,'E2')

    ##L2/3 inib ##
    fr_part = np.load('./output/frIL2_test.npy') # load the activity vectors on the test-set
    # recreate the complete response vector
    nbr_cells,nbr_elem,nbr_patches = (np.shape(fr_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(fr_part[:,i,:],(nbr_cells*nbr_patches))
    frV1 /= np.max(frV1)

    calcTSNE(frV1,test_labl,idx_list,'I2')


if __name__ == "__main__":

    startTSNE()
    
