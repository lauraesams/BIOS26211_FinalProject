import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST
import scipy.spatial.distance as distance

####
# Function to calcuate a representational dissimilarity matrices (RDM) (see Kriegeskorte et al. (2008) Frontiers)
####

def calcDisVal(v1,v2,formular):
    # formular define the formular to calculate the value between v1 and v2
    # 0: 1 - Pearson Correlation
    # 1: Euclidean Distance 
    # 2: 1 - cos(V1,v2)

    if formular == 0:
        return(1 - np.corrcoef(v1,v2)[0,1])
    elif formular == 1:
        return(np.linalg.norm(v1 - v2))
    elif formular == 2:
        return(distance.cosine(v1,v2))
    else:
        print('False formular value')
        return(0)


def calcRDM(frV,l_idx,layer_name):
    print('Calculate RDM on the response vectors of layer %s.'%(layer_name))

    n_samples_total, n_vectors = np.shape(frV)
    n_classes, n_examples = np.shape(l_idx)

    fr_M = np.zeros((n_classes* n_examples,n_vectors))

    # get n examples per class
    # get the image and the activity vector
    cnt = 0
    for n in range(n_classes):
        for e in range(n_examples):
            fr_M[cnt] = frV[int(l_idx[n,e])]
            cnt +=1


    # calc the values for the RDM (1-corr(X,Y))
    corrM = np.zeros((n_classes*n_examples,n_classes*n_examples))
    for x in range(n_classes*n_examples):
        v1 = fr_M[x]
        for y in range(n_classes*n_examples):
            v2 = fr_M[y]
            corrM[x,y] = calcDisVal(v1,v2,2)

    plt.figure()
    plt.imshow(corrM,cmap='jet', interpolation='none',vmin=0.0, vmax=1.)
    plt.xlabel('class index')
    plt.ylabel('class index')
    plt.xticks(np.linspace(n_examples/2 ,n_classes*n_examples-n_examples/2-1,5), np.linspace(5,45,5,dtype='int16') )
    #plt.yticks(np.linspace(n_examples/2 ,n_classes*n_examples-n_examples/2-1,5), np.linspace(5,45,5,dtype='int16') )
    plt.yticks(np.linspace(5*n_examples-n_examples/2 ,45*n_examples-n_examples/2-1,5), np.linspace(5,45,5,dtype='int16'))
    plt.colorbar()
    plt.savefig('RDM_'+layer_name+'.png',bbox_inches='tight',dpi=300)


def calcRDM_imgs(test_img,l_idx):
    print('Calculate RDM for the Images')
    n_samples_total, n_vectors = np.shape(test_img)
    n_classes, n_examples = np.shape(l_idx)

    test_img = test_img/np.max(test_img)

    img_M = np.zeros((n_classes* n_examples,n_vectors))
    cnt = 0
    for n in range(n_classes):
        for e in range(n_examples):
            img_M[cnt] = test_img[int(l_idx[n,e])]
            cnt +=1

    # calc the values for the RDM (1-corr(X,Y))
    corrM = np.zeros((n_classes*n_examples,n_classes*n_examples))
    for x in range(n_classes*n_examples):
        v1 = img_M[x]
        for y in range(n_classes*n_examples):
            v2 = img_M[y]
            corrM[x,y] =calcDisVal(v1,v2,2)

    plt.figure()
    plt.imshow(corrM,cmap='jet', interpolation='none',vmin=0.0, vmax=1)
    plt.xlabel('class index')
    plt.ylabel('class index')
    plt.xticks(np.linspace(n_examples/2 ,n_classes*n_examples-n_examples/2-1,5), np.linspace(5,45,5,dtype='int16') )
    plt.yticks(np.linspace(n_examples/2 ,n_classes*n_examples-n_examples/2-1,5), np.linspace(5,45,5,dtype='int16') )
    plt.colorbar()
    plt.savefig('RDM_IMG.png',bbox_inches='tight')


def startRDM():

    """
    Take n randomly chosen examples from each class, get the corresponding response vectors and creat the matrix
    """
    np.random.seed(314)
    n_examples = 100

    mndata = MNIST('emnist-balance')
    test_img, test_label = mndata.load_testing()
    test_label = np.asarray(test_label)
    n_classes = np.max(test_label)+1
    print('# classes: ',n_classes)    

    #create a list of indices to adress the random samples
    l_idx = np.zeros((n_classes,n_examples))
    for c in range(n_classes):
        idx_c = np.where(test_label == c)[0]
        l_idx[c] = idx_c[0:n_examples]

    # cacl RDM on the vector presentations of the images of the test set
    #calcRDM_imgs(test_img,l_idx)


    ## L4 Pyr ##    
    frV1_part = np.load('./output/frV1_test.npy') # load the activity vectors on the test-set
    # recreate the complete response vector
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(frV1_part[:,i,:],(nbr_cells*nbr_patches))
    frV1 /= np.max(frV1)

    calcRDM(frV1,l_idx,'E1')


    ##L4 inhib ##
    frV1_part = np.load('./output/frIL1_test.npy') # load the activity vectors on the test-set
    # recreate the complete response vector
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(frV1_part[:,i,:],(nbr_cells*nbr_patches))
    frV1 /= np.max(frV1)
    
    calcRDM(frV1,l_idx,'I1')


    ##L2/3 Pyr ##
    frV1_part = np.load('./output/frV2_test.npy') # load the activity vectors on the test-set
    # recreate the complete response vector
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(frV1_part[:,i,:],(nbr_cells*nbr_patches))
    frV1 /= np.max(frV1)
    
    calcRDM(frV1,l_idx,'E2')

    ##L2/3 inib ##
    frV1_part = np.load('./output/frIL2_test.npy') # load the activity vectors on the test-set
    # recreate the complete response vector
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(frV1_part[:,i,:],(nbr_cells*nbr_patches))
    frV1 /= np.max(frV1)
    
    calcRDM(frV1,l_idx,'I2')

if __name__ == "__main__":

    startRDM()
    

