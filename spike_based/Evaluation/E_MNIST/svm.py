import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt

from mnist import MNIST
import numpy as np
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from time import time
import json
from matplotlib.colors import LogNorm

    
def fitt(fr_train,fr_test,kind):

    mndata = MNIST('emnist-balance')
    train_img, train_labl = mndata.load_training()
    test_img, test_label = mndata.load_testing()


    n_classes = 47

    nbr_cells,nbr_elem,nbr_patches = (np.shape(fr_train))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(fr_train[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV1))



    frV1 /=np.max(frV1)
    x = frV1

    clf = LinearSVC(C=1,multi_class='ovr',verbose=0,max_iter=1000)#


    print('start fitting')    
    t1 = time()
    clf.fit(x,train_labl)
    t2 = time()
    print(t2-t1)


    nbr_cells,nbr_elem,nbr_patches = (np.shape(fr_test))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))

    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(fr_test[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV1))


    frV1 /=np.max(frV1)

    print('make some predictions')
    n = nbr_elem
    x_val = frV1
    p = clf.predict(x_val)
    t = test_label
    t = np.asarray(t)
    
    s = (np.sum((p==t)*1))/float(n) * 100.

    print(s)
    text = {'Percent right predictions':s}
    json.dump(text,open('acc_balance_'+kind+'.txt','w'))    

    f1_sc = f1_score(t,p,average=None)
    print(f1_sc)

    confM = np.zeros((n_classes,n_classes))
    for i in range(n_classes):
        idxT = np.asarray(np.where(t == i)[0])
        p_T = np.unique(p[idxT])# get the elements they are predicted
        for pred in p_T:
            s = np.sum((p[idxT]==pred)*1)
            confM[i,pred] = s

    plt.figure()
    plt.imshow(confM,cmap=plt.cm.get_cmap('inferno'),interpolation='none')
    plt.colorbar()    
    plt.savefig('confM_balance_'+kind+'.png')

if __name__ == "__main__":
    
    fr_train = np.load('./output/frE1_train.npy')
    fr_test = np.load('./output/frE1_test.npy')
    fitt(fr_train,fr_test,'E1')

    fr_train = np.load('./output/frE2_train.npy')
    fr_test = np.load('./output/frE2_test.npy')
    fitt(fr_train,fr_test,'E2')

    fr_train = np.load('./output/frIL1_train.npy')
    fr_test = np.load('./output/frIL1_test.npy')
    fitt(fr_train,fr_test,'IL1')

    fr_train = np.load('./output/frIL2_train.npy')
    fr_test = np.load('./output/frIL2_test.npy')
    fitt(fr_train,fr_test,'IL2')
