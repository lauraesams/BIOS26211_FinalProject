import numpy as np
from sklearn.svm import LinearSVC,SVC
from time import time
import json
import scipy.io as sio

def loadInput():
    data_train = sio.loadmat('./samples/train_32x32.mat')
    data_test = sio.loadmat('./samples/test_32x32.mat')

    return(data_train,data_test)
#------------------------------------------------------------------------------

def fitt(fr_train,fr_test,kind):

    data_train,data_test = loadInput()
    train_img = data_train['X']
    train_label = np.squeeze(data_train['y'])
    test_img = data_test['X']
    test_label = np.squeeze(data_test['y'])

    n_classes = 10

    nbr_cells,nbr_elem,nbr_patches = (np.shape(fr_train))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(fr_train[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV1))

    frV1 /=np.max(frV1)    

    x = frV1

    n_ele = 40000
    
    clf = LinearSVC(C=10.0,multi_class='ovr',verbose=0,max_iter=5000)

    print('start fitting')    
    t1 = time()
    clf.fit(x[0:n_ele],train_label[0:n_ele])
    t2 = time()
    print(t2-t1)



    nbr_cells,nbr_elem,nbr_patches = (np.shape(fr_test))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(fr_test[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV1))

    frV1 /=np.max(frV1)

    print('make some predictions')
    n = 10000
    x_val = frV1
    p = clf.predict(x_val[0:n,:])
    t = test_label[0:n]

    s = (np.sum((p==t)*1))/float(n) * 100.

    print(s)
    text = {'Percent right predictions':s}
    json.dump(text,open('acc_'+kind+'.txt','w'))    

#------------------------------------------------------------------------------

if __name__ == "__main__":

    fr_train = np.load('./output/frE1_train.npy')    
    fr_test = np.load('./output/frE1_test.npy')
    kind = 'E1'
    fitt(fr_train,fr_test,kind)

    fr_train = np.load('./output/frE2_train.npy')    
    fr_test = np.load('./output/frE2_test.npy')
    kind = 'E2'
    fitt(fr_train,fr_test,kind)

    fr_train = np.load('./output/frIL1_train.npy')    
    fr_test = np.load('./output/frIL1_test.npy')
    kind = 'IL1'
    fitt(fr_train,fr_test,kind)

    fr_train = np.load('./output/frIL2_train.npy')    
    fr_test = np.load('./output/frIL2_test.npy')
    kind = 'IL2'
    fitt(fr_train,fr_test,kind)
