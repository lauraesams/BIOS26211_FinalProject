from mnist import MNIST
import numpy as np
from sklearn.svm import LinearSVC,SVC
from time import time
import json

def rebuild(frV1):
    print(np.shape(frV1))
    
def startEvaluate(train_data,test_data,kind):

    mndata = MNIST('samples')
    train_img, train_labl = mndata.load_training()
    test_img, test_label = mndata.load_testing()
    n_classes = 10


    clf = LinearSVC(C=1.0,multi_class='ovr',verbose=0)#(multi_class='crammer_singer') # SVC(kernel='rbf') #

    # rebuild the activity vectors
    #rebuild(frV1)

    print('start fitting')    
    t1 = time()
    clf.fit(train_data,train_labl)
    t2 = time()
    print(t2-t1)
    
    print('make some predictions')
    n = 10000
    x_val = test_data#.T #np.asarray(test_img)
    p = clf.predict(x_val[0:n,:])
    t = test_label[0:n]

    #print(p)
    #print(t)
    print(np.sum((p==t)*1))

    s = (np.sum((p==t)*1))/float(n) * 100.

    print(s)
    text = {'Percent right predictions':s}
    json.dump(text,open('acc_'+kind+'.txt','w'))    


def main():
    # Evaluate E1
    frV1 = np.load('./output/frE1_train.npy')
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1))
    frV1_train = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1_train[i,:] = np.reshape(frV1[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV1_train))

    frV1_train /=np.max(frV1_train)
    frV1_train -= np.mean(frV1_train)
    
    frV1 = np.load('./output/frE1_test.npy')

    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1))
    frV1_test = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1_test[i,:] = np.reshape(frV1[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV1_test))

    frV1_test /=np.max(frV1_test)
    frV1_test -= np.mean(frV1_test)

    startEvaluate(frV1_train,frV1_test,'E1')
    #--------------------------------------------------
    # Evaluate I1
    frV1 = np.load('./output/frIL1_train.npy')
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1))
    frV1_train = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1_train[i,:] = np.reshape(frV1[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV1_train))

    frV1_train /=np.max(frV1_train)
    frV1_train -= np.mean(frV1_train)
    
    frV1 = np.load('./output/frIL1_test.npy')

    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1))
    frV1_test = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1_test[i,:] = np.reshape(frV1[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV1_test))

    frV1_test /=np.max(frV1_test)
    frV1_test -= np.mean(frV1_test)

    startEvaluate(frV1_train,frV1_test,'IL1')
    #--------------------------------------------------
    #Evalutate E2 

    frV2 = np.load('./output/frE2_train.npy')
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV2))
    frV2_train = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV2_train[i,:] = np.reshape(frV2[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV2_train))

    frV2_train /=np.max(frV2_train)
    frV2_train -= np.mean(frV2_train)
    
    frV2 = np.load('./output/frE2_test.npy')

    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV2))
    frV2_test = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV2_test[i,:] = np.reshape(frV2[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV2_test))

    frV2_test /=np.max(frV2_test)
    frV2_test -= np.mean(frV2_test)

    startEvaluate(frV2_train,frV2_test,'E2')
    #--------------------------------------------------
    #Evalutate I2

    frV2 = np.load('./output/frIL2_train.npy')
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV2))
    frV2_train = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV2_train[i,:] = np.reshape(frV2[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV2_train))

    frV2_train /=np.max(frV2_train)
    frV2_train -= np.mean(frV2_train)
    
    frV2 = np.load('./output/frIL2_test.npy')

    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV2))
    frV2_test = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV2_test[i,:] = np.reshape(frV2[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV2_test))

    frV2_test /=np.max(frV2_test)
    frV2_test -= np.mean(frV2_test)

    startEvaluate(frV2_train,frV2_test,'IL2')
if __name__ == "__main__":
    main()
