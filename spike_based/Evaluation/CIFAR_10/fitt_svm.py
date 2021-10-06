import numpy as np
from sklearn.svm import LinearSVC,SVC
from time import time
import json
from sklearn.externals import joblib
#------------------------------------------------------------
# load and unpickle the CIFAR_10 Dataset as mention on the
# webpage
def unpickle(file):
    import _pickle as cPickle # fix for use in python 3.X (!) -> will not work in Python 2.7 !
    with open(file,'rb') as fo:
        dict = cPickle.load(fo,encoding='latin1')
    return dict
#----------------------------------------------------------
def loadInput():
    batch1 = './samples/data_batch_1'
    batch2 = './samples/data_batch_2'
    batch3 = './samples/data_batch_3'
    batch4 = './samples/data_batch_4'
    batch5 = './samples/data_batch_5'
    
    batch1 = unpickle(batch1)
    batch2 = unpickle(batch2)
    batch3 = unpickle(batch3)
    batch4 = unpickle(batch4)
    batch5 = unpickle(batch5)

    data_train = batch1['data']
    
    data_train = np.append(data_train,batch2['data'],axis=0)
    data_train = np.append(data_train,batch3['data'],axis=0)
    data_train = np.append(data_train,batch4['data'],axis=0)
    data_train = np.append(data_train,batch5['data'],axis=0)
    
    label_train = batch1['labels']
    label_train = np.append(label_train,batch2['labels'])
    label_train = np.append(label_train,batch3['labels'])
    label_train = np.append(label_train,batch4['labels'])
    label_train = np.append(label_train,batch5['labels'])

    test_batch = unpickle('./samples/test_batch')
    test_data = test_batch['data']
    test_labels = test_batch['labels']

    return(data_train,label_train,test_data,test_labels)
#----------------------------------------------------------


def fit_inh():
    data_train,label_train,data_test,label_test = loadInput()

    n_classes = 10
    ####
    # fitt on I1 activity
    ####
    frV1_part = np.load('./output/frIL1_train.npy')
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(frV1_part[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV1))

    frV1 /=np.max(frV1)
    frV1 -= np.mean(frV1)
    

    x = frV1#np.asarray(train_img)#

    clf = LinearSVC(C=1.0,multi_class='ovr',verbose=0)#(multi_class='crammer_singer') # SVC(kernel='rbf') #

    # rebuild the activity vectors
    #rebuild(frV1)

    print('start fitting')    
    t1 = time()
    clf.fit(x,label_train)
    t2 = time()
    print(t2-t1)

    s = joblib.dump(clf,'output/svm_IL1.joblib')

    ####
    # fitt on I2 activity
    ####
    frV1_part = np.load('./output/frIL2_train.npy')
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(frV1_part[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV1))

    frV1 /=np.max(frV1)
    frV1 -= np.mean(frV1)
    

    x = frV1#np.asarray(train_img)#

    clf = LinearSVC(C=1.0,multi_class='ovr',verbose=0)#(multi_class='crammer_singer') # SVC(kernel='rbf') #

    # rebuild the activity vectors
    #rebuild(frV1)

    print('start fitting')    
    t1 = time()
    clf.fit(x,label_train)
    t2 = time()
    print(t2-t1)

    s = joblib.dump(clf,'output/svm_IL2.joblib')


def fit_exc():

    data_train,label_train,data_test,label_test = loadInput()

    n_classes = 10
    ####
    # fitt on E1 activity
    ####
    frV1_part = np.load('./output/frE1_train.npy')
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(frV1_part[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV1))

    frV1 /=np.max(frV1)
    frV1 -= np.mean(frV1)
    

    x = frV1#np.asarray(train_img)#

    clf = LinearSVC(C=1.0,multi_class='ovr',verbose=0)#(multi_class='crammer_singer') # SVC(kernel='rbf') #

    # rebuild the activity vectors
    #rebuild(frV1)

    print('start fitting')    
    t1 = time()
    clf.fit(x,label_train)
    t2 = time()
    print(t2-t1)

    s = joblib.dump(clf,'output/svm_E1.joblib')

    ####
    # fitt on E2 activityy
    ####
    frV1_part = np.load('./output/frE2_train.npy')
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_part))
    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(frV1_part[:,i,:],(nbr_cells*nbr_patches))
    print(np.shape(frV1))

    frV1 /=np.max(frV1)
    frV1 -= np.mean(frV1)
    

    x = frV1#np.asarray(train_img)#

    clf = LinearSVC(C=1.0,multi_class='ovr',verbose=0)#(multi_class='crammer_singer') # SVC(kernel='rbf') #

    # rebuild the activity vectors
    #rebuild(frV1)

    print('start fitting')    
    t1 = time()
    clf.fit(x,label_train)
    t2 = time()
    print(t2-t1)

    s = joblib.dump(clf,'output/svm_E2.joblib')

if __name__ == "__main__":
    fit_exc()
    fit_inh()
