import numpy as np
from sklearn.svm import LinearSVC,SVC
from sklearn.externals import joblib
from time import time
import json

def unpickle(file):
    import _pickle as cPickle # fix for use in python 3.X (!) -> will not work in Python 2.7 !
    with open(file,'rb') as fo:
        dict = cPickle.load(fo,encoding='latin1')
    return dict
#----------------------------------------------------------
def loadTestData():

    test_batch = unpickle('./samples/test_batch')
    test_data = test_batch['data']
    test_labels = test_batch['labels']

    return(test_data,test_labels)
    
#----------------------------------------------------------
def pred_inh():

    n_classes = 10

    data_test,label_test = loadTestData()

    frV1_s_part = np.load('./output/frIL1_test.npy')
    frV1_c_part = np.load('./output/frIL2_test.npy')

    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_s_part))
    frV1_s = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    frV1_c = np.zeros((nbr_elem,nbr_cells*nbr_patches))

    for i in range(nbr_elem):
        frV1_s[i,:] = np.reshape(frV1_s_part[:,i,:],(nbr_cells*nbr_patches))
        frV1_c[i,:] = np.reshape(frV1_c_part[:,i,:],(nbr_cells*nbr_patches))

    frV1_s /=np.max(frV1_s)
    frV1_s -= np.mean(frV1_s)

    frV1_c /=np.max(frV1_c)
    frV1_c -= np.mean(frV1_c)

    # predict on IL1 cells
    clf = joblib.load('output/svm_IL1.joblib')
    n = 1000
    p = clf.predict(frV1_s[0:n,:])
    t = label_test[0:n]

    s = (np.sum((p==t)*1))/float(n) * 100.

    print(s)
    text = {'Percent right predictions':s}
    json.dump(text,open('acc_IL1.txt','w'))    

    # predict on IL2 cells
    clf = joblib.load('output/svm_IL2.joblib')
    n = 1000
    p = clf.predict(frV1_c[0:n,:])
    t = label_test[0:n]

    s = (np.sum((p==t)*1))/float(n) * 100.

    print(s)
    text = {'Percent right predictions':s}
    json.dump(text,open('acc_IL2.txt','w'))


def pred_exc():

    n_classes = 10

    data_test,label_test = loadTestData()

    frV1_s_part = np.load('./output/frE1_test.npy')
    frV1_c_part = np.load('./output/frE2_test.npy')

    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_s_part))
    frV1_s = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    frV1_c = np.zeros((nbr_elem,nbr_cells*nbr_patches))

    for i in range(nbr_elem):
        frV1_s[i,:] = np.reshape(frV1_s_part[:,i,:],(nbr_cells*nbr_patches))
        frV1_c[i,:] = np.reshape(frV1_c_part[:,i,:],(nbr_cells*nbr_patches))

    frV1_s /=np.max(frV1_s)
    frV1_s -= np.mean(frV1_s)

    frV1_c /=np.max(frV1_c)
    frV1_c -= np.mean(frV1_c)

    # predict on E1 cells
    clf = joblib.load('output/svm_E1.joblib')
    n = 1000
    p = clf.predict(frV1_s[0:n,:])
    t = label_test[0:n]

    s = (np.sum((p==t)*1))/float(n) * 100.

    print(s)
    text = {'Percent right predictions':s}
    json.dump(text,open('acc_E1.txt','w'))    

    # predict on E2 cells
    clf = joblib.load('output/svm_E2.joblib')
    n = 1000
    p = clf.predict(frV1_c[0:n,:])
    t = label_test[0:n]

    s = (np.sum((p==t)*1))/float(n) * 100.

    print(s)
    text = {'Percent right predictions':s}
    json.dump(text,open('acc_E2.txt','w'))

if __name__ == "__main__":
    pred_exc()
    pred_inh()
