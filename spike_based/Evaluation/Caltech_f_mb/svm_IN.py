import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import f1_score
from time import time
import json
from matplotlib.colors import LogNorm

def predict_c():
    print('Start on Layer 2/3')
    n_classes = 2
    n_instances = 200
    n_faces = 435
    n_bikes = 798

    s_train = n_instances*2
    s_test = (n_faces - n_instances) + (n_bikes - n_instances)

    frV1_part = np.load('./output/frIL2_data_set.npy')
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_part))

    frV1_part /=np.max(frV1_part)

    inpt = np.load('./input/caltech_f_b.npy')
    n_ele,h,w = np.shape(inpt)


    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(frV1_part[:,i,:],(nbr_cells*nbr_patches))

    inpt_f = inpt[0:n_faces]
    inpt_b = inpt[n_faces:]


    # split into face and bike set
    data_set_f = frV1[0:n_faces,:]
    data_set_b = frV1[n_faces:,:]

    # fit n times the train_set and test on test set
    n_rep = 20
    
    acc = np.zeros(n_rep)

    for i in range(20):
        # pick 200 randomly chosen instances from the faces and bikes for the training set
        rand_idx_f = np.sort(np.random.choice(n_faces,size=n_instances,replace=False))
        rand_idx_b = np.sort(np.random.choice(n_bikes,size=n_instances,replace=False))

        train_set = np.zeros((s_train,nbr_cells*nbr_patches))
        train_set[0:n_instances,:] = data_set_f[rand_idx_f,:]
        train_set[n_instances:,:] = data_set_b[rand_idx_b,:]

        train_label = np.zeros(s_train)
        train_label[n_instances:] = 1 # 0 for faces, 1 for bikes
        

        train_img = np.zeros((s_train,h,w))
        train_img[0:n_instances] = inpt_f[rand_idx_f]
        train_img[n_instances:] = inpt_b[rand_idx_b]


        # create the test set with the rest of the instances
        test_set = np.zeros((s_test,nbr_cells*nbr_patches))
        test_img = np.zeros((s_test,h,w))

        idx_f = np.linspace(0,n_faces-1,n_faces,dtype='int32')
        idx_f[rand_idx_f] = -1 # set all indices there are in the train set -1 to sort them out
        test_idx_f = idx_f[idx_f > -1] # indecies for the test-set

        idx_b = np.linspace(0,n_bikes-1,n_bikes,dtype='int32')
        idx_b[rand_idx_b] = -1 # set all indices there are in the train set -1 to sort them out
        test_idx_b = idx_b[idx_b > -1] # indecies for the test-set


        test_set[0:(n_faces-n_instances)] = data_set_f[test_idx_f]
        test_set[(n_faces-n_instances):,:] = data_set_b[test_idx_b]

        test_label = np.zeros(s_test)
        test_label[(n_faces-n_instances):] = 1 # 0 for faces and 1 for bikes

        test_img[0:(n_faces-n_instances)] = inpt_f[test_idx_f]
        test_img[(n_faces-n_instances):] = inpt_b[test_idx_b]


        #shuffle train and test set
        idx_train = np.linspace(0,s_train-1,s_train,dtype='int32')
        np.random.shuffle(idx_train)
        train_set = train_set[idx_train,:]
        train_label = train_label[idx_train]
        train_img = train_img[idx_train]

        idx_test = np.linspace(0,s_test-1,s_test,dtype='int32')
        np.random.shuffle(idx_test)
        test_set = test_set[idx_test,:]
        test_label = test_label[idx_test]
        test_img = test_img[idx_test]   


        # create a new svm
        clf = LinearSVC(C=1,multi_class='ovr',verbose=0)#,max_iter=20000)#(multi_class='crammer_singer') # SVC(kernel='rbf') #

        # rebuild the activity vectors
        #rebuild(frV1)

        #print('start fitting')    
        t1 = time()
        clf.fit(train_set,train_label)
        t2 = time()
        #print(t2-t1)

        #print('make some predictions')
        p = clf.predict(test_set)
        t = test_label
        t = np.asarray(t)
        #print(t)
        s = (np.sum((p==t)*1))/float(s_test) * 100.
        acc[i] = s
    
    text = {'Percent right predictions (mean)':np.mean(acc),'STD of predictions':np.std(acc,ddof=1)}
    json.dump(text,open('acc_IL2.txt','w'))

def predict_s():
    print('Start on V1 Layer 4')
    n_classes = 2
    n_instances = 200
    n_faces = 435
    n_bikes = 798

    s_train = n_instances*2
    s_test = (n_faces - n_instances) + (n_bikes - n_instances)

    frV1_part = np.load('./output/frIL1_data_set.npy')
    nbr_cells,nbr_elem,nbr_patches = (np.shape(frV1_part))

    frV1_part /=np.max(frV1_part)

    inpt = np.load('./input/caltech_f_b.npy')
    n_ele,h,w = np.shape(inpt)


    frV1 = np.zeros((nbr_elem,nbr_cells*nbr_patches))
    for i in range(nbr_elem):
        frV1[i,:] = np.reshape(frV1_part[:,i,:],(nbr_cells*nbr_patches))


    inpt_f = inpt[0:n_faces]
    inpt_b = inpt[n_faces:]

    # split into face and bike set
    data_set_f = frV1[0:n_faces,:]
    data_set_b = frV1[n_faces:,:]

    # fit n times the train_set and test on test set
    n_rep = 20
    
    acc = np.zeros(n_rep)

    for i in range(20):
        # pick 200 randomly chosen instances from the faces and bikes for the training set
        rand_idx_f = np.sort(np.random.choice(n_faces,size=n_instances,replace=False))
        rand_idx_b = np.sort(np.random.choice(n_bikes,size=n_instances,replace=False))

        train_set = np.zeros((s_train,nbr_cells*nbr_patches))
        train_set[0:n_instances,:] = data_set_f[rand_idx_f,:]
        train_set[n_instances:,:] = data_set_b[rand_idx_b,:]

        train_label = np.zeros(s_train)
        train_label[n_instances:] = 1 # 0 for faces, 1 for bikes
        

        train_img = np.zeros((s_train,h,w))
        train_img[0:n_instances] = inpt_f[rand_idx_f]
        train_img[n_instances:] = inpt_b[rand_idx_b]




        # create the test set with the rest of the instances
        test_set = np.zeros((s_test,nbr_cells*nbr_patches))
        test_img = np.zeros((s_test,h,w))

        idx_f = np.linspace(0,n_faces-1,n_faces,dtype='int32')
        idx_f[rand_idx_f] = -1 # set all indices there are in the train set -1 to sort them out
        test_idx_f = idx_f[idx_f > -1] # indecies for the test-set

        idx_b = np.linspace(0,n_bikes-1,n_bikes,dtype='int32')
        idx_b[rand_idx_b] = -1 # set all indices there are in the train set -1 to sort them out
        test_idx_b = idx_b[idx_b > -1] # indecies for the test-set


        test_set[0:(n_faces-n_instances)] = data_set_f[test_idx_f]
        test_set[(n_faces-n_instances):,:] = data_set_b[test_idx_b]

        test_label = np.zeros(s_test)
        test_label[(n_faces-n_instances):] = 1 # 0 for faces and 1 for bikes

        test_img[0:(n_faces-n_instances)] = inpt_f[test_idx_f]
        test_img[(n_faces-n_instances):] = inpt_b[test_idx_b]



        #shuffle train and test set
        idx_train = np.linspace(0,s_train-1,s_train,dtype='int32')
        np.random.shuffle(idx_train)
        train_set = train_set[idx_train,:]
        train_label = train_label[idx_train]
        train_img = train_img[idx_train]

        idx_test = np.linspace(0,s_test-1,s_test,dtype='int32')
        np.random.shuffle(idx_test)
        test_set = test_set[idx_test,:]
        test_label = test_label[idx_test]
        test_img = test_img[idx_test]   


        # create a new svm
        clf = LinearSVC(C=1,multi_class='ovr',verbose=0)#,max_iter=20000)#(multi_class='crammer_singer') # SVC(kernel='rbf') #

        # rebuild the activity vectors
        #rebuild(frV1)

        #print('start fitting')    
        t1 = time()
        clf.fit(train_set,train_label)
        t2 = time()
        #print(t2-t1)

        #print('make some predictions')
        p = clf.predict(test_set)
        t = test_label
        t = np.asarray(t)
        #print(t)
        s = (np.sum((p==t)*1))/float(s_test) * 100.
        acc[i] = s
    
    text = {'Percent right predictions (mean)':np.mean(acc),'STD of predictions':np.std(acc,ddof=1)}
    json.dump(text,open('acc_IL1.txt','w'))    

if __name__ == "__main__":
    predict_s()
    predict_c()
