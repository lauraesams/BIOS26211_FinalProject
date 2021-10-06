import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
from mnist import MNIST
import numpy as np
from sklearn.svm import LinearSVC,SVC
from time import time
import json

def prepareSets(frE):

    n_neurons,n_classes,n_instances,n_images,n_patches = np.shape(frE)

    # rebuild the activity matrice to (class,instance,image,neurons*patches)    
    frV1 = np.zeros((n_classes,n_instances,n_images,n_neurons*n_patches))
    for c in range(n_classes):
        for i in range(n_instances):
            for j in range(n_images):
                frV1[c,i,j] = np.reshape(frE[:,c,i,j,:],n_neurons*n_patches)

    i_l = np.linspace(0,n_instances-1,n_instances,dtype='int32')
    np.random.shuffle(i_l)   


    train_set = frV1[:,i_l[0:5]]
    test_set = frV1[:,i_l[5:]]

    class_i = np.linspace(0,n_classes-1,n_classes,dtype='int32')
    
    train_labels = np.zeros((n_classes,5,n_images),dtype='int32')
    test_labels = np.zeros((n_classes,5,n_images),dtype='int32')
    for i in range(n_classes):
        train_labels[i,:,:]= class_i[i]
        test_labels[i,:,:]= class_i[i]

    return(train_set,test_set,train_labels,test_labels)


def analyse(frE,name):

    repeats = 20#100
    acc=np.zeros(repeats)

    for i in range(repeats):
        n_neurons,n_classes,n_instances,n_images,n_patches = np.shape(frE)

        train_set,test_set,train_labels,test_labels = prepareSets(frE)
        
        n_classes,n_instances_train,n_images,n_activities = np.shape(train_set)
        n_classes,n_instances_test,n_images,n_activities = np.shape(test_set)

        # reshape to a simple array and learn a svm with that
        train_set = np.reshape(train_set,(n_classes*n_instances_train*n_images,n_activities))
        train_labels = np.reshape(train_labels,(n_classes*n_instances_train*n_images))

        test_set = np.reshape(test_set,(n_classes*n_instances_test*n_images,n_activities))
        test_labels = np.reshape(test_labels,(n_classes*n_instances_test*n_images))

        clf = LinearSVC(C=0.1,penalty='l2',loss='squared_hinge',dual=False,multi_class='ovr',verbose=0,max_iter=10000)
        
        train_set /= np.max(train_set)


        print('start fitting')    
        t1 = time()
        clf.fit(train_set,train_labels)
        t2 = time()
        print('time: ',t2-t1)


        print('make some predictions')
        test_set /= np.max(test_set)
        p = clf.predict(test_set)
        
        n = len(test_labels)

        
        acc[i] = (np.sum((p==test_labels)*1))/float(n) * 100.
        print(acc[i])
        print('------------')
    
    plt.figure()
    plt.plot(acc,'o')
    plt.hlines(np.mean(acc),xmin=0,xmax=len(acc))
    plt.savefig('acc_'+name+'.png')


    np.save('acc_'+name,acc)

def main():
    frE = np.load('./output/frE1_ETH80_sort.npy')
    analyse(frE,'E1')

    frIL1 = np.load('./output/frIL1_ETH80_sort.npy')
    analyse(frIL1,'IL1')

    frE2 = np.load('./output/frE2_ETH80_sort.npy')
    analyse(frE2,'E2')

    frIL2 = np.load('./output/frIL2_ETH80_sort.npy')
    analyse(frIL2,'IL2')

if __name__ == "__main__":
    main()
