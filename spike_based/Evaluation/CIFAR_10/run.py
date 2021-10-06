from ANNarchy import *
setup(dt=1.0)
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from net import *
import scipy.io as sio

# showing Cifar-10 data set to create 
# response vectors

#----------------------------------------------------------
def rgbTOgray(images):
    n_ele,d,w,h = np.shape(images)
    newImages = np.zeros((n_ele,w,h))
    for i in range(n_ele):
        img = images[i,:,:,:]
        newImages[i,:,:] = np.dot(img.T, [0.299, 0.587, 0.114]).T
    return(newImages)
#------------------------------------------------------------
# load and unpickle the CIFAR_10 Dataset as mention on the
# webpage
def unpickle(file):
    import _pickle as cPickle
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
def overlapInput(rfX,rfY,sizeX,sizeY,data):
    print('normal with overlap')
    nbr_ele=np.shape(data)[0]
    data = np.reshape(data,(nbr_ele,sizeX,sizeY))
    nbr_patches = 4#np.round(sizeX/rfX)
    new_data = np.zeros((nbr_ele,nbr_patches,rfX,rfY))
    shift = sizeX - rfX
    for i in range(nbr_ele):
        j = 0
        for x in range(int(nbr_patches/2)):
            for y in range(int(nbr_patches/2)):
                new_data[i,j] = data[i,0+(shift*x):rfX+(shift*x),0+(shift*y):rfY+(shift*y)]
                j +=1
    return(new_data)
#----------------------------------------------------------
def whitening(data):
    #after Olshausen und Field 1996; rewritten from the Olshausen matlab-code
    print('whitening data')
    nbr_images,sizeX,sizeY = np.shape(data)

    fx,fy = np.meshgrid(np.arange(-sizeX/2,sizeX/2),np.arange(-sizeY/2,sizeY/2))
    rho = np.sqrt(np.multiply(fx,fx)+np.multiply(fy,fy))
    
    f_0 = 0.4*(sizeX)
    filt=np.multiply(rho,np.exp(-(rho/f_0)**4)) 

    filt = filt.T # transpose to fix wrong dimensionality caused by meshgrid

    new_data = np.zeros((nbr_images,sizeX*sizeY))

    for i in range(nbr_images):
        image = data[i,:,:]
        IF = np.fft.fft2(image)
        imagew =np.real(np.fft.ifft2(np.multiply(IF,np.fft.fftshift(filt))))
        new_data[i] = np.reshape(imagew,sizeX*sizeY)
    new_data = np.sqrt(0.1)* new_data/np.sqrt(np.mean(np.var(new_data)))
    new_data = np.reshape(new_data,(nbr_images,sizeX,sizeY))
    return(new_data)
#----------------------------------------------------------
def prepareInput(data):
    w,h = np.shape(data)
    patch = np.zeros((w,h,2))
    patch[data>0,0] = data[data>0]
    patch[data<0,1] = data[data<0]*-1
    return(patch)
#----------------------------------------------------------
def startCIFAR10(duration=125,maxFR=100):
    print('Create response vectors on CIFAR_10')

    if not os.path.exists('output'):
        os.mkdir('output')    

    w = 32
    h = 32
    d = 3

    data_train,label_train,data_test,label_test = loadInput()

    nb_eleTrain,nb_values = np.shape(data_train)
    input_img = np.reshape(data_train,(nb_eleTrain,d,w,h))
    
    nb_eleTest,nb_values = np.shape(data_test)
    test_img = np.reshape(data_test,(nb_eleTest,d,w,h))

    input_img = rgbTOgray(input_img)
    test_img = rgbTOgray(test_img)

    input_img = input_img/255.

    # switcht the order of dimension for the whitening function (must be [n_ele,Xsize,Ysize])
    input_img = whitening(input_img)

    test_img = test_img/255.
    # switcht the order of dimension for the whitening function (must be [n_ele,Xsize,Ysize])
    test_img = whitening(test_img)


    input_img = overlapInput(patchsize,patchsize,w,h,input_img)
    test_img = overlapInput(patchsize,patchsize,w,h,test_img)

    n_ele,n_patchs = np.shape(input_img)[0:2]
    print('Nr of samples: ', n_ele)
    print('Nr of patches: ', n_patchs)

    compile()
    loadWeights()

    repeats = 10

    ####                    ####
    # Present the training set #
    ####                    ####

    monV1=Monitor(popV1,['spike'])
    monV2=Monitor(popV2,['spike'])    

    monI1=Monitor(popIL1,['spike'])
    monI2=Monitor(popIL2,['spike'])    


    rec_V1_Fr = np.zeros((nbr_V1N,n_ele,n_patchs))
    rec_V2_Fr = np.zeros((nbr_V2N,n_ele,n_patchs))

    rec_IL1_Fr = np.zeros((int(nbr_V1N/4),n_ele,n_patchs))
    rec_IL2_Fr = np.zeros((int(nbr_V2N/4),n_ele,n_patchs))


    for i in range(n_ele):
        maxV = np.max(np.abs(input_img[i]))
        if((i%(n_ele/10)) == 0):
            print('Round: %i'%i)  
        for p in range(n_patchs):
            inputPatch = prepareInput(input_img[i,p])  
            for r in range(repeats):
                reset()
                popInput.rates = (inputPatch/maxV)*maxFR
                simulate(duration)

                spikesEx = monV1.get('spike')
                spikesEx_c = monV2.get('spike')

                spikesIL = monI1.get('spike')
                spikesIL_c= monI2.get('spike')

                for c in range(nbr_V1N):
                    rateEx = len(spikesEx[c])*1000/duration
                    rec_V1_Fr[c,i,p] += rateEx
                    rateEx_c = len(spikesEx_c[c])*1000/duration
                    rec_V2_Fr[c,i,p] += rateEx_c                    

                    if c < int(nbr_V1N/4):
                        rateIL = len(spikesIL[c])*1000/duration
                        rec_IL1_Fr[c,i,p] += rateIL
                        rateIL_c = len(spikesIL_c[c])*1000/duration
                        rec_IL2_Fr[c,i,p] += rateIL_c

    rec_V1_Fr/= repeats
    rec_V2_Fr/= repeats

    rec_IL1_Fr/= repeats
    rec_IL2_Fr/= repeats


    np.save('./output/frE1_train.npy',rec_V1_Fr)
    np.save('./output/frE2_train.npy',rec_V2_Fr)

    np.save('./output/frIL1_train.npy',rec_IL1_Fr)
    np.save('./output/frIL2_train.npy',rec_IL2_Fr)



    ####                ####
    # Present the test set #
    ####                ####

    n_ele,n_patchs = np.shape(test_img)[0:2]

    rec_V1_Fr = np.zeros((nbr_V1N,n_ele,n_patchs))
    rec_V2_Fr = np.zeros((nbr_V2N,n_ele,n_patchs))

    rec_IL1_Fr = np.zeros((int(nbr_V1N/4),n_ele,n_patchs))
    rec_IL2_Fr = np.zeros((int(nbr_V2N/4),n_ele,n_patchs))

    for i in range(n_ele):
        maxV = np.max(np.abs(test_img[i]))
        if((i%(n_ele/10)) == 0):
            print('Round: %i'%i)
        for p in range(n_patchs):
            for r in range(repeats):
                reset()
                inputPatch = prepareInput(test_img[i,p])
                popInput.rates = (inputPatch/maxV)*maxFR
                simulate(duration)
                spikesEx_s = monV1.get('spike')
                spikesEx_c = monV2.get('spike')

                spikesIL = monI1.get('spike')
                spikesIL_c= monI2.get('spike')

                for c in range(nbr_V1N):
                    rateEx_s = len(spikesEx_s[c])*1000/duration
                    rec_V1_Fr[c,i,p] += rateEx_s
                    rateEx_c = len(spikesEx_c[c])*1000/duration
                    rec_V2_Fr[c,i,p] += rateEx_c

                    if c < int(nbr_V1N/4):
                        rateIL = len(spikesIL[c])*1000/duration
                        rec_IL1_Fr[c,i,p] += rateIL
                        rateIL_c = len(spikesIL_c[c])*1000/duration
                        rec_IL2_Fr[c,i,p] += rateIL_c

    rec_V1_Fr /= repeats
    rec_V2_Fr /= repeats

    rec_IL1_Fr/= repeats
    rec_IL2_Fr/= repeats

    np.save('./output/frE1_test.npy',rec_V1_Fr)
    np.save('./output/frE2_test.npy',rec_V2_Fr)

    np.save('./output/frIL1_test.npy',rec_IL1_Fr)
    np.save('./output/frIL2_test.npy',rec_IL2_Fr)

    print('Finish')
#----------------------------------------------------------
if __name__ == "__main__":
    duration=125
    maxFR=100.0
    startCIFAR10(duration,maxFR)
