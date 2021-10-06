from ANNarchy import *
setup(dt=1.0)
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from net import *
from mnist import MNIST
from scipy.misc import imresize

# showing of the mnist data set to create 
# response vectors

def loadInput():
    ## Load the MNIST data set

    mndata = MNIST('samples')
    train_img, train_labl = mndata.load_training()
    test_img, test_labl = mndata.load_testing()

    return(train_img,test_img)
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
def whitening(data,sizeX,sizeY):
    #after Olshausen und Field 1996; rewritten from the Olshausen matlab-code
    print('whitening data')
    fx,fy = np.meshgrid(np.arange(-sizeX/2,sizeX/2),np.arange(-sizeY/2,sizeY/2))
    rho = np.sqrt(np.multiply(fx,fx)+np.multiply(fy,fy))
    
    f_0 = 0.4*(sizeX)
    filt=np.multiply(rho,np.exp(-(rho/f_0)**4)) 

    filt = filt.T # transpose to fix wrong dimensionality caused by meshgrid

    nbr_images = np.shape(data)[0]
    data =  np.reshape(data,(nbr_images,sizeX,sizeY))
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
def startMNIST(duration=125,maxFR=100):
    print('Create response vectors for MNIST')


    if not os.path.exists('output'):
        os.mkdir('output')    

    w,h = (28,28)

    # get the training and test set of MNIST 
    input_data,test_data = loadInput()
    
    # normalize input images to 0 and 1
    input_data = np.asarray(input_data)/255.
    # pre-whitening 
    input_data = whitening(input_data,w,h)

    test_data = np.asarray(test_data)/255.
    test_data = whitening(test_data,w,h)

    # create overlapping patches in the desired resolution
    input_data = overlapInput(patchsize,patchsize,w,h,input_data)
    test_data = overlapInput(patchsize,patchsize,w,h,test_data)

   
    n_ele,n_patchs = np.shape(input_data)[0:2]
    print('Nr of samples: ', n_ele)
    print('Nr of patches: ', n_patchs)
    compile()
    loadWeights()

    ####                    ####
    # Present the training set #
    ####                    ####
    repeats = 10
    monV1=Monitor(popV1,['spike'])
    monIL1=Monitor(popIL1,['spike'])

    monV2=Monitor(popV2,['spike'])
    monIL2=Monitor(popIL2,['spike'])

    rec_V1_Fr = np.zeros((nbr_V1N,n_ele,n_patchs))
    rec_V2_Fr = np.zeros((nbr_V2N,n_ele,n_patchs))

    rec_IL1_Fr = np.zeros((int(nbr_V1N/4),n_ele,n_patchs))
    rec_IL2_Fr = np.zeros((int(nbr_V2N/4),n_ele,n_patchs))


    for i in range(n_ele):
        maxV = np.max(np.abs(input_data[i]))
        if((i%(n_ele/10)) == 0):
            print('Round: %i'%i)  
        for p in range(n_patchs):
            inputPatch = prepareInput(input_data[i,p])  
            for r in range(repeats):
                reset()
                ## set the input patch                
                popInput.rates = (inputPatch/maxV)*maxFR
                ## present the input for duration [ms]
                simulate(duration)

                spikesExV1 = monV1.get('spike')
                spikesExV2 = monV2.get('spike')

                spikesIL1 = monIL1.get('spike')
                spikesIL2 = monIL2.get('spike')

                # save the firing rate for all four populations
                for c in range(nbr_V1N):
                    rateExV1 = len(spikesExV1[c])*1000/duration
                    rec_V1_Fr[c,i,p] += rateExV1
                    rateExV2 = len(spikesExV2[c])*1000/duration
                    rec_V2_Fr[c,i,p] += rateExV2
                    if c < int(nbr_V1N/4):
                        rateExIL1 = len(spikesIL1[c])*1000/duration
                        rec_IL1_Fr[c,i,p] += rateExIL1

                        rateExIL2 = len(spikesIL2[c])*1000/duration
                        rec_IL2_Fr[c,i,p] += rateExIL2

    # normalize with the number of repeats
    rec_V1_Fr /= repeats
    rec_V2_Fr /= repeats

    rec_IL1_Fr /= repeats
    rec_IL2_Fr /= repeats

    ## save the recorded firing rates
    np.save('./output/frV1_train.npy',rec_V1_Fr)
    np.save('./output/frV2_train.npy',rec_V2_Fr)

    np.save('./output/frIL1_train.npy',rec_IL1_Fr)
    np.save('./output/frIL2_train.npy',rec_IL2_Fr)



    ####                ####
    # Present the test set #
    ####                ####
    n_ele,n_patchs = np.shape(test_data)[0:2]

    rec_V1_Fr = np.zeros((nbr_V1N,n_ele,n_patchs))
    rec_V2_Fr = np.zeros((nbr_V2N,n_ele,n_patchs))

    rec_IL1_Fr = np.zeros((int(nbr_V1N/4),n_ele,n_patchs))
    rec_IL2_Fr = np.zeros((int(nbr_V2N/4),n_ele,n_patchs))

    for i in range(n_ele):
        maxV = np.max(np.abs(test_data[i]))
        if((i%(n_ele/10)) == 0):
            print('Round: %i'%i)
        for p in range(n_patchs):
            for r in range(repeats):
                reset()
                inputPatch = prepareInput(test_data[i,p])
                popInput.rates = (inputPatch/maxV)*maxFR
                simulate(duration)
                spikesExV1 = monV1.get('spike')
                spikesExV2 = monV2.get('spike')

                spikesIL1 = monIL1.get('spike')
                spikesIL2 = monIL2.get('spike')
                for c in range(nbr_V1N):
                    rateExV1 = len(spikesExV1[c])*1000/duration
                    rec_V1_Fr[c,i,p] += rateExV1
                    rateExV2 = len(spikesExV2[c])*1000/duration
                    rec_V2_Fr[c,i,p] += rateExV2

                    if c < int(nbr_V1N/4):
                        rateExIL1 = len(spikesIL1[c])*1000/duration
                        rec_IL1_Fr[c,i,p] += rateExIL1

                        rateExIL2 = len(spikesIL2[c])*1000/duration
                        rec_IL2_Fr[c,i,p] += rateExIL2

    rec_V1_Fr /= repeats
    rec_V2_Fr /= repeats

    rec_IL1_Fr /= repeats
    rec_IL2_Fr /= repeats

    np.save('./output/frV1_test.npy',rec_V1_Fr)
    np.save('./output/frV2_test.npy',rec_V2_Fr)

    np.save('./output/frIL1_test.npy',rec_IL1_Fr)
    np.save('./output/frIL2_test.npy',rec_IL2_Fr)
    print('Finish')
#----------------------------------------------------------
if __name__ == "__main__":
    duration=125
    maxFR=100
    startMNIST(duration,maxFR)
