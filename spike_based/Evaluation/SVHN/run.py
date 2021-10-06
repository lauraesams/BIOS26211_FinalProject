from ANNarchy import *
setup(dt=1.0)
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from net import *
import scipy.io as sio

# showing of the mnist data set to create 
# response vectors

def loadInput():
    data_train = sio.loadmat('./samples/train_32x32.mat')
    data_test = sio.loadmat('./samples/test_32x32.mat')

    return(data_train,data_test)
    
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
    f_0 = 0.4*sizeX
    filt=np.multiply(rho,np.exp(-(rho/f_0)**4))    
    
    data = np.reshape(data,(nbr_images,sizeX,sizeY))
    new_data = np.zeros((nbr_images,sizeX*sizeY))

    plt.figure()
    plt.imshow(data[0],cmap='gray',interpolation='none')
    plt.savefig('test')

    for i in range(nbr_images):
        image = data[i,:,:]
        IF = (np.fft.fft2(image.T))
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
def rgbTOgray(images):
    w,h,d,n_ele = np.shape(images)
    newImages = np.zeros((w,h,n_ele))
    for i in range(n_ele):
        img = images[:,:,:,i]
        newImages[:,:,i] = np.mean(img,axis=2)#np.dot(img[:,:,:3], [0.299, 0.587, 0.114])
    return(newImages)
#----------------------------------------------------------
def startSVHN(duration=125,maxFR=100):
    print('Create response vectors with SVHN')

    if not os.path.exists('output'):
        os.mkdir('output')    

    input_data,test_data = loadInput()
    
    input_img = input_data['X']
    input_label = input_data['y']
    w,h,d,n_eleInput = np.shape(input_img)

    test_img = test_data['X']
    test_label = test_data['y']
    w,h,d,n_eleTest = np.shape(input_img)


    input_img = rgbTOgray(input_img)
    test_img = rgbTOgray(test_img)

    input_img = input_img/255.
    # switcht the order of dimension for the whitening function (must be [n_ele,Xsize,Ysize])
    input_img = np.swapaxes(input_img,0,2)
    input_img = np.swapaxes(input_img,1,2)
    input_img = whitening(input_img)

    test_img = test_img/255.
    # switcht the order of dimension for the whitening function (must be [n_ele,Xsize,Ysize])
    test_img = np.swapaxes(test_img,0,2)
    test_img = np.swapaxes(test_img,1,2)
    test_img = whitening(test_img)


    input_img = overlapInput(patchsize,patchsize,w,h,input_img)
    test_img = overlapInput(patchsize,patchsize,w,h,test_img)


    n_ele,n_patchs = np.shape(input_img)[0:2]
    print('Nr of samples: ', n_ele)
    print('Nr of patches: ', n_patchs)
    compile()
    loadWeights()


    ####                    ####
    # Present the training set #
    ####                    ####
    repeats = 10
    monE1=Monitor(popV1,['spike'])
    monE2=Monitor(popV2,['spike'])

    monI1=Monitor(popIL1,['spike'])
    monI2=Monitor(popIL2,['spike'])

    rec_E1_Fr = np.zeros((nbr_V1N,n_ele,n_patchs))
    rec_E2_Fr = np.zeros((nbr_V2N,n_ele,n_patchs))

    rec_I1_Fr = np.zeros((int(nbr_V1N/4),n_ele,n_patchs))
    rec_I2_Fr = np.zeros((int(nbr_V2N/4),n_ele,n_patchs))

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

                spikesExV1 = monE1.get('spike')
                spikesExV2 = monE2.get('spike')

                spikesIL1 = monI1.get('spike')
                spikesIL2 = monI2.get('spike')

                for c in range(nbr_V1N):
                    rateEx = len(spikesExV1[c])*1000/duration
                    rec_E1_Fr[c,i,p] += rateEx
            
                    rateEx = len(spikesExV2[c])*1000/duration
                    rec_E2_Fr[c,i,p] += rateEx

                    if c < int(nbr_V1N/4):
                        rateInh = len(spikesIL1[c])*1000/duration
                        rec_I1_Fr[c,i,p] += rateInh

                        rateInh = len(spikesIL2[c])*1000/duration
                        rec_I2_Fr[c,i,p] += rateInh

    rec_E1_Fr /= repeats
    rec_E2_Fr /= repeats

    rec_I1_Fr /= repeats
    rec_I2_Fr /= repeats

    np.save('./output/frE1_train.npy',rec_E1_Fr)
    np.save('./output/frE2_train.npy',rec_E2_Fr)

    np.save('./output/frIL1_train.npy',rec_I1_Fr)
    np.save('./output/frIL2_train.npy',rec_I2_Fr)


    ####                ####
    # Present the test set #
    ####                ####
    n_ele,n_patchs = np.shape(test_img)[0:2]

    rec_E1_Fr = np.zeros((nbr_V1N,n_ele,n_patchs))
    rec_E2_Fr = np.zeros((nbr_V2N,n_ele,n_patchs))    

    rec_I1_Fr = np.zeros((int(nbr_V1N/4),n_ele,n_patchs))
    rec_I2_Fr = np.zeros((int(nbr_V2N/4),n_ele,n_patchs))

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
                spikesExV1 = monE1.get('spike')
                spikesExV2 = monE2.get('spike')

                spikesIL1 = monI1.get('spike')
                spikesIL2 = monI2.get('spike')

                for c in range(nbr_V1N):
                    rateEx = len(spikesExV1[c])*1000/duration
                    rec_E1_Fr[c,i,p] += rateEx

                    rateEx = len(spikesExV2[c])*1000/duration
                    rec_E2_Fr[c,i,p] += rateEx

                    if c < int(nbr_V1N/4):
                        rateInh = len(spikesIL1[c])*1000/duration
                        rec_I1_Fr[c,i,p] += rateInh

                        rateInh = len(spikesIL2[c])*1000/duration
                        rec_I2_Fr[c,i,p] += rateInh

    rec_E1_Fr /= repeats
    rec_E2_Fr /= repeats

    rec_I1_Fr /= repeats
    rec_I2_Fr /= repeats

    np.save('./output/frE1_test.npy',rec_E1_Fr)
    np.save('./output/frE2_test.npy',rec_E2_Fr)

    np.save('./output/frIL1_test.npy',rec_I1_Fr)
    np.save('./output/frIL2_test.npy',rec_I2_Fr)

    print('Finish')
#----------------------------------------------------------
if __name__ == "__main__":
    duration=125
    maxFR=100
    startSVHN(duration,maxFR)
