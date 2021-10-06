from ANNarchy import *
setup(dt=1.0)
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from net import *

# showing of the Caltech 101 faces and motorbike subset to create
# response vectors

#----------------------------------------------------------
def resizeCropInput(rfX,rfY,sizeX,sizeY,data):
    print('crop and resize the Input')
    nbr_ele = np.shape(data)[0]
    data = np.reshape(data,(nbr_ele,sizeX,sizeY))
    new_data= np.zeros((nbr_ele,rfX,rfY))
    cropW = 2
    for i in range(nbr_ele):        
        new_data[i,:] = imresize(data[i,cropW:sizeX-cropW,cropW:sizeY-cropW],size=(rfX,rfY),interp='nearest')#'lanczos'

    return(new_data)    
    
#----------------------------------------------------------
def overlapInput(rfX,rfY,sizeX,sizeY,data):
    print('normal with overlap')
    print(np.shape(data))
    nbr_ele=np.shape(data)[0]
    print(sizeX,sizeY)
    shift_Y = 10 # 10*14 = 140+18 = 158 -> loose two pixels
    shift_X = 8   # 8*28 = 224+18 = 242
    n_patchesX = (sizeX-rfX)//shift_X
    n_patchesY = (sizeY-rfY)//shift_Y

    print(n_patchesX)
    print(n_patchesY)

    new_data = np.zeros((nbr_ele,n_patchesX*n_patchesY,rfX,rfY))
    for i in range(nbr_ele):
        j = 0
        for x in range(n_patchesX):
            for y in range(n_patchesY):
                new_data[i,j] = data[i,0+(shift_Y*y):rfY+(shift_Y*y),0+(shift_X*x):rfX+(shift_X*x)]
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
def startCALTECH(duration=125,maxFR=100):
    print('Create response vectors with Faces and Bikes from Caltec 101')

    if not os.path.exists('output'):
        os.mkdir('output')    


    # show every image and choose later which input is in the test and in the train set
    input_data = np.load('./samples/caltech_f_b.npy') # first 435 images are faces, rest (798) bikes

    n_ele,h,w = np.shape(input_data)

    # normalize each image to 0 and 1
    input_data = np.asarray(input_data)/255.
    # pre-whitening
    input_data = whitening(input_data,h,w)

    input_data = overlapInput(patchsize,patchsize,w,h,input_data)

    n_ele,n_patchs = np.shape(input_data)[0:2]
    print('Nr of samples: ', n_ele)
    print('Nr of patches: ', n_patchs)

    compile()
    loadWeights()

    repeats = 10
    monV1=Monitor(popV1,['spike'])
    monI1=Monitor(popIL1,['spike'])

    monV2=Monitor(popV2,['spike'])
    monI2=Monitor(popIL2,['spike'])   

    rec_V1_Fr = np.zeros((nbr_V1N,n_ele,n_patchs))
    rec_V2_Fr = np.zeros((nbr_V2N,n_ele,n_patchs))

    rec_IL1_Fr = np.zeros((int(nbr_V1N/4), n_ele, n_patchs))
    rec_IL2_Fr = np.zeros((int(nbr_V2N/4), n_ele, n_patchs))

    for i in range(n_ele):
        maxV = np.max(np.abs(input_data[i]))
        if((i%(n_ele/10)) == 0):
            print('Round: %i'%i)  
        for p in range(n_patchs):
            inputPatch = prepareInput(input_data[i,p])  
            for r in range(repeats):
                reset()
                
                popInput.rates = (inputPatch/maxV)*maxFR
                simulate(duration)

                spikesEx = monV1.get('spike')
                spikesEx_c = monV2.get('spike')

                spikesIn = monI1.get('spike')
                spikesIn_c = monI2.get('spike')


                for c in range(nbr_V1N):
                    rateEx = len(spikesEx[c])*1000/duration
                    rec_V1_Fr[c,i,p] += rateEx
                    rateEx_c = len(spikesEx_c[c])*1000/duration
                    rec_V2_Fr[c,i,p] += rateEx_c     
                    if c < int(nbr_V1N/4):
                        rateInh = len(spikesIn[c])*1000/duration
                        rec_IL1_Fr[c,i,p] += rateInh

                        rateInh_c = len(spikesIn_c[c])*1000/duration
                        rec_IL2_Fr[c,i,p] += rateInh_c

            rec_V1_Fr[:,i,p] /= repeats
            rec_V2_Fr[:,i,p] /= repeats

            rec_IL1_Fr[:,i,p] /= repeats
            rec_IL2_Fr[:,i,p] /= repeats

    np.save('./output/frE1_data_set.npy',rec_V1_Fr)
    np.save('./output/frE2_data_set.npy',rec_V2_Fr)

    np.save('./output/frIL1_data_set.npy',rec_IL1_Fr)
    np.save('./output/frIL2_data_set.npy',rec_IL2_Fr)

    print('Finish')
#----------------------------------------------------------
if __name__ == "__main__":
    duration=125
    maxFR=100
    startCALTECH(duration,maxFR)
