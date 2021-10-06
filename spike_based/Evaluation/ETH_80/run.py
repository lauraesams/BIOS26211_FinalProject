from ANNarchy import *
setup(dt=1.0)
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from net import *
from scipy.misc import imresize

# showing of the mnist data set to create 
# response vectors

def loadInput():
    
    data = np.load('samples/ETH80_dataSort.npy')

    return(data)
#----------------------------------------------------------
def resizeInput(rfX,rfY,data):
    print('resize Input')
    n_classes,n_instances,n_ele,sizeW,sizeH = np.shape(data)

    new_data = np.zeros((n_classes,n_instances,n_ele,rfX,rfY))

    for c in range(n_classes):
        for i in range(n_instances):
            for e in range(n_ele):
                new_data[c,i,e] = imresize(data[c,i,e],(rfX,rfY))

    plt.figure()
    plt.imshow(new_data[2,0,0],cmap='gray',interpolation='none')
    plt.savefig('test_resize')

    plt.figure()
    plt.imshow(data[2,0,0],cmap='gray',interpolation='none')
    plt.savefig('test')

    return(new_data)    
#----------------------------------------------------------
def overlapInput(rfX,rfY,data):
    print('normal with overlap')
    n_classes,n_instances,n_ele,sizeX,sizeY=np.shape(data)
    nbr_patches = int(sizeX/rfX)
    new_data = np.zeros((n_classes,n_instances,n_ele,nbr_patches*nbr_patches,rfX,rfY))
    shift = rfX
    for c in range(n_classes):
        for i in range(n_instances):
            for e in range(n_ele):
                j = 0
                for x in range(nbr_patches):
                    for y in range(nbr_patches):
                        new_data[c,i,e,j] = data[c,i,e,0+(shift*x):rfX+(shift*x),0+(shift*y):rfY+(shift*y)]
                        j +=1
    return(new_data)
#----------------------------------------------------------
def whitening(data,sizeX,sizeY):
    #after Olshausen und Field 1996; rewritten from the Olshausen matlab-code
    print('whitening data')
    n_classes,n_instances,n_ele,sizeW,sizeH = np.shape(data)
    fx,fy = np.meshgrid(np.arange(-sizeX/2,sizeX/2),np.arange(-sizeY/2,sizeY/2))
    rho = np.sqrt(np.multiply(fx,fx)+np.multiply(fy,fy))
    
    f_0 = 0.4*(sizeX)
    filt=np.multiply(rho,np.exp(-(rho/f_0)**4)) 

    filt = filt.T # transpose to fix wrong dimensionality caused by meshgrid

    new_data = np.zeros((n_classes,n_instances,n_ele,sizeX*sizeY))

    plt.figure()
    plt.imshow(data[4,0,0],cmap='gray',interpolation='none')
    plt.savefig('test_pre')

    for c in range(n_classes):
        for i in range(n_instances):
            for e in range(n_ele):
                image = data[c,i,e,:,:]
                IF = (np.fft.fft2(image))
                imagew =np.real(np.fft.ifft2(np.multiply(IF,np.fft.fftshift(filt))))
                new_data[c,i,e] = np.reshape(imagew,sizeX*sizeY)

            new_data[c,i] = np.sqrt(0.1)* new_data[c,i]/np.sqrt(np.mean(np.var(new_data[c,i]))) # normalization over all images ?!
    new_data = np.reshape(new_data,(n_classes,n_instances,n_ele,sizeX,sizeY))

    plt.figure()
    plt.imshow(new_data[4,0,0],cmap='gray',interpolation='none')
    plt.savefig('test_post')

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
    print('Create response vectors with ETH-80')


    if not os.path.exists('output'):
        os.mkdir('output')    

    w,h = (126,126)


    input_data = loadInput()
    n_classes,n_instances,n_ele,sizeW,sizeH,d = np.shape(input_data)
    print(np.shape(input_data))
    #mean over axis 3 to transform into gray
    input_data = np.mean(input_data,axis=5)


    input_data = input_data/255.


    input_data = resizeInput(w,h,input_data)
    input_data = whitening(input_data,w,h)

    input_data = overlapInput(patchsize,patchsize,input_data)

    n_classes,n_instances,n_ele,n_patches,sizeW,sizeH = np.shape(input_data)

    compile()
    loadWeights()

    repeats = 10
    monV1=Monitor(popV1,['spike'])
    monV2=Monitor(popV2,['spike'])

    monI1=Monitor(popIL1,['spike'])
    monI2=Monitor(popIL2,['spike'])


    rec_E1_Fr = np.zeros((nbr_V1N,n_classes,n_instances,n_ele,n_patches))
    rec_E2_Fr = np.zeros((nbr_V1N,n_classes,n_instances,n_ele,n_patches))

    rec_IL1_Fr = np.zeros((int(nbr_V1N/4),n_classes,n_instances,n_ele,n_patches))
    rec_IL2_Fr = np.zeros((int(nbr_V1N/4),n_classes,n_instances,n_ele,n_patches))


    for c in range(n_classes):
        print('Class # %i:'%(c))
        for i in range(n_instances):
            for e in range(n_ele):
                maxV = np.max(np.abs(input_data[c,i,e]))  
                for p in range(n_patches):
                    inputPatch = prepareInput(input_data[c,i,e,p])  
                    for r in range(repeats):
                        reset()
                        popInput.rates = (inputPatch/maxV)*maxFR
                        simulate(duration)

                        spikesEx = monV1.get('spike')
                        spikesE2 = monV2.get('spike')

                        spikesIL1 = monI1.get('spike')
                        spikesIL2 = monI2.get('spike')

                        for n in range(nbr_V1N):
                            rateEx = len(spikesEx[n])*1000/duration
                            rec_E1_Fr[n,c,i,e,p] += rateEx
                    
                            rateE2 =len(spikesE2[n])*1000/duration
                            rec_E2_Fr[n,c,i,e,p] += rateE2

                            if n< int(nbr_V1N/4):
                                rateInh = len(spikesIL1[n])*1000/duration
                                rec_IL1_Fr[n,c,i,e,p] += rateInh
            
                                rateInh2 = len(spikesIL2[n])*1000/duration
                                rec_IL2_Fr[n,c,i,e,p] += rateInh2

                    rec_E1_Fr[:,c,i,e,p] /= repeats
                    rec_E2_Fr[:,c,i,e,p] /= repeats

                    rec_IL1_Fr[:,c,i,e,p] /= repeats
                    rec_IL2_Fr[:,c,i,e,p] /= repeats

    np.save('./output/frE1_ETH80_sort.npy',rec_E1_Fr)
    np.save('./output/frE2_ETH80_sort.npy',rec_E2_Fr)

    np.save('./output/frIL1_ETH80_sort.npy',rec_IL1_Fr)
    np.save('./output/frIL2_ETH80_sort.npy',rec_IL2_Fr)

    print('Finish')
#----------------------------------------------------------
if __name__ == "__main__":
    duration=125
    maxFR=100
    startMNIST(duration,maxFR)
