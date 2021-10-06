import numpy as np

## Whitens a given image, following Olshausen and Field 1997
#data:  [images,imagevalues] or [images,valuesX,valuesY]
#sizeX: image width
#sizeY: image height
#f_0:   cut of frequency (Olshausen used 200)
#n:     steepness parameter (default=4)
#norm_mode: normalization mode 0 - no normalization, 1 - Olshausen
def whiten(data, sizeX=-1, sizeY=-1, f_0=200, n=4, norm_mode=0):
    #Ensure that single images are represented as set of images with one element
    if len(data.shape)==2:
        data=np.expand_dims(data,0)

    #Determine image size
    if sizeX<1:
        sizeX=data.shape[1]
    if sizeY<1:
        sizeY=data.shape[2]

    #Estimate f_0 when no value is given
    if f_0<0:
        f_0 = 0.4*sizeX
        
    #Span up grid of image size with center zero, repeats the array from sizeY [1xm] sizeX times [nx1]
    fx,fy = np.meshgrid(np.arange(-sizeY/2,sizeY/2),np.arange(-sizeX/2,sizeX/2))
    
    #Squared distance matrix to center [Image/2,Image/2]
    rho = np.sqrt(np.multiply(fx,fx)+np.multiply(fy,fy))
    
    #Filter kernel
    filt=np.multiply(rho,np.exp(-(rho/f_0)**n))
    
    #Apply filter on all images
    nbr_images = np.shape(data)[0]
    new_data = np.zeros((nbr_images,sizeX,sizeY))

    for i in range(nbr_images):
        #Apply filter in the fourier spectrum
        IF = np.fft.fft2(data[i,:,:])
        imagew =np.real(np.fft.ifft2(np.multiply(IF.squeeze(),np.fft.fftshift(filt))))
        
        #Reshape data
        new_data[i,:,:]=imagew
    
    ## Normalize data
    if norm_mode==1:
        new_data = np.sqrt(0.1)* new_data/np.sqrt(np.mean(np.var(new_data)))
    
    #new_data = np.reshape(new_data,(nbr_images,sizeX,sizeY))
    return(new_data) 
