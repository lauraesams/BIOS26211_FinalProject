from __future__ import print_function

import numpy as np
import h5py
import scipy
import time
import Whiten
from ANNarchy import populations,simulate

##########################################################################
## Function for testing the network
def present_Images(Inputlayer,inputdata, savefile='test_resposes', presentation_time=100, norm_factor=0, norm_max_value=0.5):
    print("Start time test:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    print("Test data:", inputdata)
    
    #Load testing images
    element=0
    images = scipy.io.loadmat(inputdata).values()[element]
    while str(type(images))!="<type 'numpy.ndarray'>": #For unknown reasons the data are not always stored in the first item
        element=element+1
        images = scipy.io.loadmat(inputdata).values()[element]
    
    #Expand dims to always have 4d geometry
    if len(images.shape)<4:
        images=images[:,:,:,np.newaxis]
    img_shape=images.shape
    
    #Init datastructures for storage, using a list structure filled with numpy arrays for every regarded population
    poptosave_Rec_r=list()
    for pop in populations():
        #if pop.name!=Inputlayer.name:
        poptosave_Rec_r.append(np.zeros((img_shape[0],pop.width,pop.height,pop.depth)))

    
    ##Present data to the network
    for i in range(0,img_shape[0]):
        # Time measurement
        if i==1:
            t0=time.time()
        if i==11:
            t1000=time.time()
            expected=(t1000-t0)*(img_shape[0]-10)/10
            days=int(expected / 86400)
            hours=int(expected % 86400/ 3600)
            minutes=int(expected % 3600/ 60)
            seconds=int(expected % 60)
            print('Time expected: ' + str(days) + ' - ' + str(hours) + ':' + str(minutes) + ':'+ str(seconds) + ' d-h:m:s')

        #Output
        if (i>=1) and (i<6):
            output='Image:'+str(i)+' '
            for pop in populations(): 
                output=output+pop.name+'#Mean:'+str(np.mean(pop.r))+' Max:'+str(np.max(pop.r))+'|'
            print(output)

        #Generate empty image of input layer size
        image=np.zeros(Inputlayer.geometry)
        
        #Decide if images have to be preprocessed
        
        #If input image size differes cut out patch at the center and place it at the center
        #Start and end for the cut from the image
        x_start=max((img_shape[1]-Inputlayer.geometry[0])/2, 0)
        x_end=x_start+Inputlayer.geometry[0]
        y_start=max((img_shape[2]-Inputlayer.geometry[1])/2, 0)
        y_end=y_start+Inputlayer.geometry[1]
        
        #Start and end for the placement in the resulting image given to the network
        xi_start=max((Inputlayer.geometry[0]-img_shape[1])/2, 0)
        xi_end=xi_start+img_shape[1]
        yi_start=max((Inputlayer.geometry[1]-img_shape[2])/2, 0)
        yi_end=yi_start+img_shape[2]
        
        if img_shape[3]==Inputlayer.geometry[2]:
            #No preprocessing in terms of whitening needed
            image[xi_start:xi_end,yi_start:yi_end,:]=images[i,x_start:x_end,y_start:y_end,:]
            
        else:
            #It is assumed that the image should be whitened, else something is totally wrong
            tmp=np.array(Whiten.whiten(data=np.expand_dims(images[i,:,:,:].squeeze(),0), sizeX=-1, sizeY=-1, f_0=-1))
            image[xi_start:xi_end,yi_start:yi_end,0]=np.fmax(tmp[x_start:x_end,y_start:y_end],0)
            image[xi_start:xi_end,yi_start:yi_end,1]=np.fmax(-tmp[x_start:x_end,y_start:y_end],0)
            

        
        #Determine the norm factor based on the image containing target and mask
        if norm_factor==0 and norm_max_value>0:
            norm_factor=norm_max_value/(np.max(np.abs(image))+np.finfo(float).eps)
        

        #Reset the network activities
        for pop in populations():
            pop.reset(attributes=['r'])
            pop.mp=0
            pop.I_exc=0
            pop.I_ff=0
            pop.I_inh=0
            if hasattr(pop,'I_fb')==True:
                pop.I_fb=0
            if hasattr(pop,'I_lat')==True:
                pop.I_lat=0
            if hasattr(pop,'r_slow')==True:
                pop.r_slow=0
                
        #Set input patch
        Inputlayer.input=image*norm_factor
        
        #Run network
        simulate(presentation_time)
        
        #Run additional time steps to compensate synaptic delays for recording
        t_prev=0#Cummulated time the network has additionally runned
        for it_add in range(len(unique_delay)):
            #Time the network should run additionally
            t_add=unique_delay[it_add]-t_prev
            if t_add>0:
                simulate(t_add)

            ## Record activities
            pop_counter=0
            for pop in populations():
                # Do not record from input populations
                #if pop.name!=Inputlayer.name:
                #Get responses of each neuron in the population
                if delays[pop.name]==unique_delay[it_add]:
                    poptosave_Rec_r[pop_counter][i,:,:,:] = pop.get("r")
                pop_counter=pop_counter+1
                
            t_prev=unique_delay[it_add]
            

                
    ## Save
    print("Saving results...")
    pop_counter=0
    for pop in populations():
        #if pop.name!=Inputlayer.name:
        #Save using h5py
        f=h5py.File(str(savefile+'_'+pop.name+'.h5'),'w')
        f[str('r_'+pop.name)] = poptosave_Rec_r[pop_counter]
        f.close()
        pop_counter=pop_counter+1


##########################################################################
## Function for testing the network
def present_Imagesd(Inputlayer,inputdata, savefile='test_resposes', presentation_time=100, norm_factor=0, norm_max_value=0.5, delays={"LGN":0, "V1_L4_Exc":2, "V1_L4_Inh":2, "V1_L23_Exc":4, "V1_L23_Inh":4, "V2_L4_Exc":7, "V2_L4_Inh":7, "V2_L23_Exc":9, "V2_L23_Inh":9}):
    print("Start time test:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    print("Test data:", inputdata)
    print("Records from", delays.keys())
    
    # Calculate delays for determining the saving points
    delay=[]
    for element in delays:
        delay.append(delays[element])
    unique_delay=np.unique(delay)
    
    #Load testing images
    import scipy.io
    dict_images = scipy.io.loadmat(inputdata)
    #Select the ndarray | the routine takes the first one
    for str_key,key_value in dict_images.items():
        #print(str_key)
        #print(type(key_value))
        if str(type(key_value))=="<class 'numpy.ndarray'>":
            images=key_value
            print(images.shape)
            break
    
    #Expand dims to always have 4d geometry
    if len(images.shape)<4:
        images=images[:,:,:,np.newaxis]
    img_shape=images.shape
    
    #Init datastructures for storage, using a list structure filled with numpy arrays for every regarded population
    poptosave_Rec_r=list()
    for pop in populations():
        #if pop.name!=Inputlayer.name:
        poptosave_Rec_r.append(np.zeros((img_shape[0],pop.width,pop.height,pop.depth)))

    
    ##Present data to the network
    for i in range(0,img_shape[0]):
        # Time measurement
        if i==1:
            t0=time.time()
        if i==11:
            t1000=time.time()
            expected=(t1000-t0)*(img_shape[0]-10)/10
            days=int(expected / 86400)
            hours=int(expected % 86400/ 3600)
            minutes=int(expected % 3600/ 60)
            seconds=int(expected % 60)
            print('Time expected: ' + str(days) + ' - ' + str(hours) + ':' + str(minutes) + ':'+ str(seconds) + ' d-h:m:s')

        #Output
        if (i>=1) and (i<6):
            output='Image:'+str(i)+' '
            for pop in populations(): 
                output=output+pop.name+'#Mean:'+str(np.mean(pop.r))+' Max:'+str(np.max(pop.r))+'|'
            print(output)

        #Generate empty image of input layer size
        image=np.zeros(Inputlayer.geometry)
        
        #Decide if images have to be preprocessed
        
        #If input image size differes cut out patch at the center and place it at the center
        #Start and end for the cut from the image
        x_start=int(max((img_shape[1]-Inputlayer.geometry[0])/2, 0))
        x_end=int(x_start+Inputlayer.geometry[0])
        y_start=int(max((img_shape[2]-Inputlayer.geometry[1])/2, 0))
        y_end=int(y_start+Inputlayer.geometry[1])
        
        #Start and end for the placement in the resulting image given to the network
        xi_start=int(max((Inputlayer.geometry[0]-img_shape[1])/2, 0))
        xi_end=int(xi_start+img_shape[1])
        yi_start=int(max((Inputlayer.geometry[1]-img_shape[2])/2, 0))
        yi_end=int(yi_start+img_shape[2])
        
        if img_shape[3]==Inputlayer.geometry[2]:
            #No preprocessing in terms of whitening needed
            image[xi_start:xi_end,yi_start:yi_end,:]=images[i,x_start:x_end,y_start:y_end,:]
            
        else:
            #It is assumed that the image should be whitened, else something is totally wrong
            tmp=np.array(Whiten.whiten(data=np.expand_dims(images[i,:,:,:].squeeze(),0), sizeX=-1, sizeY=-1, f_0=-1))
            image[xi_start:xi_end,yi_start:yi_end,0]=np.fmax(tmp[x_start:x_end,y_start:y_end],0)
            image[xi_start:xi_end,yi_start:yi_end,1]=np.fmax(-tmp[x_start:x_end,y_start:y_end],0)
            

        
        #Determine the norm factor based on the image containing target and mask
        if norm_factor==0 and norm_max_value>0:
            norm_factor=norm_max_value/(np.max(np.abs(image))+np.finfo(float).eps)
        

        #Reset the network activities
        for pop in populations():
            pop.reset(attributes=['r'])
            pop.mp=0
            pop.I_exc=0
            pop.I_ff=0
            pop.I_inh=0
            if hasattr(pop,'I_fb')==True:
                pop.I_fb=0
            if hasattr(pop,'I_lat')==True:
                pop.I_lat=0
            if hasattr(pop,'r_slow')==True:
                pop.r_slow=0
                
        #Set input patch
        Inputlayer.input=image*norm_factor
        
        #Run network
        simulate(presentation_time)
        
        ## Record activities
        pop_counter=0
        for pop in populations():
            # Do not record from input populations
            #if pop.name!=Inputlayer.name:
            #Get responses of each neuron in the population
            poptosave_Rec_r[pop_counter][i,:,:,:] = pop.get("r")
            pop_counter=pop_counter+1
                
    ## Save
    print("Saving results...")
    pop_counter=0
    for pop in populations():
        #if pop.name!=Inputlayer.name:
        #Save using h5py
        f=h5py.File(str(savefile+'_'+pop.name+'.h5'),'w')
        f[str('r_'+pop.name)] = poptosave_Rec_r[pop_counter]
        f.close()
        pop_counter=pop_counter+1


##########################################################################
## Function for testing the network
def present_Images_tiles2(Inputlayer, inputdata, savefile='test_resposes', presentation_time=100, norm_factor=0, norm_max_value=0.5, tilesX=np.inf, tilesY=np.inf, overlapX=0, overlapY=0, compression=0):
    print("Start time test:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    print("Test data:", inputdata)
    
    #Load testing images
    element=0
    images = scipy.io.loadmat(inputdata).values()[element]
    while str(type(images))!="<type 'numpy.ndarray'>": #For unknown reasons the data are not always stored in the first item
        element=element+1
        images = scipy.io.loadmat(inputdata).values()[element]
    
    #Expand dims to always have 4d geometry
    if len(images.shape)<4:
        images=images[:,:,:,np.newaxis]
    img_shape=images.shape

    #Amount of used tiles
    #How many tiles are maximally needed n=(img-overlap)/(input-overlap); assumes that the tiles just cover the whole image
    max_tilesX=np.ceil( (img_shape[1]-overlapX) / float(Inputlayer.geometry[0]-overlapX) )
    max_tilesY=np.ceil( (img_shape[2]-overlapY) / float(Inputlayer.geometry[1]-overlapY) )
    
    if tilesX<=0:
        tilesX=1
    elif tilesX==np.inf:
        tilesX=max_tilesX
        
    if tilesY<=0:
        tilesY=tilesX
    elif tilesY==np.inf:
        tilesY=max_tilesY

    tilesX=int(min(max_tilesX,tilesX))
    tilesY=int(min(max_tilesY,tilesY))
    print('tilesX:',tilesX,'tilesY:',tilesY)


    #Init datastructures for storage, using a list structure filled with numpy arrays for every regarded population
    #Save using h5py
    f=h5py.File(str(savefile+'.h5'),'w', rdcc_nbytes=1024*1024*1024*1, rdcc_w0=1)#http://docs.h5py.org/en/stable/high/file.html#chunk-cache Might just work from version 2.9
    for pop in populations():
        if compression==1:
            f.create_dataset('r_'+pop.name, (img_shape[0],tilesX,tilesY,pop.width,pop.height,pop.depth), dtype='float32', chunks=(10,tilesX,tilesY,pop.width,pop.height,pop.depth), compression="gzip",  compression_opts=1)#gzip compression costs time, lzf is faster, BUT MATLAB DOESNT LOAD IT; chucks cost a bit time, chuncks=True is much faster than defining a size
        else:
            f.create_dataset('r_'+pop.name, (img_shape[0],tilesX,tilesY,pop.width,pop.height,pop.depth), dtype='float32')


    #Init time measurement
    t1=time.time()
    
    ##Present data to the network
    for i in range(0,img_shape[0]):
        # Time measurement
        if i==1:
            t0=time.time()
        if i==2:
            t1000=time.time()
            expected=(t1000-t0)*(img_shape[0]-1)/1
            days=int(expected / 86400)
            hours=int(expected % 86400/ 3600)
            minutes=int(expected % 3600/ 60)
            seconds=int(expected % 60)
            print('Time expected: ' + str(days) + ' - ' + str(hours) + ':' + str(minutes) + ':'+ str(seconds) + ' d-h:m:s')

        #Generate empty image of input layer size
        image_patch=np.zeros(Inputlayer.geometry)
        
        #Decide if images have to be preprocessed
        if not img_shape[3]==Inputlayer.geometry[2]:
            #It is assumed that the image should be whitened, else something is totally wrong
            tmp=np.squeeze(np.array(Whiten.whiten(data=np.expand_dims(images[i,:,:,:].squeeze(),0), sizeX=-1, sizeY=-1, f_0=-1)))
            image_whiten=np.zeros((tmp.shape[0],tmp.shape[1],2))
            image_whiten[:,:,0]=np.fmax(tmp,0)
            image_whiten[:,:,1]=np.fmax(-tmp,0)
            
        for ti_x in range(0,tilesX):
            for ti_y in range(0,tilesY):
                
                #If input image size differs cut out patch at the center and place it at the center
                #Start and end for the cut from the image
                x_start=max(ti_x*Inputlayer.geometry[0] - ti_x*overlapX,0)
                x_end=min(x_start+min(Inputlayer.geometry[0],img_shape[1]),img_shape[1])
                y_start=max(ti_y*Inputlayer.geometry[1] - ti_y*overlapY,0)
                y_end=min(y_start+min(Inputlayer.geometry[1],img_shape[2]),img_shape[2])


                #Generate empty image of input layer size
                image_patch=np.zeros(Inputlayer.geometry)
                    
                #Decide if images have to be preprocessed
                if img_shape[3]==Inputlayer.geometry[2]:
                    #No preprocessing in terms of whitening needed
                    image_patch=images[i,x_start:x_end,y_start:y_end,:]
                else:
                    image_patch=image_whiten[x_start:x_end,y_start:y_end,:]
                
                #Determine the norm factor based on the image containing target and mask
                if norm_factor==0 and norm_max_value>0:
                    norm_factor=norm_max_value/(np.max(np.abs(image_patch))+np.finfo(float).eps)
                

                #Reset the network activities
                for pop in populations():
                    pop.reset(attributes=['r'])
                    pop.mp=0
                    pop.I_exc=0
                    pop.I_ff=0
                    pop.I_inh=0
                    if hasattr(pop,'I_fb')==True:
                        pop.I_fb=0
                    if hasattr(pop,'I_lat')==True:
                        pop.I_lat=0
                    if hasattr(pop,'r_slow')==True:
                        pop.r_slow=0
                        
                #Set input patch
                Inputlayer.input=image_patch*norm_factor
                
                #Run network
                simulate(presentation_time)
                
                ## Record activities
                pop_counter=0
                for pop in populations():
                    # Do not record from input populations
                    #if pop.name!=Inputlayer.name:
                    #Get responses of each neuron in the population
                    f[str('r_'+pop.name)][i,ti_x,ti_y,:,:,:] = pop.get("r")
                    pop_counter=pop_counter+1
                
                #Output
                if (i==0) and (ti_x==ti_y):
                    #Counting output length for nicer multiline output
                    prev_lines=0
                    output=''
                    for pop in populations():
                        output=output+pop.name+'#Mean:'+"%.3f"%(np.mean(pop.r))+' Max:'+"%.3f"%(np.max(pop.r))
                        if len(output)-prev_lines>200:
                            output=output+'\n'
                            prev_lines=len(output)
                        else:
                            output=output+'|'
                    print(output)
                    output='Image:'+str(i)+' '
                
    ## Save
    print("Saving results...")
    f.close()
    
##########################################################################
## Function for testing the network
def present_Images_tiles3(Inputlayer, inputdata, savefile='test_resposes', presentation_time=100, norm_factor=0, norm_max_value=0.5, tilesX=np.inf, tilesY=np.inf, overlapX=0, overlapY=0, delays={"LGN":0, "V1_L4_Exc":2, "V1_L4_Inh":2, "V1_L23_Exc":4, "V1_L23_Inh":4, "V2_L4_Exc":7, "V2_L4_Inh":7, "V2_L23_Exc":9, "V2_L23_Inh":9}, compression=0):
    print("Start time test:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    print("Test data:", inputdata)
    print("Records from", delays.keys())
    
    # Calculate delays for determining the saving points
    delay=[]
    for element in delays:
        delay.append(delays[element])
    unique_delay=np.unique(delay)
    
    #Load testing images
    #Uses scipy for Matlab files before v7.3
    dict_images = scipy.io.loadmat(inputdata)
    #Select the ndarray | the routine takes the first one
    for str_key,key_value in dict_images.items():
        #print(str_key)
        #print(type(key_value))
        if str(type(key_value))=="<class 'numpy.ndarray'>":
            images=key_value
            print(images.shape)
            break
    
    #Expand dims to always have 4d geometry
    if len(images.shape)<4:
        images=images[:,:,:,np.newaxis]
    img_shape=images.shape

    #Amount of used tiles
    #How many tiles are maximally needed n=(img-overlap)/(input-overlap); assumes that the tiles just cover the whole image
    max_tilesX=np.ceil( (img_shape[1]-overlapX) / float(Inputlayer.geometry[0]-overlapX) )
    max_tilesY=np.ceil( (img_shape[2]-overlapY) / float(Inputlayer.geometry[1]-overlapY) )
    
    if tilesX<=0:
        tilesX=1
    elif tilesX==np.inf:
        tilesX=max_tilesX
        
    if tilesY<=0:
        tilesY=tilesX
    elif tilesY==np.inf:
        tilesY=max_tilesY

    tilesX=int(min(max_tilesX,tilesX))
    tilesY=int(min(max_tilesY,tilesY))
    print('tilesX:',tilesX,'tilesY:',tilesY)


    #Init datastructures for storage, using a list structure filled with numpy arrays for every regarded population
    #Save using h5py
    f=h5py.File(str(savefile+'.h5'),'w', rdcc_nbytes=1024*1024*1024*1, rdcc_w0=1)#http://docs.h5py.org/en/stable/high/file.html#chunk-cache Might just work from version 2.9
    for pop in populations():
        if compression==1:
            f.create_dataset('r_'+pop.name, (img_shape[0],tilesX,tilesY,pop.width,pop.height,pop.depth), dtype='float32', chunks=(10,tilesX,tilesY,pop.width,pop.height,pop.depth), compression="gzip",  compression_opts=1)#gzip compression costs time, lzf is faster, BUT MATLAB DOESNT LOAD IT; chucks cost a bit time, chuncks=True is much faster than defining a size
        else:
            f.create_dataset('r_'+pop.name, (img_shape[0],tilesX,tilesY,pop.width,pop.height,pop.depth), dtype='float32')


    #Init time measurement
    t1=time.time()
    
    ##Present data to the network
    for i in range(0,img_shape[0]):
        # Time measurement
        if i==1:
            t0=time.time()
        if i==2:
            t1000=time.time()
            expected=(t1000-t0)*(img_shape[0]-1)/1
            days=int(expected / 86400)
            hours=int(expected % 86400/ 3600)
            minutes=int(expected % 3600/ 60)
            seconds=int(expected % 60)
            print('Time expected: ' + str(days) + ' - ' + str(hours) + ':' + str(minutes) + ':'+ str(seconds) + ' d-h:m:s')

        #Generate empty image of input layer size
        image_patch=np.zeros(Inputlayer.geometry)
        
        #Decide if images have to be preprocessed
        if not img_shape[3]==Inputlayer.geometry[2]:
            #It is assumed that the image should be whitened, else something is totally wrong
            tmp=np.squeeze(np.array(Whiten.whiten(data=np.expand_dims(images[i,:,:,:].squeeze(),0), sizeX=-1, sizeY=-1, f_0=-1)))
            image_whiten=np.zeros((tmp.shape[0],tmp.shape[1],2))
            image_whiten[:,:,0]=np.fmax(tmp,0)
            image_whiten[:,:,1]=np.fmax(-tmp,0)
            
        for ti_x in range(0,tilesX):
            for ti_y in range(0,tilesY):
                
                #If input image size differs cut out patch at the center and place it at the center
                #Start and end for the cut from the image
                x_start=max(ti_x*Inputlayer.geometry[0] - ti_x*overlapX,0)
                x_end=min(x_start+min(Inputlayer.geometry[0],img_shape[1]),img_shape[1])
                y_start=max(ti_y*Inputlayer.geometry[1] - ti_y*overlapY,0)
                y_end=min(y_start+min(Inputlayer.geometry[1],img_shape[2]),img_shape[2])


                #Generate empty image of input layer size
                image_patch=np.zeros(Inputlayer.geometry)
                    
                #Decide if images have to be preprocessed
                if img_shape[3]==Inputlayer.geometry[2]:
                    #No preprocessing in terms of whitening needed
                    image_patch[0:x_end-x_start,0:y_end-y_start]=images[i,x_start:x_end,y_start:y_end,:]
                else:
                    image_patch[0:x_end-x_start,0:y_end-y_start]=image_whiten[x_start:x_end,y_start:y_end,:]
                
                #Determine the norm factor based on the image containing target and mask
                if norm_factor==0 and norm_max_value>0:
                    norm_factor=norm_max_value/(np.max(np.abs(image_patch))+np.finfo(float).eps)
                

                #Reset the network activities
                for pop in populations():
                    pop.reset(attributes=['r'])
                    pop.mp=0
                    pop.I_exc=0
                    pop.I_ff=0
                    pop.I_inh=0
                    if hasattr(pop,'I_fb')==True:
                        pop.I_fb=0
                    if hasattr(pop,'I_lat')==True:
                        pop.I_lat=0
                    if hasattr(pop,'r_slow')==True:
                        pop.r_slow=0
                        
                #Set input patch
                Inputlayer.input=image_patch*norm_factor
                
                #Run network
                simulate(presentation_time)
                
                #Run additional time steps to compensate synaptic delays for recording
                t_prev=0#Cummulated time the network has additionally runned
                for it_add in range(len(unique_delay)):
                    #Time the network should run additionally
                    t_add=unique_delay[it_add]-t_prev
                    if t_add>0:
                        simulate(t_add)

                    ## Record activities
                    for pop in populations():
                        # Do not record from input populations
                        #if pop.name!=Inputlayer.name:
                        #Get responses of each neuron in the population
                        if delays[pop.name]==unique_delay[it_add]:
                            f[str('r_'+pop.name)][i,ti_x,ti_y,:,:,:] = pop.get("r")
                            
                    t_prev=unique_delay[it_add]
                    
                #Output
                if (i<=4) and (ti_x==ti_y):
                    #Counting output length for nicer multiline output
                    prev_lines=0
                    output=''
                    for pop in populations():
                        output=output+pop.name+'#Mean:'+"%.3f"%(np.mean(pop.r))+' Max:'+"%.3f"%(np.max(pop.r))
                        if len(output)-prev_lines>200:
                            output=output+'\n'
                            prev_lines=len(output)
                        else:
                            output=output+'|'
                    print(output)
                    output='Image:'+str(i)+' '
                
    ## Save
    print("Saving results...")
    f.close()
