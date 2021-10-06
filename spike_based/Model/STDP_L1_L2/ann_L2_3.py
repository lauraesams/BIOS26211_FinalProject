#----------------------imports and environment---------------------------------
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
from ANNarchy import *
setup(dt=1.0)
from time import *
import scipy.io as sio
import numpy as np

#------------------------global Variables------------------------------------
nbrOfPatches = 400000 # total number of patches for training
duration = 125 # time (in ms) that each patch is presented
patchsize = 18 # size of a patch
inputNeurons = patchsize*patchsize*2 # number of LGN neurons
n_Exc_1 = patchsize*patchsize # number of excitatory neurons in layer 4
n_Exc_2 = n_Exc_1 # number of excitatory neurons in layer 2/3
n_Inh_1 = int(n_Exc_1/4) # number of inhibitory neurons in layer 4
n_Inh_2 = int(n_Exc_2/4) # number of inhibitory neurons in layer 2/3

#---------------------------------neuron definitions-------------------------

## Neuron Model for LGN/Input Layer ##
params = """
EL = -70.4		:population
VTrest = -50.4	:population
taux = 15.0		:population
"""
#if g_exc > -50.4: 1 else:
inpt_eqs ="""
    dg_exc/dt = EL/1000 : min=EL, init=-70.6
    Spike = if state == 1:1.0 else: 0.0
    dresetvar/dt =if state==1:+1 else: -resetvar
    dxtrace/dt = if state==1:+1/taux else: - xtrace/taux  :init=0.0
    state = if state >0 : -1 else: 0
    """

spkNeurLGN = Neuron(parameters=params,
                          equations=inpt_eqs,
                          reset="""g_exc=EL 
                                   state = 1""", 
                          spike="""g_exc > VTrest""")

## Neuron Model for V1-Layer, after Clopath et al.(2010) ##
params = """
gL = 30.0		:population
DeltaT = 2.0 	:population
tauw = 144.0 	:population
a = 4.0 		:population	
b = 0.0805 		:population
EL = -70.6 		:population
C = 281.0 		:population
tauz = 40.0		:population
tauVT= 50.0		:population
Isp = 400.0		:population
VTMax = -30.4	:population
VTrest = -50.4	:population
taux = 15.0 	:population	
tauLTD = 10.0	:population
tauLTP= 7.0 	:population	
taumean = 750.0 :population
tau_gExc = 1.0	:population
tau_gInh= 10.0	:population
sigma = 0.0		:population
"""

neuron_eqs = """
noise = Normal(0.0,1.0)
dvm/dt = if state>=2:+3.462 else: if state==1:-(vm+51.75)+1/C*(Isp - (wad+b))+g_Exc-g_Inh else:1/C * ( -gL * (vm - EL) + gL * DeltaT * exp((vm - VT) / DeltaT) - wad + z ) + g_Exc -g_Inh: init = -70.6
dvmean/dt = (pos(vm - EL)**2 - vmean)/taumean    :init = 0.0
dumeanLTD/dt = (vm - umeanLTD)/tauLTD : init=-70.0
dumeanLTP/dt = (vm - umeanLTP)/tauLTP : init =-70.0
dxtrace/dt = if state==1:+1/taux else: - xtrace/taux  :init=0.0
dwad/dt = if state ==2:0 else:if state==1:+b/tauw else: (a * (vm - EL) - wad)/tauw : init = 0.0
dz/dt = if state==1:-z+Isp-10 else:-z/tauz  : init = 0.0
dVT/dt =if state==1: +(VTMax - VT)-0.4 else:(VTrest - VT)/tauVT  : init=-50.4
dg_Exc/dt = -g_Exc/tau_gExc
dg_Inh/dt = -g_Inh/tau_gInh
dresetvar/dt =if state==1:+1 else: -resetvar
Spike = if state == 1:1.0 else: 0.0
state = if state > 0: state-1 else:0
           """
spkNeuron = Neuron(parameters = params,equations=neuron_eqs,spike="""(vm>VT) and (state==0)""",
                         reset="""vm = 29.0
                                  state = 2.0 
                                  VT = VTMax
                                """)

#----------------------------------synapse definitions----------------------

#----- Synapse from Poisson to Input-Layer -----#
inputSynapse =  Synapse(
    parameters = "",
    equations = "",
    pre_spike = """
        g_target += w
                """
)

#--STDP Synapses after Clopath et. al(2010)for Input- to Exitatory- Layer--#
equatSTDP = """
    ltdTerm = if w>wMin : (aLTD*(post.vmean/urefsquare)*pre.Spike * pos(post.umeanLTD - thetaLTD)) else : 0.0
    ltpTerm = if w<wMax : (aLTP * pos(post.vm - thetaLTP) *(pre.xtrace)* pos(post.umeanLTP - thetaLTD)) else : 0.0
      dw/dt = ( -ltdTerm + ltpTerm) :min=0.0,explicite"""

#-------STDP synapse from Clopath et. al(2010) -> V1 to V2 ---------#

parameterFFV2="""
urefsquare = 50.0 	:projection
thetaLTD = -70.6 	:projection
thetaLTP = -45.3	:projection
aLTD = 0.00012*0.625  :projection
aLTP = 0.00014*0.625	:projection
wMin = 0.0			:projection
wMax = 3.25			:projection
"""

#-70.6
#-45.3#-53.5 
ffV2Syn = Synapse( parameters = parameterFFV2,
    equations= equatSTDP, 
    pre_spike='''g_target += w''') 

#------STDP Synapse like above, with other parameters for exitatory -> inhibitory ----#
parameterV2toIL2="""
urefsquare = 50.0	 :projection
thetaLTD = -70.6 	 :projection
thetaLTP = -45.3  	 :projection
aLTD = 0.000014 *0.4 :projection	
aLTP = 0.000018 *0.4 :projection
wMin = 0.0			 :projection
wMax = 1.0			 :projection
"""

V2toIL2Syn = Synapse( parameters = parameterV2toIL2,
    equations=equatSTDP , 
    pre_spike='''g_target += w''')

#------STDP Synapse like above, with other parameters for exitatory -> inhibitory ----#
parameterV1toIL2="""
urefsquare = 45.0	 :projection
thetaLTD = -70.6 	 :projection
thetaLTP = -45.3  	 :projection
aLTD = 0.00012 *0.4 :projection	
aLTP = 0.00015 *0.4 :projection
wMin = 0.0			 :projection
wMax = 1.0			 :projection
"""

V1toIL2Syn = Synapse( parameters = parameterV1toIL2,
    equations=equatSTDP , 
    pre_spike='''g_target += w''')

##-> delete later, if not necessary

#########################################################################################
#------------- iSTDP Synapse for Inhibitory- to Exitatory- Layer -----------------------#

equatInhib = ''' dtracePre /dt = - tracePre/ taupre
                 dtracePost/dt = - tracePost/ taupre'''

#------------- iSTDP Synapse for Inhibitory2 - to Exitatory2 - Layer -----------------------#


parameterIL2toV2="""
taupre = 10				:projection
aPlus = 8.*10**(-6)	:projection
Offset = 50.0 *10**(-2)	:projection
hMin=0.0				:projection
hMax =0.5				:projection
"""
IL2toV2Syn = Synapse(parameters = parameterIL2toV2,
                    equations = equatInhib,
                    pre_spike ='''
                         g_target +=w
                         w+= aPlus * (tracePost - Offset)*(1+w) :min=hMin, max=hMax
                         tracePre  += 1 ''',
                    post_spike='''
                         w+=aPlus * (tracePre)*(1+w) :min=hMin, max=hMax
                         tracePost += 1''')
                         
#------------------- iSTDP Synapse for Lateral Inhibitory 2----------------------------#

parameterIL2Lat="""
taupre = 10				:projection
aPlus = 4.*10**(-6)	:projection
Offset = 60.0*10**(-2)	:projection
hMin=0.0				:projection
hMax =0.5				:projection
"""
latIL2Syn = Synapse(parameters = parameterIL2Lat,
                    equations = equatInhib,
                    pre_spike ='''
                         g_target +=w
                         w+= aPlus * (tracePost - Offset)*(1+w) :min=hMin, max=hMax
                         tracePre  += 1 ''',
                    post_spike='''
                         w+=aPlus * (tracePre)*(1+w) :min=hMin, max=hMax
                         tracePost += 1''')

#-----------------------population defintions-----------------------------------#
popInput = PoissonPopulation(geometry=(patchsize,patchsize,2),rates=50.0)
popLGN = Population(geometry=(patchsize,patchsize,2),neuron=spkNeurLGN,name="LGN" )
popV1 = Population(geometry= n_Exc_1,neuron=spkNeuron, name="Ex1")
popIL1 = Population(geometry= n_Inh_1, neuron = spkNeuron,name="IL1")
popV2 = Population(geometry= n_Exc_2,neuron=spkNeuron, name="Ex2")
popIL2 = Population(geometry= n_Inh_2, neuron = spkNeuron,name="IL2")
#-----------------------projection definitions----------------------------------
#projPreLayer_PostLayer
projInput_LGN = Projection(
    pre = popInput,
    post = popLGN,
    target = 'exc',
    synapse = inputSynapse
).connect_one_to_one(weights = 30.0)

projLGN_V1 = Projection(
    pre=popLGN, 
    post=popV1, 
    target='Exc',
    synapse=inputSynapse
).connect_all_to_all(weights = Uniform(0.025,1.0))

projLGN_IL1 = Projection(
    pre = popLGN,
    post= popIL1,
    target='Exc',
    synapse= inputSynapse
).connect_all_to_all(weights = Uniform(0.01,0.15))

projV1_IL1 = Projection(
    pre = popV1,
    post = popIL1,
    target = 'Exc',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.01,0.15))

projIL1_V1 = Projection(
    pre = popIL1,
    post= popV1,
    target = 'Inh',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.01,0.15))

projIL1_Lat = Projection(
    pre = popIL1,
    post = popIL1,
    target = 'Inh',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.01,0.15))

projV1_V2 = Projection(
    pre = popV1,
    post = popV2,
    target = 'Exc',
    synapse = ffV2Syn
).connect_all_to_all(weights = Uniform(0.20,1.25), delays=Uniform(1,3))

projV2_IL2 = Projection(
    pre = popV2,
    post = popIL2,
    target = 'Exc',
    synapse = V2toIL2Syn
).connect_all_to_all(weights = Uniform(0.01,0.5), delays=Uniform(1,3))

projV1_IL2 = Projection(
    pre = popV1,
    post = popIL2,
    target = 'Exc',
    synapse = V1toIL2Syn
).connect_all_to_all(weights = Uniform(0.01,0.1))

projIL2_V2 = Projection(
    pre = popIL2,
    post= popV2,
    target = 'Inh',
    synapse = IL2toV2Syn
).connect_all_to_all(weights = Uniform(0.1,0.3))

projIL2_Lat = Projection(
    pre = popIL2,
    post = popIL2,
    target = 'Inh',
    synapse = latIL2Syn
).connect_all_to_all(weights = Uniform(0.05,0.13))

#----------------------------further functions---------------------------------
def preprocessData(matData):
    # function to split the prewhitened images into on and off counterparts
    images = matData['IMAGES']
    w,h,n_images = np.shape(images)
    new_images = np.zeros((w,h,2,n_images))
    for i in range(n_images):
        new_images[images[:,:,i] > 0, 0, i] = images[images[:,:,i] > 0, i]
        new_images[images[:,:,i] < 0, 1, i] = images[images[:,:,i] < 0, i]*-1

    return(new_images)
#------------------------------------------------------------------------------
def getInput(images,patchsize):
    pictNbr = np.random.randint(0,10)
    length,width = images.shape[0:2]
    xPos = np.random.randint(0,length-patchsize)
    yPos = np.random.randint(0,width-patchsize)
    inputPatch = images[xPos:xPos+patchsize,yPos:yPos+patchsize,:,pictNbr]
    maxVal = np.max(images[:,:,:,pictNbr])
    return(inputPatch,maxVal)
#------------------------------------------------------------------------------
def setInputPatch(images,patchsize):
    inputPatch,maxVal = getInput(images,patchsize)
    ## flip the patch along the x- and y- axis to increase the number of possible orientations
    if np.random.rand() <0.5:
        inputPatch=np.fliplr(inputPatch)
    if np.random.rand()<0.5:
        inputPatch=np.flipud(inputPatch)
    ## set the target rate of the Poisson Input-Population
    popInput.rates = inputPatch/maxVal *100.0

#-------------------------------------------------------------------------------
def createDir():
    if not os.path.exists('output_L2_3'):
        os.mkdir('output_L2_3')
    if not os.path.exists('output_L2_3/exitatory'):
        os.mkdir('output_L2_3/exitatory')
    if not os.path.exists('output_L2_3/inhibitory'):
        os.mkdir('output_L2_3/inhibitory')
    if not os.path.exists('output_L2_3/V1Layer'):
        os.mkdir('output_L2_3/V1Layer')
    if not os.path.exists('output_L2_3/V2Layer'):
        os.mkdir('output_L2_3/V2Layer')
    if not os.path.exists('output_L2_3/InhibitLayer1'):
        os.mkdir('output_L2_3/InhibitLayer1')
    if not os.path.exists('output_L2_3/InhibitLayer2'):
        os.mkdir('output_L2_3/InhibitLayer2')
#-------------------------------------------------------------------------------
def saveValues(values,valName,name):
    if (name == 'Exi'):
        np.save('output/V1Layer/'+valName,values)
    if (name == 'Inhib'):
        np.save('output/InhibitLayer/'+valName,values)
#-------------------------------------------------------------------------------
def saveWeights(nbr =0):
    np.savetxt('output_L2_3/exitatory/V2weight_%i.txt'%(nbr), projV1_V2.w)
    np.savetxt('output_L2_3/exitatory/V2toIN2_%i.txt'%(nbr), projV2_IL2.w)
    np.savetxt('output_L2_3/exitatory/V1toIN2_%i.txt'%(nbr), projV1_IL2.w)
    np.savetxt('output_L2_3/inhibitory/IN2toV2_%i.txt'%(nbr), projIL2_V2.w)
    np.savetxt('output_L2_3/inhibitory/IN2Lat_%i.txt'%(nbr), projIL2_Lat.w)
#------------------------------------------------------------------------------
def loadWeights():
    projLGN_V1.w  = np.loadtxt('output_L4/exitatory/V1weight_200000.txt')
    projLGN_IL1.w = np.loadtxt('output_L4/exitatory/InhibW_200000.txt')
    projV1_IL1.w  = np.loadtxt('output_L4/V1toIN/V1toIN_200000.txt')
    projIL1_V1.w  = np.loadtxt('output_L4/inhibitory/INtoV1_200000.txt')
    projIL1_Lat.w = np.loadtxt('output_L4/inhibitory/INLat_200000.txt')
#------------------------------------------------------------------------------
def saveDelays():
    np.savetxt('output_L2_3/exitatory/V1toV2_Delay.txt', projV1_V2.delay)
#------------------------------main function------------------------------------
def run():
    createDir()

    matData = sio.loadmat('./input/IMAGES.mat')
    images = preprocessData(matData)

    compile()
    loadWeights()
    saveDelays()
    #------- neuron Monitors --------#

    V1Mon = Monitor(popV1,['spike'])
    V2Mon = Monitor(popV2,['spike'])
    InhibMon=Monitor(popIL1,['spike'])
    InhibMon2 = Monitor(popIL2,['spike'])
    #--------synapse Monitors--------#

    dendriteV1toV2 = projV1_V2.dendrite(0)
    v1v2Mon = Monitor(dendriteV1toV2,'w',period=duration)

    dendriteV1toIn2 = projV1_IL2.dendrite(0)    
    v1IL2Mon = Monitor(dendriteV1toIn2,'w',period=duration)

    dendriteV2toIn2 = projV2_IL2.dendrite(0)    
    v2IL2Mon = Monitor(dendriteV2toIn2,'w',period=duration)

    dendriteInExV2 = projIL2_V2.dendrite(0)
    inEx2Mon = Monitor(dendriteInExV2,'w',period=duration)

    dendriteInLat2 = projIL2_Lat.dendrite(0)
    lat2Mon = Monitor(dendriteInLat2,'w',period=duration)

    #------Spike Rate Monitor--------#
    rec_V1_frEx = np.zeros((nbrOfPatches,n_Exc_1))

    rec_V2_frEx = np.zeros((nbrOfPatches,n_Exc_2))

    rec_frInh = np.zeros((nbrOfPatches,n_Inh_1))
    rec_frInh2 = np.zeros((nbrOfPatches,n_Inh_2))

    t1 = time()

    print('start Simulation')
    for i in range(nbrOfPatches):
        setInputPatch(images,patchsize)
        simulate(duration)       
        spikesEx = V1Mon.get('spike')
        spikesV2 = V2Mon.get('spike')    
        spikesInh = InhibMon.get('spike')
        spikesInh2 = InhibMon2.get('spike')
        for j in range(n_Exc_1):
            rateEx = len(spikesEx[j])*1000/duration
            rec_V1_frEx[i,j] = rateEx
            if (j < (n_Inh_1)):
                rateInh = len(spikesInh[j])*1000/duration
                rec_frInh[i,j] = rateInh         
        for j in range(n_Exc_2):
            rateExV2 = len(spikesV2[j])*1000/duration
            rec_V2_frEx[i,j] = rateExV2
            if (j < (n_Inh_2)):
                rateInh2 = len(spikesInh2[j])*1000/duration
                rec_frInh2[i,j] = rateInh2

        if((i%(nbrOfPatches/20)) == 0):
            print("Round %i of %i" %(i,nbrOfPatches))
            saveWeights(i)           
    t2 = time()
    saveWeights(nbrOfPatches)

    #------get recording data---------#

    v2W = v1v2Mon.get('w')
    v2IL2W = v2IL2Mon.get('w')
    v1IL2W = v1IL2Mon.get('w')
    inV2W = inEx2Mon.get('w')
    inLat2W = lat2Mon.get('w')

    #--------print Time difference-----------#
    print("time of simulation= "+str((duration*nbrOfPatches)/1000)+" s")
    print("time of calculation= "+str(t2-t1)+" s")
    print("factor= "+str((t2-t1)/(duration*nbrOfPatches/1000)))
      #----------------plot output---------------#


    for i in range(1):
        fig = plt.figure()
        fig.add_subplot(4,1,1) 
        plt.plot(rec_V1_frEx[:,0+(4*i)])
        fig.add_subplot(4,1,2) 
        plt.plot(rec_V1_frEx[:,1+(4*i)])
        fig.add_subplot(4,1,3) 
        plt.plot(rec_V1_frEx[:,2+(4*i)])
        fig.add_subplot(4,1,4) 
        plt.plot(rec_V1_frEx[:,3+(4*i)])
        plt.savefig('output_L2_3/V1Layer/frEx_'+str(i)+'.png')

    for i in range(4):
        fig = plt.figure()
        fig.add_subplot(4,1,1) 
        plt.plot(rec_V2_frEx[:,0+(4*i)])
        fig.add_subplot(4,1,2) 
        plt.plot(rec_V2_frEx[:,1+(4*i)])
        fig.add_subplot(4,1,3) 
        plt.plot(rec_V2_frEx[:,2+(4*i)])
        fig.add_subplot(4,1,4) 
        plt.plot(rec_V2_frEx[:,3+(4*i)])
        plt.savefig('output_L2_3/V2Layer/frEx_'+str(i)+'.png')


    for i in range(4):
        fig = plt.figure()
        fig.add_subplot(4,1,1) 
        plt.plot(rec_frInh2[:,0+(4*i)])
        fig.add_subplot(4,1,2) 
        plt.plot(rec_frInh2[:,1+(4*i)])
        fig.add_subplot(4,1,3) 
        plt.plot(rec_frInh2[:,2+(4*i)])
        fig.add_subplot(4,1,4) 
        plt.plot(rec_frInh2[:,3+(4*i)])
        plt.savefig('output_L2_3/InhibitLayer2/frEx_'+str(i)+'.png')


    print('MeanFr E1: ',np.mean(rec_V1_frEx))
    print('MeanFr E2: ',np.mean(rec_V2_frEx))
    print('MeanFr I1: ',np.mean(rec_frInh))
    print('MeanFr I2: ',np.mean(rec_frInh2))

    plt.figure()   
    plt.plot(np.mean(v2W,axis=1))
    plt.savefig('output_L2_3/V2Mean.png')

    plt.figure()   
    plt.plot(np.mean(v2IL2W,axis=1))
    plt.savefig('output_L2_3/V2toIL2Mean.png')

    plt.figure()   
    plt.plot(np.mean(v1IL2W,axis=1))
    plt.savefig('output_L2_3/V1toIL2Mean.png')

    plt.figure()   
    plt.plot(inV2W)
    plt.savefig('output_L2_3/IL2toV2Mean.png')

    plt.figure()   
    plt.plot(inLat2W)
    plt.savefig('output_L2_3/IL2LatMean.png')

    print("finish")
#------------------------------------------------------------------------------------
run()
