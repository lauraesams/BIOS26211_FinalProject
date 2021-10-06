#----------------------imports and environment---------------------------------
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
from ANNarchy import *
setup(dt=1.0)
from time import *
import numpy as np
import scipy.io as sio

#------------------------global Variables------------------------------------
nbrOfPatches = 200000 # total number of patches for training
duration = 125 # time (in ms) that each patch is presented
patchsize = 18 # size of a patch
inputNeurons = patchsize*patchsize*2 # number of LGN neurons
n_Exc = patchsize*patchsize # number of excitatory neurons
n_Inh = int(n_Exc/4) # number of inhibitory neurons
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
spkNeurV1 = Neuron(parameters = params,equations=neuron_eqs,spike="""(vm>VT) and (state==0)""",
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

parameterFF="""
urefsquare = 60.0 	:projection
thetaLTD = -70.6 	:projection
thetaLTP = -45.3	:projection
aLTD = 0.00014*0.6	:projection
aLTP = 0.00018*0.6	:projection
wMin = 0.0			:projection
wMax = 5.0			:projection
"""

#-70.6
#-45.3#-53.5 
ffSyn = Synapse( parameters = parameterFF,
    equations= equatSTDP, 
    pre_spike='''g_target += w''')


#------STDP Synapse like above, with other parameters for input -> inhibitory ----#
parameterInptInhib="""
urefsquare = 55.0 	:projection
thetaLTD = -70.6 	:projection
thetaLTP = -45.3 	:projection
aLTD = 0.00014 *0.3:projection
aLTP = 0.00018 *0.3:projection
wMin = 0.0			:projection
wMax = 3.0			:projection	
"""

ff2Syn = Synapse(parameters = parameterInptInhib,
    equations=equatSTDP,
    pre_spike = '''g_target +=w ''')          
#------STDP Synapse like above, with other parameters for exitatory -> inhibitory ----#
parameterInhib="""
urefsquare = 55.0	 :projection
thetaLTD = -70.6 	 :projection
thetaLTP = -45.3  	 :projection
aLTD = 0.000012 *0.03:projection	
aLTP = 0.000015 *0.03:projection
wMin = 0.0			 :projection
wMax = 0.7			 :projection
"""

InhibSyn = Synapse( parameters = parameterInhib,
    equations=equatSTDP , 
    pre_spike='''g_target += w''')

 
##-> delete later, if not necessary
#------------- iSTDP Synapse for Inhibitory- to Exitatory- Layer -----------------------#

equatInhib = ''' dtracePre /dt = - tracePre/ taupre
                 dtracePost/dt = - tracePost/ taupre'''

parameter_iSTDPback="""
taupre = 10				:projection
aPlus = 4.0*10**(-6)	:projection
Offset = 35.0 *10**(-2)	:projection
hMin=0.0				:projection
hMax =0.5				:projection
"""
inhibBackSynapse = Synapse(parameters = parameter_iSTDPback,
                    equations = equatInhib,
                    pre_spike ='''
                         g_target +=w
                         w+= aPlus * (tracePost - Offset)*(1+w) :min=hMin, max=hMax
                         tracePre  += 1 ''',
                    post_spike='''
                         w+=aPlus * (tracePre)*(1+w) :min=hMin, max=hMax
                         tracePost += 1''')
                         
#------------------- iSTDP Synapse for Lateral Inhibitory ----------------------------#

parameter_iSTDPlat="""
taupre = 10				:projection
aPlus = 4.0*10**(-6)	:projection
Offset = 45.0*10**(-2)	:projection
hMin=0.0				:projection
hMax =0.5				:projection
"""
inhibLatSynapse = Synapse(parameters = parameter_iSTDPlat,
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
popLGN = Population(geometry=(patchsize,patchsize,2),neuron=spkNeurLGN )
popV1 = Population(geometry=n_Exc,neuron=spkNeurV1, name="V1")
popInhibit = Population(geometry=n_Inh, neuron = spkNeurV1)
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
    synapse=ffSyn
).connect_all_to_all(weights = Uniform(0.025,1.0))

projLGN_Inhib = Projection(
    pre = popLGN,
    post= popInhibit,
    target='Exc',
    synapse=ff2Syn
).connect_all_to_all(weights = Uniform(0.0,0.1))

projV1_Inhib = Projection(
    pre = popV1,
    post = popInhibit,
    target = 'Exc',
    synapse = InhibSyn
).connect_all_to_all(weights = Uniform(0.0,0.1))

projInhib_V1 = Projection(
    pre = popInhibit,
    post= popV1,
    target = 'Inh',
    synapse = inhibBackSynapse
).connect_all_to_all(weights = 0.0)

projInhib_Lat = Projection(
    pre = popInhibit,
    post = popInhibit,
    target = 'Inh',
    synapse = inhibLatSynapse
).connect_all_to_all(weights = 0.0)

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
    if not os.path.exists('output_L4'):
        os.mkdir('output_L4')
    if not os.path.exists('output_L4/exitatory'):
        os.mkdir('output_L4/exitatory')
    if not os.path.exists('output_L4/inhibitory'):
        os.mkdir('output_L4/inhibitory')
    if not os.path.exists('output_L4/V1Layer'):
        os.mkdir('output_L4/V1Layer')
    if not os.path.exists('output_L4/V1toIN'):
        os.mkdir('output_L4/V1toIN')
#------------------------------------------------------------------------------
def normWeights():

    weights= projLGN_V1.w
    for i in range(n_Exc):
        onoff  = np.reshape(weights[i],(patchsize,patchsize,2))
        onNorm = np.sqrt(np.sum(onoff[:,:,0]**2))
        offNorm= np.sqrt(np.sum(onoff[:,:,1]**2))
        onoff[:,:,0]*=offNorm/onNorm
        weights[i] = np.reshape(onoff,patchsize*patchsize*2)
    projLGN_V1.w = weights
    weights = projLGN_Inhib.w
    for i in range(n_Inh):
        onoff  = np.reshape(weights[i],(patchsize,patchsize,2))
        onNorm = np.sqrt(np.sum(onoff[:,:,0]**2))
        offNorm= np.sqrt(np.sum(onoff[:,:,1]**2))
        onoff[:,:,0]*=offNorm/onNorm
        weights[i] = np.reshape(onoff,patchsize*patchsize*2)
    projLGN_Inhib.w = weights
#-------------------------------------------------------------------------------
def saveWeights(nbr =0):
    np.savetxt('output_L4/exitatory/V1weight_%i.txt'%(nbr)  ,projLGN_V1.w)
    np.savetxt('output_L4/exitatory/InhibW_%i.txt'%(nbr), projLGN_Inhib.w)
    np.savetxt('output_L4/V1toIN/V1toIN_%i.txt'%(nbr)  ,projV1_Inhib.w)
    np.savetxt('output_L4/inhibitory/INtoV1_%i.txt'%(nbr) ,projInhib_V1.w)
    np.savetxt('output_L4/inhibitory/INLat_%i.txt'%(nbr) ,projInhib_Lat.w)
#------------------------------main function------------------------------------
def run():
    createDir()

    matData = sio.loadmat('./input/IMAGES.mat')
    images = preprocessData(matData)

    step = 100
    compile()
    #------- neuron Monitors --------#
    V1Mon = Monitor(popV1,['spike'])
    InhibMon=Monitor(popInhibit,['spike'])
    #--------synapse Monitors--------#
    dendriteFF = projLGN_V1.dendrite(0)
    ffWMon = Monitor(dendriteFF,['w','ltdTerm','ltpTerm'],period=duration*step)
    dendriteFFI = projLGN_Inhib.dendrite(0)
    ffIMon= Monitor(dendriteFFI,'w',period=duration*step)
    dendriteExIn = projV1_Inhib.dendrite(0)
    exInMon = Monitor(dendriteExIn,'w',period=duration*step)
    dendriteInEx = projInhib_V1.dendrite(0)
    inExMon = Monitor(dendriteInEx,'w',period=duration*step)
    dendriteInLat = projInhib_Lat.dendrite(0)
    latMon = Monitor(dendriteInLat,'w',period=duration*step)
    #------Spike Rate Monitor--------#
    rec_frEx = np.zeros((int(nbrOfPatches/step),n_Exc))
    rec_frInh= np.zeros((int(nbrOfPatches/step),n_Inh))
    t1 = time()

    print('start Simulation')
    for i in range(nbrOfPatches):
        setInputPatch(images,patchsize)
        simulate(duration)       
        if ((i*duration)%20000) == 0:
            normWeights() 
        if ((i%step) == 0):
            spikesEx = V1Mon.get('spike')
            spikesInh = InhibMon.get('spike')
            for j in range(n_Exc):
                rateEx = len(spikesEx[j])*1000./(duration*step)
                rec_frEx[int(i/step),j] = rateEx
                if (j < (n_Inh)):
                    rateInh = len(spikesInh[j])*1000./(duration*step)
                    rec_frInh[int(i/step),j] = rateInh         
        if((i%(nbrOfPatches/20)) == 0):
            print("Round %i of %i" %(i,nbrOfPatches))
            saveWeights(i)           
    t2 = time()
    saveWeights(nbrOfPatches)
    #------get recording data---------#
    
    ffW = ffWMon.get('w')
    ffLTD = ffWMon.get('ltdTerm')
    ffLTP = ffWMon.get('ltpTerm')
    ffI = ffIMon.get('w')
    exInW = exInMon.get('w')
    inExW = inExMon.get('w')
    inLatW= latMon.get('w')
    #--------print Time difference-----------#
    print("time of simulation= "+str((duration*nbrOfPatches)/1000)+" s")
    print("time of calculation= "+str(t2-t1)+" s")
    print("factor= "+str((t2-t1)/(duration*nbrOfPatches/1000)))
    
    #----------------plot output---------------#

    for i in range(1):
        fig = plt.figure()
        fig.add_subplot(4,1,1) 
        plt.plot(rec_frEx[:,0+(4*i)])
        fig.add_subplot(4,1,2) 
        plt.plot(rec_frEx[:,1+(4*i)])
        fig.add_subplot(4,1,3) 
        plt.plot(rec_frEx[:,2+(4*i)])
        fig.add_subplot(4,1,4) 
        plt.plot(rec_frEx[:,3+(4*i)])
        plt.savefig('output_L4/V1Layer/frEx_'+str(i)+'.png')


    np.save('output_L4/frExc',rec_frEx)
    np.save('output_L4/frInh',rec_frInh)


    plt.figure()   
    plt.plot(np.mean(ffW,axis=1))
    plt.savefig('output_L4/ffWMean.png')
    print(np.mean(ffW))

    plt.figure()   
    plt.plot(np.mean(ffLTD,axis=1))
    plt.savefig('output_L4/ffWLTD.png')
    print(np.sum(ffLTD))
    plt.figure()   
    plt.plot(np.mean(ffLTP,axis=1))
    plt.savefig('output_L4/ffWLTP.png')
    print(np.sum(ffLTP))
    plt.figure()
    plt.plot(np.mean(ffI,axis=1))
    plt.savefig('output_L4/ffIMean.png')

    plt.figure()
    plt.plot(np.mean(exInW,axis=1))
    plt.savefig('output_L4/exInMeanW.png')

    plt.figure()
    plt.plot(inExW)
    plt.savefig('output_L4/InExW.png')
    plt.figure()
    plt.plot(inLatW)
    plt.savefig('output_L4/inLatW.png')

    print("finish")
#------------------------------------------------------------------------------------
if __name__ == "__main__":
    if os.path.isfile('./input/IMAGES.mat'):
        run()
    else:
        print("""No IMAGES.mat found, please download the file from:
        https://www.rctn.org/bruno/sparsenet/IMAGES.mat
        and put in the code directory""")
