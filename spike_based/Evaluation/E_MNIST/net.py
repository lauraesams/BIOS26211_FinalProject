#from ANNarchy import Neuron,PoissonPopulation,Population,Projection,Synapse,Uniform
from ANNarchy import *
#------------------------global Variables------------------------------------
patchsize = 18#8
numberOfInputN = patchsize*patchsize*2
nbr_V1N = patchsize*patchsize# because ON-OFF 
nbr_IL1N = nbr_V1N/4
nbr_V2N = nbr_V1N
nbr_IL2N =nbr_V2N/4
#---------------------------------neuron definitions-------------------------

## Neuron Model for LGN/Input Layer ##
params = """
EL = -70.6
VTrest = -50.4
taux = 15.0
"""

inpt_eqs ="""
    dg_exc/dt = EL/1000 : min=EL, init=-70.6
    Spike =if g_exc > -50.4: 1 else: 0.0
    dresetvar / dt = 1/(1.0) * (-resetvar)
    dxtrace /dt = - xtrace /taux
    """

spkNeurLGN = Neuron(parameters=params,
                          equations=inpt_eqs,
                          reset="""g_exc=EL 
                                   Spike = 1.0
                                   resetvar=1.0
                                   xtrace+=1/taux""", 
                          spike="""g_exc > VTrest""")

## Neuron Model for V1-Layer, after Clopath et al.(2008) ##
params = """
gL = 30.0
DeltaT = 2.0 
tauw = 144.0 
a = 4.0 
b = 0.0805 
EL = -70.6 
C = 281.0 
tauz = 40.0
tauVT= 50.0
Isp = 400.0
VTMax = -30.4 
VTrest = -50.4
taux = 15.0 
tauLTD = 10.0
tauLTP= 7.0 
taumean = 750.0 
tau_gExc = 1.0
tau_gInh = 10.0 
"""

neuron_eqs = """
noise = Normal(0.0,1.0)
dvm/dt = if state>=2:+3.462 else: if state==1:-(vm+51.75)+1/C*(Isp - (wad+b))+g_Exc-g_Inh else:1/C * ( -gL * (vm - EL) + gL * DeltaT * exp((vm - VT) / DeltaT) - wad + z ) + g_Exc-g_Inh: init = -70.6
dvmean/dt = ((vm - EL)**2 - vmean)/taumean    :init = 0.0
dumeanLTD/dt = (vm - umeanLTD)/tauLTD : init=-70.0
dumeanLTP/dt = (vm - umeanLTP)/tauLTP : init =-70.0
dxtrace /dt = (- xtrace )/taux
dwad/dt = if state ==2:0 else:if state==1:+b/tauw else: (a * (vm - EL) - wad)/tauw : init = 0.0
dz/dt = if state==1:-z+Isp-10 else:-z/tauz  : init = 0.0
dVT/dt =if state==1: +(VTMax - VT)-0.4 else:(VTrest - VT)/tauVT  : init=-50.4
dg_Exc/dt = 1/tau_gExc * (-g_Exc)
dg_Inh/dt = 1/tau_gInh*(-g_Inh)
state = if state > 0: state-1 else:0
Spike = 0.0
dresetvar / dt = 1/(1.0) * (-resetvar)
vmTemp = vm
           """
#
#if state>=2:0 else: if state==1:-5 + (VTrest - VT)/tauVT else:
spkNeurV1 = Neuron(parameters = params,equations=neuron_eqs,spike="""(vm>VT) and (state == 0.0)""",
                         reset="""vm = 29.4
                                  state = 2.0                      
                                  Spike = 1.0
                                  resetvar = 1.0
                                  xtrace+= 1/taux""")

#----------------------------------synapse definitions----------------------

#----- Synapse from Poisson to Input-Layer -----#
inputSynapse =  Synapse(
    parameters = "",
    equations = "",
    pre_spike = """
        g_target += w
                """
)

#-----------------------population defintions-----------------------------------#
popInput = PoissonPopulation(geometry=(patchsize,patchsize,2),rates=50.0)
popLGN = Population(geometry=(patchsize,patchsize,2),neuron=spkNeurLGN )
popV1 = Population(geometry=nbr_V1N, neuron = spkNeurV1)
popIL1 = Population(geometry=nbr_IL1N, neuron = spkNeurV1)
popV2 = Population(geometry=nbr_V2N, neuron = spkNeurV1)
popIL2 = Population(geometry=nbr_IL2N, neuron = spkNeurV1)

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
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

projLGN_IL1 = Projection(
    pre=popLGN, 
    post=popIL1, 
    target='Exc',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

projV1_IL1 = Projection(
    pre = popV1,
    post = popIL1,
    target = 'Exc',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

projIL1_V1 = Projection(
    pre = popIL1,
    post= popV1,
    target = 'Inh',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

projIL1_Lat = Projection(
    pre = popIL1,
    post = popIL1,
    target = 'Inh',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

projV1_V2 = Projection(
    pre = popV1,
    post = popV2,
    target = 'Exc',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))#, delays = Uniform(1,3))

projV2_IL2 = Projection(
    pre = popV2,
    post = popIL2,
    target = 'Exc',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

projV1_IL2 = Projection(
    pre = popV1,
    post = popIL2,
    target = 'Exc',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

projIL2_V2 = Projection(
    pre = popIL2,
    post= popV2,
    target = 'Inh',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

projIL2_Lat = Projection(
    pre = popIL2,
    post = popIL2,
    target = 'Inh',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))
#----------------------------further global functions---------------------------------
def loadWeights():
    projLGN_V1.w = np.loadtxt('network_data/V1weight.txt')
    projLGN_IL1.w =np.loadtxt('network_data/InhibW.txt')
    projV1_IL1.w = np.loadtxt('network_data/V1toIN.txt')
    projIL1_V1.w = np.loadtxt('network_data/INtoV1.txt')
    projIL1_Lat.w =np.loadtxt('network_data/INLat.txt')
    projV1_V2.w  = np.loadtxt('network_data/V2weight.txt')
    projV2_IL2.w = np.loadtxt('network_data/V2toIN2.txt')
    projV1_IL2.w = np.loadtxt('network_data/V1toIN2.txt')
    projIL2_V2.w = np.loadtxt('network_data/IN2toV2.txt')
    projIL2_Lat.w= np.loadtxt('network_data/IN2Lat.txt')

    #projV1_V2.delay = np.loadtxt('network_data/V1toV2_Delay.txt')
    #print(projLGN_V1.w)

