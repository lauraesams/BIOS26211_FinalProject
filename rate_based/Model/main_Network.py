# -*- coding: utf-8 -*-
"""
Created on Wed Nov 07 2018

@author: temi
"""
#ANNarchy
from ANNarchy import *
num_threads=8
setup(structural_plasticity=True, verbose=False, num_threads=num_threads)

#Model scripts
from PatchPattern import *
from SaccadeGenerator import *

#Other packages
import scipy.io
import sys
import os
import time
import numpy as np
from numpy import random

###---Alpha initialization---###
alpha_discount=1.0
alpha_discount_fb=0.6 #Initializes alpha lower

###---Structural Plasticity---###
#Enable structural plasticity
sp_use=1 #0-off, 1-on
#Standard time interval for which the probabilities are valid
sp_baseinterval=10000
#Structural plasticity interval
sp_interval=5 * sp_baseinterval
#Structural plasticity probabilities scaling factor
sp_factor=1

##FF
#Neighborhood range
sp_distance_FF = [3,3,np.inf]#has to be a list
# The weight value that will be deleted with probability sp_pdelmax/2
sp_whalf_FF = .01
#Maximum delete probability per base interval
sp_pdelmax_FF = 0.05 * sp_factor
#Slope of the sigmoidal function for the weight deletion probability
sp_deltemp_FF = sp_whalf_FF / np.log(1 / 0.1 - 1)
#Factor to got appropriate build probabilities
sp_pbuild_FF = 0.1 * sp_factor

##LAT
#Neighborhood range
sp_distance_LAT = [3,3,np.inf]#has to be a list
# The weight value that will be deleted with probability sp_pdelmax/2
sp_whalf_LAT = .15
#Maximum delete probability per base interval
sp_pdelmax_LAT = 0.05 * sp_factor
#Slope of the sigmoidal function for the weight deletion probability
sp_deltemp_LAT = sp_whalf_LAT / np.log(1 / 0.1 - 1)
#Factor to got appropriate build probabilities
sp_pbuild_LAT = 0.1 * sp_factor

##FB
#Neighborhood range
sp_distance_FB = [3,3,np.inf]#has to be a list
# The weight value that will be deleted with probability sp_pdelmax/2
sp_whalf_FB = .01
#Maximum delete probability per base interval
sp_pdelmax_FB = 0.05 * sp_factor
#Slope of the sigmoidal function for the weight deletion probability
sp_deltemp_FB = sp_whalf_FB / np.log(1 / 0.1 - 1)
#Factor to got appropriate build probabilities
sp_pbuild_FB = 0.1 * sp_factor




###---Defining the neurons---###
LGNNeuron = Neuron(
    parameters = """
        tau = 10.0 : population
        input = 0.0
        avg_tau = 50000 : population
    """,
    equations="""
        tau * dr/dt + r = input
        avg_tau * davg_r/dt + avg_r = r : init = 0.09
    """ 
)

CalciumNeuron = Neuron( 
	parameters = """
		tau = 10.0 : population
		ca_tau = 10.0 : population
		ip_on = 1 : population
		tau_ip = 10000 : population
		drift_th=1/100 : population
		drift_a=1/100 : population
		avg_tau = 50000 : population
		FF_on = 1 : population
		FB_on = 1 : population
		LAT_on = 1 : population
	""",
	equations = """
    	I_exc = FF_on*sum(FF) + FB_on*sum(FB)
    	I_ff = FF_on*sum(FF)
    	I_fb = FB_on*sum(FB)
    	I_inh = LAT_on*sum(LAT_NEG)
		tau * dmp/dt + mp = mp_a*(FF_on*sum(FF) + FB_on*sum(FB) - LAT_on*sum(LAT_NEG) - mp_th) : init = 0.0
		#r = if mp > 1.0 : (0.5 + 1 / (1 + exp(-3.5 * (mp - 1)))) else: pos(mp) : init = 0.0
		r = pos(mp) : init = 0.0, max = 3.0
		ca_tau * dca/dt + ca = r : init = 0.0
		ip_drift_th = if mp_th>0 : drift_th else : -drift_th : init =0.0
		mp_th = if ip_on==1 : mp_th + (r-mean(r) - ip_drift_th)*dt/tau_ip else : mp_th : init = 0.0
		sqr_r = r^2 : init = 0.0
		ip_drift_a = if (mp_a)>1 : drift_a else : -drift_a : init =0.0
		mp_a = if ip_on==1 : mp_a + (mean(sqr_r) - r^2 - ip_drift_a)*dt/tau_ip else : mp_a : init = 1.0
		avg_tau * davg_r/dt + avg_r = r : init = 0.05
		avg_tau * davg_mp/dt + avg_mp = mp : init = 0.0
		avg_tau * davg_ca/dt + avg_ca = ca : init = 0.05
		avg_tau * davg_I_ff/dt + avg_I_ff = I_ff : init = 0.0
		avg_tau * davg_I_fb/dt + avg_I_fb = I_fb : init = 0.0
		avg_tau * davg_I_inh/dt + avg_I_inh = I_inh : init = 0.0
	"""
)

CalciumNeuronNoSat = Neuron( 
	parameters = """
		tau = 10.0 : population
		ca_tau = 10.0 : population
		ip_on = 1 : population
		tau_ip = 10000 : population
		drift_th=1/100 : population
		drift_a=1/100 : population
		avg_tau = 50000 : population
		FF_on = 1 : population
		FB_on = 1 : population
		LAT_on = 1 : population
	""",
	equations = """
    	I_exc = FF_on*sum(FF) + FB_on*sum(FB)
    	I_ff = FF_on*sum(FF)
    	I_fb = FB_on*sum(FB)
    	I_inh = LAT_on*sum(LAT_NEG)
		tau * dmp/dt + mp = mp_a*(FF_on*sum(FF) + FB_on*sum(FB) - LAT_on*sum(LAT_NEG) - mp_th) : init = 0.0
		#r = if mp > 1.0 : (0.5 + 1 / (1 + exp(-3.5 * (mp - 1)))) else: pos(mp) : init = 0.0
		r = pos(mp) : init = 0.0
		ca_tau * dca/dt + ca = r : init = 0.0
		ip_drift_th = if mp_th>0 : drift_th else : -drift_th : init =0.0
		mp_th = if ip_on==1 : mp_th + (r-mean(r) - ip_drift_th)*dt/tau_ip else : mp_th : init = 0.0
		sqr_r = r^2 : init = 0.0
		ip_drift_a = if (mp_a)>1 : drift_a else : -drift_a : init =0.0
		mp_a = if ip_on==1 : mp_a + (mean(sqr_r) - r^2 - ip_drift_a)*dt/tau_ip else : mp_a : init = 1.0
		avg_tau * davg_r/dt + avg_r = r : init = 0.05
		avg_tau * davg_mp/dt + avg_mp = mp : init = 0.0
		avg_tau * davg_ca/dt + avg_ca = ca : init = 0.05
		avg_tau * davg_I_ff/dt + avg_I_ff = I_ff : init = 0.0
		avg_tau * davg_I_fb/dt + avg_I_fb = I_fb : init = 0.0
		avg_tau * davg_I_inh/dt + avg_I_inh = I_inh : init = 0.0
	"""
)

###---Defining the learning rules---###
LearnCalciumFF = Synapse(
	parameters ="""
		Gamma = 0.6 : projection
		Gamma_current = 1.3 : projection
		K = 0.005 : projection
		TauH = 100 : projection
		alpha_minus = 0.015 : projection
		TauAlpha = 50000 : projection
		LearnA = 5000 : projection
		LearnB = 30000 : projection
		LearnC = 15 : projection
	""",
	equations = """	
		TauH * dH/dt + H = 0.5*pos(post.r - Gamma)*post.I_ff + 0.1*pos(post.I_ff - Gamma_current) - K : min=0.0, max=3.0, init = 0.0, postsynaptic
		TauAlpha * dalpha/dt = (0.01 + alpha) * (H + 10*alpha_minus*post.mp_th - alpha_minus) : min=0.0, init = 1.0, postsynaptic
		LearnTau = LearnA + LearnB * exp(-LearnC * pos(post.ca)) : postsynaptic
		LearnTau * dw/dt = (pre.r - mean(pre.r)) * post.ca - alpha * post.r^2 * w : min = 0.0, init = 0.0
	"""
)

LearnCalciumFB = Synapse(
	parameters ="""
		Gamma = 0.6 : projection
		Gamma_current = 0.0 : projection
		K = 0.005 : projection
		TauH = 100 : projection
		alpha_minus = 0.015 : projection
		TauAlpha = 50000 : projection
		LearnA = 5000 : projection
		LearnB = 30000 : projection
		LearnC = 15 : projection
	""",
	equations = """	
		TauH * dH/dt + H = 0.5*pos(post.r - Gamma)*post.I_fb + 0.1*pos(post.I_fb - Gamma_current) - K : min=0.0, max=3.0, init = 0.0, postsynaptic
		TauAlpha * dalpha/dt = (0.01 + alpha) * (H + 10*alpha_minus*post.mp_th - alpha_minus) : min=0.0, init = 1.0, postsynaptic
		LearnTau = LearnA + LearnB * exp(-LearnC * pos(post.ca)) : postsynaptic
		LearnTau * dw/dt = (pre.r - mean(pre.r)) * post.ca - alpha * post.r^2 * w : min = 0.0, init = 0.0
	"""
)

Antihebb = Synapse(
	parameters ="""
		TauAH=5000 : projection
        alpha = 1 : projection
        gamma = 1 : projection
		th_singleside=0.0 : projection
	""",
	equations = """	
    	TauAH * dw/dt = pre.r * pos(post.r - th_singleside) - pre.r * pos(post.avg_r-post.mp_th) *(gamma + alpha * w) : min = 0.0, init = 0.0
	"""
)


###---Creating the populations---###
#Reference amount of neurons in the layers, taken from Potjans2014
ref_N_L23_Exc=20683.
ref_N_L23_Inh=5834.
ref_N_L4_Exc=21915.
ref_N_L4_Inh=5479.
#ref_N_L5_Exc=4850.
#ref_N_L5_Inh=1065.
#ref_N_L6_Exc=14395.
#ref_N_L6_Inh=2948.

#Reference amount of neurons in the areas, rough estimation inspired by DiCarlo2007
ref_N_V1=190.
ref_N_V2=150.


#Initial receptive field sizes in the feedforward path, should hold the amount of pre synapses in a roughly similar range
RFsize_FF_V1_L4=10
RFsize_FF_V1_L23=7
RFsize_FF_V2_L4=5
RFsize_FF_V2_L23=4
#RFsize_FF_V4_L4=9
#RFsize_FF_V4_L23=9

#Initial receptive field size of the lateral connections
RFsize_Lat_V1_L4=int(np.floor((RFsize_FF_V1_L4+1.)/2.)*2.+1)
Offset_Lat_V1_L4=int(np.floor(RFsize_Lat_V1_L4/2.))
RFsize_Lat_V1_L23=int(np.floor((RFsize_FF_V1_L23+1.)/2.)*2.+1)
Offset_Lat_V1_L23=int(np.floor(RFsize_Lat_V1_L23/2.))

RFsize_Lat_V2_L4=int(np.floor((RFsize_FF_V2_L4+1.)/2.)*2.+1)
Offset_Lat_V2_L4=int(np.floor(RFsize_Lat_V2_L4/2.))
RFsize_Lat_V2_L23=int(np.floor((RFsize_FF_V2_L23+1.)/2.)*2.+1)
Offset_Lat_V2_L23=int(np.floor(RFsize_Lat_V2_L23/2.))

#Initial receptive field size of the feedback connection
RFsize_FB_V1_L4=RFsize_FF_V1_L23
Offset_FB_V1_L4=RFsize_FB_V1_L4-1
RFsize_FB_V1_L23=RFsize_FF_V2_L4+RFsize_FF_V2_L23-1
Offset_FB_V1_L23=RFsize_FB_V1_L23-1
RFsize_FB_V2_L4=RFsize_FF_V2_L23
Offset_FB_V2_L4=RFsize_FB_V2_L4-1

# How to determine layer sizes: (inlayersize - RFsize) / RFshift + 1 = outlayersize | e.g. inlayersize=24 RFsize=12 RFshift=1 | V1_L4_Exc: (24-12)/1+1=13 | V1_L23_Exc: (13-7)/1+1=7 
# Now with symetric offset: (inlayersize - RFsize - 2 * offset) / RFshift + 1 = outlayersize | V1_L4_Inh: (13-11+2*5)/1+1=13
# The offset can be chosen as following: -(RFsize-1)/2 | V1_L4_Inh: -(11-1)/2=-5 | V1_L23_Inh: -(5-1)/2=-2

LGN = Population(name = "LGN", geometry=(32, 32, 2), neuron=LGNNeuron)
V1_L4_Exc = Population(name = "V1_L4_Exc",geometry=(LGN.width-RFsize_FF_V1_L4+1, LGN.height-RFsize_FF_V1_L4+1, 4), neuron=CalciumNeuron)#24-10: 15x15x4=900 12:13x13x4=676 | 48-10:39x39x4=6084 | 32-10:23x23x4=2116
V1_L4_Inh = Population(name = "V1_L4_Inh", geometry=(LGN.width-RFsize_FF_V1_L4+1, LGN.height-RFsize_FF_V1_L4+1, np.round(ref_N_L4_Inh/ref_N_L4_Exc*V1_L4_Exc.depth)), neuron=CalciumNeuronNoSat)
V1_L23_Exc = Population(name = "V1_L23_Exc", geometry=(V1_L4_Exc.width-RFsize_FF_V1_L23+1, V1_L4_Exc.height-RFsize_FF_V1_L23+1,np.round(V1_L4_Exc.size * ref_N_L23_Exc/ref_N_L4_Exc / ((V1_L4_Exc.width-RFsize_FF_V1_L23+1)*(V1_L4_Exc.height-RFsize_FF_V1_L23+1)))), neuron = CalciumNeuron)#10-15:9x9x11=891 12-13:7x7x13=637 32-23:17x17x7=2023
V1_L23_Inh = Population(name = "V1_L23_Inh", geometry=(V1_L4_Exc.width-RFsize_FF_V1_L23+1, V1_L4_Exc.height-RFsize_FF_V1_L23+1,np.round(ref_N_L23_Inh/ref_N_L23_Exc*V1_L23_Exc.depth)), neuron = CalciumNeuronNoSat)

V2_L4_Exc = Population(name = "V2_L4_Exc", geometry=(V1_L23_Exc.width-RFsize_FF_V2_L4+1, V1_L23_Exc.height-RFsize_FF_V2_L4+1,np.round(V1_L4_Exc.size * ref_N_V2/ref_N_V1 / ((V1_L23_Exc.width-RFsize_FF_V2_L4+1)*(V1_L23_Exc.height-RFsize_FF_V2_L4+1)))), neuron = CalciumNeuron)#10-15/9:5x5x28=700 12-13/7:3x3x59=531 32-23/17:13x13x10=1690
V2_L4_Inh = Population(name = "V2_L4_Inh", geometry=(V1_L23_Exc.width-RFsize_FF_V2_L4+1, V1_L23_Exc.height-RFsize_FF_V2_L4+1,np.round(ref_N_L4_Inh/ref_N_L4_Exc*V2_L4_Exc.depth)), neuron = CalciumNeuronNoSat)
V2_L23_Exc = Population(name= "V2_L23_Exc", geometry=(V2_L4_Exc.width-RFsize_FF_V2_L23+1, V2_L4_Exc.height-RFsize_FF_V2_L23+1,np.round(V2_L4_Exc.size * ref_N_L23_Exc/ref_N_L4_Exc / ((V2_L4_Exc.width-RFsize_FF_V2_L23+1)*(V2_L4_Exc.height-RFsize_FF_V2_L23+1)))), neuron = CalciumNeuron)#10-15/9/5:3x3x73=657 12-13/7/3:1x1x501 32-23/17/13:11x11x13=1573 32-23/17/13:10x10x16=1600
V2_L23_Inh = Population(name= "V2_L23_Inh", geometry=(V2_L4_Exc.width-RFsize_FF_V2_L23+1, V2_L4_Exc.height-RFsize_FF_V2_L23+1,np.round(ref_N_L23_Inh/ref_N_L23_Exc*V2_L23_Exc.depth)), neuron = CalciumNeuronNoSat)

#Set calcium trace length for the layers 2/3
V1_L23_Exc.ca_tau = 500               
V2_L23_Exc.ca_tau = 500


#Expectation values layers E_r_...
E_r_LGN=0.1057#Expectation value input image, mean(LGN)
E_r_V1_L4=0.1
E_r_V1_L4_Inh=0.1
E_r_V1_L23=0.1
E_r_V1_L23_Inh=0.1
E_r_V2_L4=0.1
E_r_V2_L4_Inh=0.1
E_r_V2_L23=0.1
E_r_V2_L23_Inh=0.1


###---Creating the connections---###
##--Feedforward pathway--##
# LGN to V1 connections excitatory
#V1_L4_Exc
#Calculate initial weights based on the expectation value of the input
E_w = 1.0 / (RFsize_FF_V1_L4 * RFsize_FF_V1_L4 * LGN.depth * E_r_LGN)
lowerbound = 0 * E_w
upperbound = 2 * E_w
FF_V1_L4_Exc = Projection(
        pre = LGN, 
        post = V1_L4_Exc, 
        target = 'FF', 
        synapse = LearnCalciumFF
)
FF_V1_L4_Exc.connect_with_func( 
    method = patch_pattern, 
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_FF_V1_L4,
    rfY = RFsize_FF_V1_L4,
    offsetX = 0,
    offsetY = 0,
    delay=2
)
FF_V1_L4_Exc.alpha=1/((RFsize_FF_V1_L4 * RFsize_FF_V1_L4 * LGN.depth)**0.5 * E_w) * alpha_discount

#V1_L4_Inh
E_w = 0.334 / (RFsize_FF_V1_L4 * RFsize_FF_V1_L4 * LGN.depth * E_r_LGN)
lowerbound = 0 * E_w
upperbound = 2 * E_w
FF_V1_L4_Inh = Projection(
        pre = LGN, 
        post = V1_L4_Inh, 
        target = 'FF', 
        synapse = LearnCalciumFF
)
FF_V1_L4_Inh.connect_with_func( 
    method = patch_pattern, 
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_FF_V1_L4,
    rfY = RFsize_FF_V1_L4,
    offsetX = 0,
    offsetY = 0
)
FF_V1_L4_Inh.alpha=1/((RFsize_FF_V1_L4 * RFsize_FF_V1_L4 * LGN.depth)**0.5 * E_w) * alpha_discount

#V1_L4 to V1_L23 connections excitatory
#V1_L23_Exc
E_w = 0.5 / (RFsize_FF_V1_L23*RFsize_FF_V1_L23*V1_L4_Exc.depth * E_r_V1_L4)      
lowerbound = 0 * E_w
upperbound = 2 * E_w
FF_V1_L23_Exc = Projection(
    pre = V1_L4_Exc,
    post = V1_L23_Exc,
    target = 'FF',
    synapse = LearnCalciumFF
)
FF_V1_L23_Exc.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_FF_V1_L23,
    rfY = RFsize_FF_V1_L23,
    offsetX = 0,
	offsetY = 0,
	delay=2
)
FF_V1_L23_Exc.alpha=1/((RFsize_FF_V1_L23*RFsize_FF_V1_L23*V1_L4_Exc.depth)**0.5 * E_w) * alpha_discount
FF_V1_L23_Exc.Gamma_current=0.8

#V1_L23_Inh
E_w = 0.334 / (RFsize_FF_V1_L23*RFsize_FF_V1_L23*V1_L4_Exc.depth * E_r_V1_L4)      
lowerbound = 0 * E_w
upperbound = 2 * E_w
FF_V1_L23_Inh = Projection(
    pre = V1_L4_Exc,
    post = V1_L23_Inh,
    target = 'FF',
    synapse = LearnCalciumFF
)
FF_V1_L23_Inh.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_FF_V1_L23,
    rfY = RFsize_FF_V1_L23,
    offsetX = 0,
	offsetY = 0
)
FF_V1_L23_Inh.alpha=1/((RFsize_FF_V1_L23*RFsize_FF_V1_L23*V1_L4_Exc.depth)**0.5 * E_w) * alpha_discount
FF_V1_L23_Inh.Gamma_current=0.8

#V1_L23_to V2_L4 connections excitatory
#V2_L4_Exc
E_w = 1.0 / (RFsize_FF_V2_L4*RFsize_FF_V2_L4*V1_L23_Exc.depth * E_r_V1_L23)
lowerbound = 0 * E_w
upperbound = 2 * E_w
FF_V2_L4_Exc = Projection(
    pre = V1_L23_Exc,
    post = V2_L4_Exc,
    target = 'FF',
    synapse = LearnCalciumFF
)
FF_V2_L4_Exc.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_FF_V2_L4,                 
    rfY = RFsize_FF_V2_L4,
    offsetX = 0,            
	offsetY = 0,
	delay=3
)
FF_V2_L4_Exc.alpha=1/((RFsize_FF_V2_L4*RFsize_FF_V2_L4*V1_L23_Exc.depth)**0.5 * E_w) * alpha_discount

#V2_L4_Inh
E_w = 0.334 / (RFsize_FF_V2_L4*RFsize_FF_V2_L4*V1_L23_Exc.depth * E_r_V1_L23)                  
lowerbound = 0 * E_w
upperbound = 2 * E_w
FF_V2_L4_Inh = Projection(
    pre = V1_L23_Exc,
    post = V2_L4_Inh,
    target = 'FF',
    synapse = LearnCalciumFF
)
FF_V2_L4_Inh.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_FF_V2_L4,                 
    rfY = RFsize_FF_V2_L4,
    offsetX = 0,            
	offsetY = 0,
	delay=2
)
FF_V2_L4_Inh.alpha=1/((RFsize_FF_V2_L4*RFsize_FF_V2_L4*V1_L23_Exc.depth)**0.5 * E_w) * alpha_discount

#V2_L4 to V2_L23 connections exitatory
#V2_L23_Exc
E_w = 1.0 / (RFsize_FF_V2_L23*RFsize_FF_V2_L23*V2_L4_Exc.depth * E_r_V2_L4)                  
lowerbound = 0 * E_w
upperbound = 2 * E_w
FF_V2_L23_Exc = Projection(
    pre = V2_L4_Exc,
    post = V2_L23_Exc,
    target = 'FF',
    synapse = LearnCalciumFF
)
FF_V2_L23_Exc.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_FF_V2_L23,                 
    rfY = RFsize_FF_V2_L23,
    offsetX = 0,                  
	offsetY = 0,
	delay=2
)
FF_V2_L23_Exc.alpha=1/((RFsize_FF_V2_L23*RFsize_FF_V2_L23*V2_L4_Exc.depth)**0.5 * E_w) * alpha_discount

#V2_L23_Inh
E_w = 0.5 / (RFsize_FF_V2_L23*RFsize_FF_V2_L23*V2_L4_Exc.depth * E_r_V2_L4)                 
lowerbound = 0 * E_w
upperbound = 2 * E_w
FF_V2_L23_Inh = Projection(
    pre = V2_L4_Exc,
    post = V2_L23_Inh,
    target = 'FF',
    synapse = LearnCalciumFF
)
FF_V2_L23_Inh.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_FF_V2_L23,                 
    rfY = RFsize_FF_V2_L23,
    offsetX = 0,                  
	offsetY = 0 
)
FF_V2_L23_Inh.alpha=1/((RFsize_FF_V2_L23*RFsize_FF_V2_L23*V2_L4_Exc.depth)**0.5 * E_w) * alpha_discount


##--Lateral pathway--##
# V1_L4 connections
#Lateral excitatory
E_w = 0.334 / (RFsize_Lat_V1_L4*RFsize_Lat_V1_L4*V1_L4_Exc.depth * E_r_V1_L4)
lowerbound = 0 * E_w
upperbound = 2 * E_w
Lat_V1_L4_Inh = Projection(
	pre = V1_L4_Exc,
	post = V1_L4_Inh,
	target = 'FF',
	synapse = LearnCalciumFF
)
Lat_V1_L4_Inh.connect_with_func( 
	method = patch_pattern, 
	min_value = lowerbound,
	max_value = upperbound,
	shiftRF = 1,
	rfX = RFsize_Lat_V1_L4,
	rfY = RFsize_Lat_V1_L4,
	offsetX = -Offset_Lat_V1_L4,
	offsetY = -Offset_Lat_V1_L4
)
Lat_V1_L4_Inh.alpha=1/((RFsize_Lat_V1_L4*RFsize_Lat_V1_L4*V1_L4_Exc.depth)**0.5 * E_w) * alpha_discount

#Lateral inhibitory
E_w = 1. / (RFsize_Lat_V1_L4*RFsize_Lat_V1_L4*V1_L4_Inh.depth * E_r_V1_L4_Inh)
lowerbound = 0 * E_w
upperbound = 2 * E_w
Lat_V1_L4_Exc = Projection(
	pre = V1_L4_Inh, 
	post = V1_L4_Exc,
	target = 'LAT_NEG',
	synapse = Antihebb
)
Lat_V1_L4_Exc.connect_with_func( 
	method = patch_pattern, 
	min_value = lowerbound,
	max_value = upperbound,
	shiftRF = 1,
	rfX = RFsize_Lat_V1_L4,
	rfY = RFsize_Lat_V1_L4,
	offsetX = -Offset_Lat_V1_L4,
	offsetY = -Offset_Lat_V1_L4
)

#Lateral inhibitory, selfinhibition
E_w = 1. / (RFsize_Lat_V1_L4*RFsize_Lat_V1_L4*V1_L4_Inh.depth * E_r_V1_L4_Inh)
lowerbound = 0 * E_w
upperbound = 2 * E_w
Self_V1_L4_Inh = Projection(
	pre = V1_L4_Inh, 
	post = V1_L4_Inh,
	target = 'LAT_NEG',
	synapse = Antihebb
)
Self_V1_L4_Inh.connect_with_func( 
	method = patch_pattern, 
	min_value = lowerbound,
	max_value = upperbound,
	shiftRF = 1,
	rfX = RFsize_Lat_V1_L4,
	rfY = RFsize_Lat_V1_L4,
	offsetX = -Offset_Lat_V1_L4,
	offsetY = -Offset_Lat_V1_L4
)

# V1_L23 connections
#Lateral excitatory
E_w = 0.334 / (RFsize_Lat_V1_L23*RFsize_Lat_V1_L23*V1_L23_Exc.depth * E_r_V1_L23)                                 
lowerbound = 0 * E_w
upperbound = 2 * E_w
Lat_V1_L23_Inh = Projection(
    pre = V1_L23_Exc,
    post = V1_L23_Inh,
    target = 'FF',
    synapse = LearnCalciumFF
)
Lat_V1_L23_Inh.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_Lat_V1_L23,                 
    rfY = RFsize_Lat_V1_L23,
    offsetX = -Offset_Lat_V1_L23,          
	offsetY = -Offset_Lat_V1_L23
)
Lat_V1_L23_Inh.alpha=1/((RFsize_Lat_V1_L23*RFsize_Lat_V1_L23*V1_L23_Exc.depth)**0.5 * E_w) * alpha_discount

#Lateral inhibitory
E_w = 0.5 / (RFsize_Lat_V1_L23*RFsize_Lat_V1_L23*V1_L23_Inh.depth * E_r_V1_L23_Inh)                                 
lowerbound = 0 * E_w
upperbound = 2 * E_w
Lat_V1_L23_Exc = Projection(
    pre = V1_L23_Inh,
    post = V1_L23_Exc,
    target = 'LAT_NEG',
    synapse = Antihebb
)
Lat_V1_L23_Exc.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_Lat_V1_L23,
    rfY = RFsize_Lat_V1_L23,
    offsetX = -Offset_Lat_V1_L23,
    offsetY = -Offset_Lat_V1_L23
)

#Lateral inhibitory, selfinhibition
E_w = 0.5 / (RFsize_Lat_V1_L23*RFsize_Lat_V1_L23*V1_L23_Inh.depth * E_r_V1_L23_Inh)                                 
lowerbound = 0 * E_w
upperbound = 2 * E_w
Self_V1_L23_Inh = Projection(
    pre = V1_L23_Inh,
    post = V1_L23_Inh,
    target = 'LAT_NEG',
    synapse = Antihebb
)
Self_V1_L23_Inh.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_Lat_V1_L23,
    rfY = RFsize_Lat_V1_L23,
    offsetX = -Offset_Lat_V1_L23,
    offsetY = -Offset_Lat_V1_L23
)

# V2_L4 connections
#Lateral excitatory
E_w = 0.334 / (RFsize_Lat_V2_L4*RFsize_Lat_V2_L4*V2_L4_Exc.depth * E_r_V2_L4)                  
lowerbound = 0 * E_w
upperbound = 2 * E_w
Lat_V2_L4_Inh = Projection(
    pre = V2_L4_Exc,
    post = V2_L4_Inh,
    target = 'FF',
    synapse = LearnCalciumFF
)
Lat_V2_L4_Inh.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_Lat_V2_L4,                 
    rfY = RFsize_Lat_V2_L4,
    offsetX = -Offset_Lat_V2_L4,                  
	offsetY = -Offset_Lat_V2_L4
)
Lat_V2_L4_Inh.alpha=1/((RFsize_Lat_V2_L4*RFsize_Lat_V2_L4*V2_L4_Exc.depth)**0.5 * E_w) * alpha_discount

#Lateral inhibitory
E_w = 1 / (RFsize_Lat_V2_L4*RFsize_Lat_V2_L4*V2_L4_Inh.depth * E_r_V2_L4_Inh)                  
lowerbound = 0 * E_w
upperbound = 2 * E_w
Lat_V2_L4_Exc = Projection(
    pre = V2_L4_Inh,
    post = V2_L4_Exc,
    target = 'LAT_NEG',
    synapse = Antihebb
)
Lat_V2_L4_Exc.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_Lat_V2_L4,
    rfY = RFsize_Lat_V2_L4,
    offsetX = -Offset_Lat_V2_L4,
    offsetY = -Offset_Lat_V2_L4
)

#Lateral inhibitory, selfinhibition
E_w = 1 / (RFsize_Lat_V2_L4*RFsize_Lat_V2_L4*V2_L4_Inh.depth * E_r_V2_L4_Inh)                  
lowerbound = 0 * E_w
upperbound = 2 * E_w
Self_V2_L4_Inh = Projection(
    pre = V2_L4_Inh,
    post = V2_L4_Inh,
    target = 'LAT_NEG',
    synapse = Antihebb
)
Self_V2_L4_Inh.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_Lat_V2_L4,
    rfY = RFsize_Lat_V2_L4,
    offsetX = -Offset_Lat_V2_L4,
    offsetY = -Offset_Lat_V2_L4
)

# V2_L23 connections
#Lateral excitatory
E_w = 0.5 / (RFsize_Lat_V2_L23*RFsize_Lat_V2_L23*V2_L23_Exc.depth * E_r_V2_L23)                  
lowerbound = 0 * E_w
upperbound = 2 * E_w
Lat_V2_L23_Inh = Projection(
    pre = V2_L23_Exc,
    post = V2_L23_Inh,
    target = 'FF',
    synapse = LearnCalciumFF
)
Lat_V2_L23_Inh.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_Lat_V2_L23,                 
    rfY = RFsize_Lat_V2_L23,
    offsetX = -Offset_Lat_V2_L23,                  
	offsetY = -Offset_Lat_V2_L23
)
Lat_V2_L23_Inh.alpha=1/((RFsize_Lat_V2_L23*RFsize_Lat_V2_L23*V2_L23_Exc.depth)**0.5 * E_w) * alpha_discount

#Lateral inhibitory
E_w = 0.5 / (RFsize_Lat_V2_L23*RFsize_Lat_V2_L23*V2_L23_Inh.depth * E_r_V2_L23_Inh)                  
lowerbound = 0 * E_w
upperbound = 2 * E_w
Lat_V2_L23_Exc = Projection(
    pre = V2_L23_Inh,
    post = V2_L23_Exc,
    target = 'LAT_NEG',
    synapse = Antihebb
)
Lat_V2_L23_Exc.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_Lat_V2_L23,
    rfY = RFsize_Lat_V2_L23,
    offsetX = -Offset_Lat_V2_L23,
    offsetY = -Offset_Lat_V2_L23
)

#Lateral inhibitory, selfinhibition
E_w = 0.5 / (RFsize_Lat_V2_L23*RFsize_Lat_V2_L23*V2_L23_Inh.depth * E_r_V2_L23_Inh)                  
lowerbound = 0 * E_w
upperbound = 2 * E_w
Self_V2_L23_Inh = Projection(
    pre = V2_L23_Inh,
    post = V2_L23_Inh,
    target = 'LAT_NEG',
    synapse = Antihebb
)
Self_V2_L23_Inh.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_Lat_V2_L23,
    rfY = RFsize_Lat_V2_L23,
    offsetX = -Offset_Lat_V2_L23,
    offsetY = -Offset_Lat_V2_L23
)


##--Feedback pathway--##
#Interarea feedback
#V2_L23  to V1_l23 connections exhibitory
E_w = 0.5 / (RFsize_FB_V1_L23*RFsize_FB_V1_L23*V2_L23_Exc.depth * E_r_V2_L23)                  
lowerbound = 0 * E_w
upperbound = 2 * E_w
exc = Projection(
    pre = V2_L23_Exc,
    post = V1_L23_Exc,
    target = 'FB',
    synapse = LearnCalciumFB
)
exc.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_FB_V1_L23,                 
    rfY = RFsize_FB_V1_L23,
    offsetX = -Offset_FB_V1_L23,                  
	offsetY = -Offset_FB_V1_L23,
	delay=3
)
exc.alpha=1/((RFsize_FB_V1_L23*RFsize_FB_V1_L23*V2_L23_Exc.depth)**0.5 * E_w) * alpha_discount_fb

E_w = 0.5 / (RFsize_FB_V1_L23*RFsize_FB_V1_L23*V2_L23_Exc.depth * E_r_V2_L23)                  
lowerbound = 0 * E_w
upperbound = 2 * E_w
exc = Projection(
    pre = V2_L23_Exc,
    post = V1_L23_Inh,
    target = 'FB',
    synapse = LearnCalciumFB
)
exc.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_FB_V1_L23,                 
    rfY = RFsize_FB_V1_L23,
    offsetX = -Offset_FB_V1_L23,                  
	offsetY = -Offset_FB_V1_L23,
	delay=2
)
exc.alpha=1/((RFsize_FB_V1_L23*RFsize_FB_V1_L23*V2_L23_Exc.depth)**0.5 * E_w) * alpha_discount_fb

#Feedback to inhibitory neurons
#V1-L4
#V1_l23 to V1_l4 connections excitatory
E_w = 0.334 / (RFsize_FB_V1_L4*RFsize_FB_V1_L4*V1_L23_Exc.depth * E_r_V1_L23)               
lowerbound = 0 * E_w
upperbound = 2 * E_w
exc = Projection(
    pre = V1_L23_Exc,
    post = V1_L4_Inh,
    target = 'FB',
    synapse = LearnCalciumFB
)
exc.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_FB_V1_L4,
    rfY = RFsize_FB_V1_L4,
    offsetX = -Offset_FB_V1_L4,     
	offsetY = -Offset_FB_V1_L4
)
exc.alpha=1/((RFsize_FB_V1_L4*RFsize_FB_V1_L4*V1_L23_Exc.depth)**0.5 * E_w) * alpha_discount_fb

#V2_L23 to V2_L4 connections exitatory
E_w = 0.334 / (RFsize_FB_V2_L4*RFsize_FB_V2_L4*V2_L23_Exc.depth * E_r_V2_L23)                  
lowerbound = 0 * E_w
upperbound = 2 * E_w
exc = Projection(
    pre = V2_L23_Exc,
    post = V2_L4_Inh,
    target = 'FB',
    synapse = LearnCalciumFB
)
exc.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_FB_V2_L4,                 
    rfY = RFsize_FB_V2_L4,
    offsetX = -Offset_FB_V2_L4,                  
	offsetY = -Offset_FB_V2_L4 
)
exc.alpha=1/((RFsize_FB_V2_L4*RFsize_FB_V2_L4*V2_L23_Exc.depth)**0.5 * E_w) * alpha_discount_fb

##--Feedforward inhibition--##
#V1_L23_Exc
E_w = 0.5 / (RFsize_FF_V1_L23*RFsize_FF_V1_L23*V1_L4_Inh.depth * E_r_V1_L4_Inh)                  
lowerbound = 0 * E_w
upperbound = 2 * E_w
inh = Projection(
    pre = V1_L4_Inh,
    post = V1_L23_Exc,
    target = 'LAT_NEG',
    synapse = Antihebb
)
inh.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_FF_V1_L23,                 
    rfY = RFsize_FF_V1_L23,
    offsetX = 0,
    offsetY = 0
)

#V1_L23_Inh
E_w = 0.5 / (RFsize_FF_V1_L23*RFsize_FF_V1_L23*V1_L4_Inh.depth * E_r_V1_L4_Inh)                  
lowerbound = 0 * E_w
upperbound = 2 * E_w
inh = Projection(
    pre = V1_L4_Inh,
    post = V1_L23_Inh,
    target = 'LAT_NEG',
    synapse = Antihebb
)
inh.connect_with_func(
    method = patch_pattern,     
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,  
    rfX = RFsize_FF_V1_L23,
    rfY = RFsize_FF_V1_L23,
    offsetX = 0,
    offsetY = 0
)

#V2_L23_Exc
E_w = 0.5 / (RFsize_FF_V2_L23*RFsize_FF_V2_L23*V2_L4_Inh.depth * E_r_V2_L4_Inh)                  
lowerbound = 0 * E_w
upperbound = 2 * E_w
inh = Projection(
    pre = V2_L4_Inh,
    post = V2_L23_Exc,
    target = 'LAT_NEG',
    synapse = Antihebb
)
inh.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_FF_V2_L23,
    rfY = RFsize_FF_V2_L23,
    offsetX = 0,
    offsetY = 0
)

#V2_L23_Inh
E_w = 0.5 / (RFsize_FF_V2_L23*RFsize_FF_V2_L23*V2_L4_Inh.depth * E_r_V2_L4_Inh)                  
lowerbound = 0 * E_w
upperbound = 2 * E_w
inh = Projection(
    pre = V2_L4_Inh,
    post = V2_L23_Inh,
    target = 'LAT_NEG',
    synapse = Antihebb
)
inh.connect_with_func(
    method = patch_pattern,
    min_value = lowerbound,
    max_value = upperbound,
    shiftRF = 1,
    rfX = RFsize_FF_V2_L23,
    rfY = RFsize_FF_V2_L23,
    offsetX = 0,
    offsetY = 0
)



###---Structural plasticity---###
def sp_outgrowth_mt(proj, sp_distance=[1], sp_pbuild=0.0015, sp_pdelmax=0.001, sp_whalf=0.01,sp_deltemp=0.01, sp_interval=20000, sp_baseinterval=1000):   
    #Define the distance value for all dimensions
    if len(sp_distance)==1:
        for i in range(len(sp_distance),3):
            sp_distance.append(sp_distance[0])
    
    #When distance is just defined in two dimensions, set it to infinity for the thirth
    if len(sp_distance)<3:
        for i in range(len(sp_distance),3):
            sp_distance.append(np.inf)
            
    #Get geometry of the presynaptic layer
    pre_geometry = list(proj.pre.geometry)
    
    #Ensure 3 dimensions
    if len(pre_geometry) < 3:
        for i in range(len(pre_geometry),3):
            pre_geometry.append(1)
    
    #Get connection delay
    if type(proj.delay)==list:
        conn_delay=max(max(proj.delay))
    elif type(proj.delay)==float:
        conn_delay=proj.delay
    else:
        raise TypeError(str(type(proj.delay))+" is not float or list")
        
    #For all postsynaptic neurons
    for post_rank in proj.post.ranks:
        
        #Get weight vector of the neuron
        lp = proj.dendrite(post_rank)

        #Init variable describing the neighborhood
        s_neighborhood = np.zeros((3,2),dtype=np.int) #proj.pre.dimension
        
        #Init array for the build probabilities
        p_create = np.zeros(pre_geometry, np.double)
        
        #Prohibit self connections
        if proj.pre == proj.post:
            p_create[proj.post.coordinates_from_rank(post_rank)] = -np.inf
        
        #Calculate maximal weight for weight scaling
        max_weight = max(np.max(np.abs(lp.w)),np.finfo(float).eps)
        
        #Scale delete probability relative to the maximal weight of the neuron
        sp_whalf_local = sp_whalf * max_weight
        
        #For all connections
        it_k = 0
        for s_rank in lp.pre_rank:
            
            #First remove synapses via weight dependent random removal
            if lp.size>1:
                try:
                    tmp_dprob = sp_pdelmax / (1 + np.exp((np.abs(lp.w[it_k]) - sp_whalf_local) / sp_deltemp))
                    dprob = 1 - (1 - tmp_dprob) ** (sp_interval / sp_baseinterval)
                except OverflowError:
                    if np.abs(lp.w[it_k]) > sp_whalf_local:
                        dprob = 0
                    else:
                        drpob = sp_pdelmax
    
                if random.random() < dprob:
                    lp.prune_synapse(s_rank)
                    #Do not execute weight accumulation for a removed synapse
                    continue



            #Calculate Synapse building probabilities
            #Get positions
            s_coord = proj.pre.coordinates_from_rank(s_rank)
            
            #Remove from available entries
            p_create[s_coord] = -np.inf
            
            #Determine neighborhood
            neigh_weight = 1  #total size
            for dim in range(0, 3): #pre dimension
                s_neighborhood[dim, 0] = max(0, s_coord[dim] - sp_distance[dim])
                s_neighborhood[dim, 1] = min(pre_geometry[dim], s_coord[dim] + sp_distance[dim] +1)
                neigh_weight *= (s_neighborhood[dim, 1] - s_neighborhood[dim, 0])
            
            #Remove the synapse itself
            neigh_weight -= 1
            
            #Accumulate weights in the neighborhood of existing synapses
            p_create[s_neighborhood[0, 0]:s_neighborhood[0, 1], s_neighborhood[1, 0]:s_neighborhood[1, 1], s_neighborhood[2, 0]:s_neighborhood[2, 1]] \
            += (max(np.abs(lp.w[it_k]),0) / max_weight) / neigh_weight
            it_k = it_k + 1



        ## Add synapses
        # TODO: Is there a more efficient way to touch only synapses with a probability above 0?
        tmp_p_create = p_create[np.where(p_create > 0)]  # values
        tmp_p_create_indices = np.argwhere(p_create > 0)  # indices
        rnd_create = np.random.rand(*tmp_p_create.shape)
        for it_i in range(0, len(tmp_p_create)):
            if (rnd_create[it_i] < (1 - (1 - sp_pbuild * tmp_p_create[it_i]) ** (sp_interval / sp_baseinterval))):
                #Insert new weight with value around w_half * max_w
                lp.create_synapse(proj.pre.rank_from_coordinates((int(tmp_p_create_indices[it_i][0]),
                                                                  int(tmp_p_create_indices[it_i][1]),
                                                                  int(tmp_p_create_indices[it_i][2]))),
                                                                  2*random.random()*sp_whalf_local,
                                                                  conn_delay)
		
		
		
###---Network training---###
def learn_task(step_start=0,steps=20000000,presentation_time=100, rnd_seed=-1):
    #Learning
    print("Start time learn:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    
    ## Creation of the environment
    #Read in image data for world
    #[width, heigth, OnOff, ImageNumber]
    images = scipy.io.loadmat('./DATA/IMAGES_NORM2015.mat')['IMAGES_NORM']
    images_dim = images.shape
    
    #Time measurement
    t1=time.time()
    
    #Set numpy random seed for reproducable saccades
    if rnd_seed==None or rnd_seed>=0:
        np.random.seed(rnd_seed)
    
    #When having stored the state | yet not implemented in the interface
    #if rnd_state !=None:
    #    np.random.set_state(rnd_state)
    #else:
        #Storing the used seed | complex tuple
    #    rnd_state=np.random.get_state()
        
    
    #Get initial saccade position
    img,x,y = saccade_new(images_dim[3], images_dim[0], images_dim[1], LGN.width, LGN.height)
    
    ##Train network
    for i in range(step_start,steps,presentation_time):
        #Get saccade position
        #img,x,y =saccade_update(curr_x, curr_y, curr_img, saccade_nbr, saccades_perimg, imgs_nbr, img_width, img_height, patch_width, patch_height, mu, sigma)
        img,x,y = saccade_update(x, y, img, i/presentation_time, 10, images_dim[3], images_dim[0], images_dim[1], LGN.width, LGN.height, 0, 2)
        
        #Cut out input patch
        img_patch=images[x:x+LGN.width, y:y+LGN.height,: ,img]

        #Save results  
        if i%50000==0:
            tmp_mat='./results/network_'+str(get_current_step())+'.mat'
            save(tmp_mat)
        if i%2000000==0:
            tmp_data='./results/network_'+str(get_current_step())+'.data'
            save(tmp_data)

        #Output
        if (i%500==0 or i==100) and (i<=10000) and (i>0):
            #Counting output length for nicer multiline output
            prev_lines=0
            output='Step:'+str(get_current_step())+' '
            for pop in populations():
                output=output+pop.name+'#Mean:'+"%.3f"%(np.mean(pop.r))+' Max:'+"%.3f"%(np.max(pop.r))
                if len(output)-prev_lines>200:
                    output=output+'\n'
                    prev_lines=len(output)
                else:
                    output=output+'|'
            print(output)

        # Forecast
        if i==100 or i==sp_interval:
            t0=time.time()
            t0_step=i
        if i==1000 or i==2*sp_interval:
            t1=time.time()
            expected=(t1-t0)*steps/(i-t0_step)
            days=int(expected / 86400)
            hours=int(expected % 86400/ 3600)
            minutes=int(expected % 3600/ 60)
            seconds=int(expected % 60)
            print('Time expected: ' + str(days) + ' - ' + str(hours) + ':' + str(minutes) + ':'+ str(seconds) + ' d-h:m:s')
            
        if i%500000==0:
            t2=time.time()
            output='Step:'+str(get_current_step())+' Time consumed:'+str(t2-t1)+'s '
            t1=t2
            #Counting output length for nicer multiline output
            prev_lines=0
            for pop in populations():
                output=output+pop.name+'#Mean:'+"%.3f"%(np.mean(pop.r))+' Max:'+"%.3f"%(np.max(pop.r))
                if len(output)-prev_lines>200:
                    output=output+'\n'
                    prev_lines=len(output)
                else:
                    output=output+'|'
            print(output)




        # Call structural plasticity
        if i>0 and i% sp_interval == 0 and sp_use == 1:
            
            for proj in projections():

                if proj.target == 'FF':
                    sp_outgrowth_mt(proj, sp_distance_FF, sp_pbuild_FF, sp_pdelmax_FF, sp_whalf_FF, sp_deltemp_FF, sp_interval, sp_baseinterval)

                if proj.target == 'LAT_NEG':
                    sp_outgrowth_mt(proj,  sp_distance_LAT, sp_pbuild_LAT, sp_pdelmax_LAT, sp_whalf_LAT, sp_deltemp_LAT, sp_interval, sp_baseinterval)

                if proj.target == 'FB':
                    sp_outgrowth_mt(proj, sp_distance_FB, sp_pbuild_FB, sp_pdelmax_FB, sp_whalf_FB, sp_deltemp_FB, sp_interval, sp_baseinterval)

                            

        #Set input patch
        LGN.input=np.array(img_patch)
        
        #Execute network | non blocking
        simulate(presentation_time)
    
    #Reset numpy random seed
    #numpy.random.seed(seed=None)
    #When having stored the state
    #numpy.random.set_state(state)
    
    #Save results    
    tmp_data='./results/network_'+str(get_current_step())+'.data'
    tmp_mat='./results/network_'+str(get_current_step())+'.mat'
    save(tmp_data)
    save(tmp_mat)


def set_FF_on(value):
    for pop in populations():
        if hasattr(pop,'FF_on')==True:
            pop.FF_on = value
            print('Set FF_on =', value, 'for population', pop.name)
            
def set_FB_on(value):
    for pop in populations():
        if hasattr(pop,'FB_on')==True:
            pop.FB_on = value
            print('Set FB_on =', value, 'for population', pop.name)
            
def set_LAT_on(value):
    for pop in populations():
        if hasattr(pop,'LAT_on')==True:
            pop.LAT_on = value
            print('Set LAT_on =', value, 'for population', pop.name)
            


if __name__=='__main__':
    compile()

    #Parse arguments
    arguments=sys.argv[1:]

    if "-j" in arguments:
        if arguments.index("-j")+1<len(arguments):
                num_threads=int(arguments[arguments.index("-j")+1])

    ##---Load network---##
    if "load" in arguments:
        if arguments.index("load")+1<len(arguments):
            path=arguments[arguments.index("load")+1]

            # Creation of the network
            net = load(path)
            print("Starting from saved file!")
            
        else:
            print("No path given.\nExample:",sys.argv[0],"load ./results/network_20000000.data")
            exit()
    
    else: 
        net= Network(True)

    ##---Count neurons and synapses---##
    #Neuron amount
    total_neurons=0
    for pop in populations():
        total_neurons=total_neurons+pop.size
        print("The amount of neurons of", pop.name, "is", str(pop.size), "with geometry", str(pop.geometry))
    print("The total amount of neurons is", str(total_neurons))
    
    #Synapse amount
    total_connections=0
    for proj in projections():
        total_connections=total_connections+proj.nb_synapses
        print("The amount of connections from", proj.pre.name, "to", proj.post.name, "type", proj.target, "is", str(proj.nb_synapses))
    print("The total amount of connections is", str(total_connections))



    ##---Training---##
    if 'learn' in arguments:
        report(filename="./results/model_description.tex")
        #Train the network for 20m steps (200,000 stimulus presentations) on natural scenes
        if get_current_step()<20000000:
            #Train the network for the remaining timesteps
            learn_task(get_current_step(),20000000,100)



    ##---Select network evaluations---##
    if 'eval' in arguments:
        #Turn off plasticities for evaluations
        #Intrinsic plasticity
        print('Turn of intrinsic plasticity')
        for pop in populations():
            if pop.name != 'LGN':
                 pop.ip_on = 0

        #Synaptic plasticity
        print('Turn of synaptic plasticity')
        for proj in projections():
            proj.disable_learning()

        if 'compression' in arguments:
            compression=1
        else:
            compression=0

        #Evaluations
        for arg in arguments:

            if arg=='MNIST':
                import PresentImages
                set_FB_on(1)
                PresentImages.present_Images(LGN,'./DATA/MNIST_train.mat','./MNIST_responses/train_responses',100, 0.10)
                PresentImages.present_Images(LGN,'./DATA/MNIST_test.mat','./MNIST_responses/test_responses',100, 0.10)
                PresentImages.present_Images(LGN,'./DATA/MNIST_train.mat','./MNIST_responses/train_responses_t30',30, 0.10)
                PresentImages.present_Images(LGN,'./DATA/MNIST_test.mat','./MNIST_responses/test_responses_t30',30, 0.10)
                
                set_FB_on(0)
                PresentImages.present_Images(LGN,'./DATA/MNIST_train.mat','./MNIST_responses/train_responses_FB_off',100, 0.10)
                PresentImages.present_Images(LGN,'./DATA/MNIST_test.mat','./MNIST_responses/test_responses_FB_off',100, 0.10)
                PresentImages.present_Images(LGN,'./DATA/MNIST_train.mat','./MNIST_responses/train_responses_t30_FB_off',30, 0.10)
                PresentImages.present_Images(LGN,'./DATA/MNIST_test.mat','./MNIST_responses/test_responses_t30_FB_off',30, 0.10)
                
                set_FB_on(1)
                
            if arg=='MNISTd':
                import PresentImages
                set_FB_on(1)
                PresentImages.present_Imagesd(LGN,'./DATA/MNIST_train.mat','./MNIST_responses/train_responses_t42d',42, 0.10)
                PresentImages.present_Imagesd(LGN,'./DATA/MNIST_test.mat','./MNIST_responses/test_responses_t42d',42, 0.10)
                
                set_FB_on(0)
                PresentImages.present_Imagesd(LGN,'./DATA/MNIST_train.mat','./MNIST_responses/train_responses_t42d_FB_off',42, 0.10)
                PresentImages.present_Imagesd(LGN,'./DATA/MNIST_test.mat','./MNIST_responses/test_responses_t42d_FB_off',42, 0.10)
                
                set_FB_on(1)
                
            if arg=='CIFAR10':
                import PresentImages
                set_FB_on(1)
                PresentImages.present_Images(LGN,'./DATA/CIFAR-10_train_gray.mat','./CIFAR10_responses/train_responses',100, 0.2)
                PresentImages.present_Images(LGN,'./DATA/CIFAR-10_test_gray.mat','./CIFAR10_responses/test_responses',100, 0.2)
                PresentImages.present_Images(LGN,'./DATA/CIFAR-10_train_gray.mat','./CIFAR10_responses/train_responses_t30',30, 0.2)
                PresentImages.present_Images(LGN,'./DATA/CIFAR-10_test_gray.mat','./CIFAR10_responses/test_responses_t30',30, 0.2)
                
                set_FB_on(0)
                PresentImages.present_Images(LGN,'./DATA/CIFAR-10_train_gray.mat','./CIFAR10_responses/train_responses_FB_off',100, 0.2)
                PresentImages.present_Images(LGN,'./DATA/CIFAR-10_test_gray.mat','./CIFAR10_responses/test_responses_FB_off',100, 0.2)
                PresentImages.present_Images(LGN,'./DATA/CIFAR-10_train_gray.mat','./CIFAR10_responses/train_responses_t30_FB_off',30, 0.2)
                PresentImages.present_Images(LGN,'./DATA/CIFAR-10_test_gray.mat','./CIFAR10_responses/test_responses_t30_FB_off',30, 0.2)
                
                set_FB_on(1)
                
            if arg=='CIFAR10d':
                import PresentImages
                set_FB_on(1)
                PresentImages.present_Imagesd(LGN,'./DATA/CIFAR-10_train_gray.mat','./CIFAR10_responses/train_responses_t42d',42, 0.2)
                PresentImages.present_Imagesd(LGN,'./DATA/CIFAR-10_test_gray.mat','./CIFAR10_responses/test_responses_t42d',42, 0.2)
                
                set_FB_on(0)
                PresentImages.present_Imagesd(LGN,'./DATA/CIFAR-10_train_gray.mat','./CIFAR10_responses/train_responses_t42d_FB_off',42, 0.2)
                PresentImages.present_Imagesd(LGN,'./DATA/CIFAR-10_test_gray.mat','./CIFAR10_responses/test_responses_t42d_FB_off',42, 0.2)
                
                set_FB_on(1)
                
            if arg=='SVHN':
                import PresentImages
                set_FB_on(1)
                PresentImages.present_Images(LGN,'./DATA/SVHN_train.mat','./SVHN_responses/train_responses',100, 0.2)
                PresentImages.present_Images(LGN,'./DATA/SVHN_test.mat','./SVHN_responses/test_responses',100, 0.2)
                PresentImages.present_Images(LGN,'./DATA/SVHN_train.mat','./SVHN_responses/train_responses_t30',30, 0.2)
                PresentImages.present_Images(LGN,'./DATA/SVHN_test.mat','./SVHN_responses/test_responses_t30',30, 0.2)
                
                set_FB_on(0)
                PresentImages.present_Images(LGN,'./DATA/SVHN_train.mat','./SVHN_responses/train_responses_FB_off',100, 0.2)
                PresentImages.present_Images(LGN,'./DATA/SVHN_test.mat','./SVHN_responses/test_responses_FB_off',100, 0.2)
                PresentImages.present_Images(LGN,'./DATA/SVHN_train.mat','./SVHN_responses/train_responses_t30_FB_off',30, 0.2)
                PresentImages.present_Images(LGN,'./DATA/SVHN_test.mat','./SVHN_responses/test_responses_t30_FB_off',30, 0.2)
                
                set_FB_on(1)
                
            if arg=='SVHNd':
                import PresentImages
                set_FB_on(1)
                PresentImages.present_Imagesd(LGN,'./DATA/SVHN_train.mat','./SVHN_responses/train_responses_t42d',42, 0.2)
                PresentImages.present_Imagesd(LGN,'./DATA/SVHN_test.mat','./SVHN_responses/test_responses_t42d',42, 0.2)
                
                set_FB_on(0)
                PresentImages.present_Imagesd(LGN,'./DATA/SVHN_train.mat','./SVHN_responses/train_responses_t42d_FB_off',42, 0.2)
                PresentImages.present_Imagesd(LGN,'./DATA/SVHN_test.mat','./SVHN_responses/test_responses_t42d_FB_off',42, 0.2)
                
                set_FB_on(1)
                
            if arg=='EMNIST':
                import PresentImages
                set_FB_on(1)
                PresentImages.present_Images(LGN,'./DATA/EMNIST_train.mat','./EMNIST_responses/train_responses',100, 0.15)
                PresentImages.present_Images(LGN,'./DATA/EMNIST_test.mat','./EMNIST_responses/test_responses',100, 0.15)
                PresentImages.present_Images(LGN,'./DATA/EMNIST_train.mat','./EMNIST_responses/train_responses_t30',30, 0.15)
                PresentImages.present_Images(LGN,'./DATA/EMNIST_test.mat','./EMNIST_responses/test_responses_t30',30, 0.15)
                
                set_FB_on(0)
                PresentImages.present_Images(LGN,'./DATA/EMNIST_train.mat','./EMNIST_responses/train_responses_FB_off',100, 0.15)
                PresentImages.present_Images(LGN,'./DATA/EMNIST_test.mat','./EMNIST_responses/test_responses_FB_off',100, 0.15)
                PresentImages.present_Images(LGN,'./DATA/EMNIST_train.mat','./EMNIST_responses/train_responses_t30_FB_off',30, 0.15)
                PresentImages.present_Images(LGN,'./DATA/EMNIST_test.mat','./EMNIST_responses/test_responses_t30_FB_off',30, 0.15)
                
                set_FB_on(1)
                
            if arg=='EMNISTd':
                import PresentImages
                set_FB_on(1)
                PresentImages.present_Imagesd(LGN,'./DATA/EMNIST_train.mat','./EMNIST_responses/train_responses_t42d',42, 0.15)
                PresentImages.present_Imagesd(LGN,'./DATA/EMNIST_test.mat','./EMNIST_responses/test_responses_t42d',42, 0.15)
                
                set_FB_on(0)
                PresentImages.present_Imagesd(LGN,'./DATA/EMNIST_train.mat','./EMNIST_responses/train_responses_t42d_FB_off',42, 0.15)
                PresentImages.present_Imagesd(LGN,'./DATA/EMNIST_test.mat','./EMNIST_responses/test_responses_t42d_FB_off',42, 0.15)
                
                set_FB_on(1)
                
            if arg=='ETH80':
                import PresentImages
                set_FB_on(1)
                #present_Images_tiles2(Inputlayer, inputdata, savefile, presentation_time, norm_factor, norm_max_value, tilesX, tilesY, overlapX, overlapY):
                PresentImages.present_Images_tiles2(LGN,'./DATA/ETH80_gray_vector.mat','./ETH80_responses/train_responses',100, 0.05, compression=compression)
                PresentImages.present_Images_tiles2(LGN,'./DATA/ETH80_gray_vector.mat','./ETH80_responses/train_responses_t30',30, 0.05, compression=compression)
                PresentImages.present_Images_tiles2(LGN,'./DATA/ETH80_gray_vector.mat','./ETH80_responses/train_responses_t30_o8',30, 0.05, overlapX=8, overlapY=8, compression=compression)
                PresentImages.present_Images_tiles2(LGN,'./DATA/ETH80_gray_vector.mat','./ETH80_responses/train_responses_t30_o16',30, 0.05, overlapX=16, overlapY=16, compression=compression)

                set_FB_on(0)
                PresentImages.present_Images_tiles2(LGN,'./DATA/ETH80_gray_vector.mat','./ETH80_responses/train_responses_FB_off',100, 0.05, compression=compression)
                PresentImages.present_Images_tiles2(LGN,'./DATA/ETH80_gray_vector.mat','./ETH80_responses/train_responses_t30_FB_off',30, 0.05, compression=compression)
                PresentImages.present_Images_tiles2(LGN,'./DATA/ETH80_gray_vector.mat','./ETH80_responses/train_responses_t30_o8_FB_off',30, 0.05, overlapX=8, overlapY=8, compression=compression)
                PresentImages.present_Images_tiles2(LGN,'./DATA/ETH80_gray_vector.mat','./ETH80_responses/train_responses_t30_o16_FB_off',30, 0.05, overlapX=16, overlapY=16, compression=compression)
                
                set_FB_on(1)
                
            if arg=='ETH80d':
                import PresentImages
                set_FB_on(1)
                #present_Images_tiles2(Inputlayer, inputdata, savefile, presentation_time, norm_factor, norm_max_value, tilesX, tilesY, overlapX, overlapY):
                PresentImages.present_Images_tiles3(LGN,'./DATA/ETH80_gray_vector.mat','./ETH80_responses/train_responses_t42d',42, 0.05, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/ETH80_gray_vector.mat','./ETH80_responses/train_responses_t42d_o8',42, 0.05, overlapX=8, overlapY=8, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/ETH80_gray_vector.mat','./ETH80_responses/train_responses_t42d_o16',42, 0.05, overlapX=16, overlapY=16, compression=compression)

                set_FB_on(0)
                PresentImages.present_Images_tiles3(LGN,'./DATA/ETH80_gray_vector.mat','./ETH80_responses/train_responses_t42d_FB_off',42, 0.05, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/ETH80_gray_vector.mat','./ETH80_responses/train_responses_t42d_o8_FB_off',42, 0.05, overlapX=8, overlapY=8, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/ETH80_gray_vector.mat','./ETH80_responses/train_responses_t42d_o16_FB_off',42, 0.05, overlapX=16, overlapY=16, compression=compression)
                
                set_FB_on(1)
                
            if arg=='CalTech101':
                import PresentImages
                set_FB_on(1)
                #present_Images_tiles2(Inputlayer, inputdata, savefile, presentation_time, norm_factor, norm_max_value, tilesX, tilesY, overlapX, overlapY):
                PresentImages.present_Images_tiles2(LGN,'./DATA/CalTech101_images.mat','./CalTech101_responses/train_responses',100, 0.03, compression=compression)
                PresentImages.present_Images_tiles2(LGN,'./DATA/CalTech101_images.mat','./CalTech101_responses/train_responses_t30',30, 0.03, compression=compression)
                PresentImages.present_Images_tiles2(LGN,'./DATA/CalTech101_images.mat','./CalTech101_responses/train_responses_t30_o8',30, 0.03, overlapX=8, overlapY=8, compression=compression)
                PresentImages.present_Images_tiles2(LGN,'./DATA/CalTech101_images.mat','./CalTech101_responses/train_responses_t30_o16',30, 0.03, overlapX=16, overlapY=16, compression=compression)

                set_FB_on(0)
                PresentImages.present_Images_tiles2(LGN,'./DATA/CalTech101_images.mat','./CalTech101_responses/train_responses_FB_off',100, 0.03, compression=compression)
                PresentImages.present_Images_tiles2(LGN,'./DATA/CalTech101_images.mat','./CalTech101_responses/train_responses_t30_FB_off',30, 0.03, compression=compression)
                PresentImages.present_Images_tiles2(LGN,'./DATA/CalTech101_images.mat','./CalTech101_responses/train_responses_t30_o8_FB_off',30, 0.03, overlapX=8, overlapY=8, compression=compression)
                PresentImages.present_Images_tiles2(LGN,'./DATA/CalTech101_images.mat','./CalTech101_responses/train_responses_t30_o16_FB_off',30, 0.03, overlapX=16, overlapY=16, compression=compression)
                
                set_FB_on(1)
                
            if arg=='CalTech101d':
                import PresentImages
                set_FB_on(1)
                #present_Images_tiles3(Inputlayer, inputdata, savefile='test_resposes', presentation_time, norm_factor, norm_max_value, tilesX, tilesY, overlapX, overlapY, delays, compression):
                PresentImages.present_Images_tiles3(LGN,'./DATA/CalTech101_images.mat','./CalTech101_responses/train_responses_t42d',42, 0.03, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/CalTech101_images.mat','./CalTech101_responses/train_responses_t42d_o8',42, 0.03, overlapX=8, overlapY=8, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/CalTech101_images.mat','./CalTech101_responses/train_responses_t42d_o16',42, 0.03, overlapX=16, overlapY=16, compression=compression)

                set_FB_on(0)
                PresentImages.present_Images_tiles3(LGN,'./DATA/CalTech101_images.mat','./CalTech101_responses/train_responses_t42d_FB_off',42, 0.03, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/CalTech101_images.mat','./CalTech101_responses/train_responses_t42d_o8_FB_off',42, 0.03, overlapX=8, overlapY=8, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/CalTech101_images.mat','./CalTech101_responses/train_responses_t42d_o16_FB_off',42, 0.03, overlapX=16, overlapY=16, compression=compression)
                
                set_FB_on(1)
                
            if arg=='CalTech101Subsetd':
                import PresentImages
                set_FB_on(1)
                #present_Images_tiles3(Inputlayer, inputdata, savefile='test_resposes', presentation_time, norm_factor, norm_max_value, tilesX, tilesY, overlapX, overlapY, delays, compression):
                PresentImages.present_Images_tiles3(LGN,'./DATA/CalTech101_subset.mat','./CalTech101_responses/train_responses_t42sd',42, 0.03, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/CalTech101_subset.mat','./CalTech101_responses/train_responses_t42sd_o8',42, 0.03, overlapX=8, overlapY=8, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/CalTech101_subset.mat','./CalTech101_responses/train_responses_t42sd_o16',42, 0.03, overlapX=16, overlapY=16, compression=compression)

                set_FB_on(0)
                PresentImages.present_Images_tiles3(LGN,'./DATA/CalTech101_subset.mat','./CalTech101_responses/train_responses_t42sd_FB_off',42, 0.03, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/CalTech101_subset.mat','./CalTech101_responses/train_responses_t42sd_o8_FB_off',42, 0.03, overlapX=8, overlapY=8, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/CalTech101_subset.mat','./CalTech101_responses/train_responses_t42sd_o16_FB_off',42, 0.03, overlapX=16, overlapY=16, compression=compression)
                
                set_FB_on(1)
                
            if arg=='STL10d':
                import PresentImages
                set_FB_on(1)
                #present_Images_tiles3(Inputlayer, inputdata, savefile='test_resposes', presentation_time, norm_factor, norm_max_value, tilesX, tilesY, overlapX, overlapY, delays, compression):
                PresentImages.present_Images_tiles3(LGN,'./DATA/STL10_train.mat','./STL10_responses/train_responses_t42d',42, 0.06, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/STL10_test.mat','./STL10_responses/test_responses_t42d',42, 0.06, compression=compression)

                PresentImages.present_Images_tiles3(LGN,'./DATA/STL10_train.mat','./STL10_responses/train_responses_t42d_o16',42, 0.06, overlapX=16, overlapY=16, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/STL10_test.mat','./STL10_responses/test_responses_t42d_o16',42, 0.06, overlapX=16, overlapY=16, compression=compression)
                
                PresentImages.present_Images_tiles3(LGN,'./DATA/STL10_train.mat','./STL10_responses/train_responses_t42d_o8',42, 0.06, overlapX=8, overlapY=8, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/STL10_test.mat','./STL10_responses/test_responses_t42d_o8',42, 0.06, overlapX=8, overlapY=8, compression=compression)

                set_FB_on(0)
                PresentImages.present_Images_tiles3(LGN,'./DATA/STL10_train.mat','./STL10_responses/train_responses_t42d_FB_off',42, 0.06, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/STL10_test.mat','./STL10_responses/test_responses_t42d_FB_off',42, 0.06, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/STL10_train.mat','./STL10_responses/train_responses_t42d_o8_FB_off',42, 0.06, overlapX=8, overlapY=8, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/STL10_test.mat','./STL10_responses/test_responses_t42d_o8_FB_off',42, 0.06, overlapX=8, overlapY=8, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/STL10_train.mat','./STL10_responses/train_responses_t42d_o16_FB_off',42, 0.06, overlapX=16, overlapY=16, compression=compression)
                PresentImages.present_Images_tiles3(LGN,'./DATA/STL10_test.mat','./STL10_responses/test_responses_t42d_o16_FB_off',42, 0.06, overlapX=16, overlapY=16, compression=compression)
                
                set_FB_on(1)

    
    print('simulation done.',time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
