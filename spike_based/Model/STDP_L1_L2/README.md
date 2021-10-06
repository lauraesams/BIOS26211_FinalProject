Spike based neural network - Training
=====================================

This directory contains the implementation of the spike based neural network for the https://doi.org/10.1016/j.neunet.2021.08.009 publication.
All excitatory synapses follow the Clopath et al. (2010) voltage-based triplet STDP rule and
all inhibitory synapses follow the Vogels et al. (2011) iSTDP rule.
The model uses the natural scene dataset from Olshausen et al. (1996) (https://www.rctn.org/bruno/sparsenet/IMAGES.mat)

## Requirements

- Python >= 3.6
- ANNarchy >= 4.6
- numpy >= 1.19
- matplotlib >= 3.3
- scipy >= 1.5

## Training

The neural network implements the V1 layer 4 and V1 layer 2/3. Each layer consists of a population of excitatory and inhibitory neurons.
Both layers are trained separately from each other. The following stpes are neccessary to start the training process:
1. Make sure that the natural scene file (IMAGES.mat) exists in the 'input' directory, if not please download it and copy it into the corresponding directory.
2. Start the training of the V1 layer 4 model with the command:

```
python ann_L4.py
```

3. After the training of V1 layer 4 is finished, copy the **rezeptiv.py** file into the *output_L4* directory and start it from there.
This function will create a new directory called *ONOFF*, which contains images of the receptive fields of the excitatory and inhibitory neurons.

4. Start the training of layer 2/3:

```
python ann_L2_3.py
```

After the training is done succesfully, there should be two directories: *output_L4* and *output_L2_3*, with the following structure:
- output_L4:
    - **excitatory** : contains the feedforward weights from the LGN to the excitatory(*V1weight_N.txt*) and to the inhibitory layer(*InhibW_N.txt*) 
    - **inhibitory** : contains the inhibitory feedback weights (*INtoV1_N.txt*) and the weights from the lateral inhibition (*INLat_N.txt*)
    - **V1toIN** : contains the weihts of the connection from the excitatory to the inhibitory population(*V1toIN_N.txt*)
    - **V1Layer** : contains an image with the recorded firing rates for four neurons 
    
- output_L2_3:
    - **exitatory** : contains the weights from layer 4 excitatory to layer 2/3 excitatory (*V2weight_N.txt*), to inhibitory layer 2/3 (*V1toIN2_N.txt*), corresponding synaptic delays (*V1toV2_Delay.txt*), and the weights from layer 2/3 excitatory to layer 2/3 inhibition(*V2toIN2_N.txt*)
    - **inhibitory** : contains the weights from layer 2/3 inhibitory to layer 2/3 excitatory (*IN2toV2_N.txt*) and the layer 2/3 lateral inhibitory weights (*IN2Lat_N.txt*)
    

Please notice that the *_N* refers to the numbers of patches presented already to the network as the corresping weights are saved. 

### Speed up Training

The complete training of the network can lead up to several days (depending on the machine).
To speed up the training you can reduce the complete number of presented patches. The default value is *400,000*, which leads to the best accuracy values.
At around *200,000* input patches, receptive fields emerge. But have in mind, that the accuracy values would be not as expected.

Another way to speed up the network training is to run the simulation on multiple cpu cores.
To do this, you can use the *-jn* argument, where *n* is the numbers of cpu cores you want to use in parallel.
For example:

```
python ann_L4.py -j4
```

or

```
python ann_L2_3.py -j4
```

