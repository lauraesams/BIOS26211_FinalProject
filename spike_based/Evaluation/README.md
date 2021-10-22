Spike based neural network - Evaluation
========================================


## Requirements

- Python >= 3.6
- ANNarchy >= 4.6
- numpy >= 1.19
- matplotlib >= 3.3
- scipy >= 1.5
- scikit-learn >= 0.24

## Measure Accuracy

1. Put all weight-files to the **network_data** directory (*V1weight_N.txt*, *InhibW_N.txt*, *INtoV1_N.txt*, *INLat_N.txt*, *V1toIN_N.txt*, *V2weight_N.txt*, *V1toIN2_N.txt*, *V2toIN2_N.txt*, *IN2toV2_N.txt*,*IN2Lat_N.txt*) and remove the number from the name of each file (for example: rename *V1weight_N.txt* to *V1weight.txt*).

2. Make sure, that the **samples** directory contains the input images of the corresponding dataset.

3. Record the neuron activity with:
```
python run.py
```
It creates a new directory **output**, containing the firing rates of the different populations for each sample of the training- and test- set.

4. To use a linear SVM to measure the performance of all excitatory and inhibitory populations start the *svm.py* Python file. 
   For **CIFAR_10** use *fitt_svm.py* to fitt the SVM with the response vectors on the training set and use *pred_svm.py* to measure the performance of the response vectors of the test set.
   In **Caltech_f_mb** start additionally the **svm_IN.py** to measure the performance of the inhibitory populations.
5. Additionally, the **E_MNIST**, **MNIST** and **CIFAR_10** directories contain a *RDM.py* file to create the representational dissimilarity matrix 
   and a *tsne.py* to use the t-SNE algorithm to create a 2D-Plot of the multi-dimensional response vectors.
