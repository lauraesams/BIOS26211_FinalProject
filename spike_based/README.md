Spike based neural network
==========================

## Structure
- **Evaluation**
    - **Caltech_f_mb**
        - **network_data**: Weight files of the spike based neural network
        - **samples**: Contains the files of the corresponding dataset
        - *net.py*: ANNarchy implementation of the complete spike based network
        - *run.py*: Python script to record the neuron activity
        - *svm.py*: Python script to use a linear SVM to measure the performance of both excitatory populations 
        - *svm_IN.py*: Python script to use a linear SVM to measure the performance of both inhibitory populations 
    - **CIFAR_10**
        - **network_data**: Weight files of the spike based neural network
        - **samples**: Contains the files of the corresponding dataset
        - *net.py*: ANNarchy implementation of the complete spike based network
        - *run.py*: Python script to record the neuron activity
        - *fitt_svm.py*: Python script to fitt a linear SVM with the recorded activity on the training set
        - *pred_svm.py*: Python script to measure the performance on the test set with the previous fitted SVM
        - *RDM.py*: Python script to create the representational dissimilarity matrix
        - *tsne.py*: Python script to use the t-SNE algorithm
    - **E_MNIST**
        - **network_data**: Weight files of the spike based neural network
        - **samples**: Contains the files of the corresponding dataset
        - *net.py*: ANNarchy implementation of the complete spike based network
        - *run.py*: Python script to record the neuron activity
        - *svm.py*: Python script to use a linear SVM to measure the network performance 
        - *RDM.py*: Python script to create the representational dissimilarity matrix
        - *tsne.py*: Python script to use the t-SNE algorithm
    - **ETH_80**
        - **network_data**: Weight files of the spike based neural network
        - **samples**: Contains the files of the corresponding dataset
        - *net.py*: ANNarchy implementation of the complete spike based network
        - *run.py*: Python script to record the neuron activity
        - *svm.py*: Python script to use a linear SVM to measure the network performance 
    - **MNIST**
        - **network_data**: Weight files of the spike based neural network
        - **samples**: Contains the files of the corresponding dataset
        - *net.py*: ANNarchy implementation of the complete spike based network
        - *run.py*: Python script to record the neuron activity
        - *svm.py*: Python script to use a linear SVM to measure the network performance 
        - *RDM.py*: Python script to create the representational dissimilarity matrix
        - *tsne.py*: Python script to use the t-SNE algorithm
    - **SVHN**
        - **network_data**: Weight files of the spike based neural network
        - **samples**: Contains the files of the corresponding dataset
        - *net.py*: ANNarchy implementation of the complete spike based network
        - *run.py*: Python script to record the neuron activity
        - *svm.py*: Python script to use a linear SVM to measure the network performance 

- **Model**
    - **STDP_L1_L2**
        - **input** : directory for the input data
        - *ann_L4.py* : Python script to start the training of the V1 layer 4 excitatory and inhibitory simple cells 
        - *ann_L2_3.py* : Python script to start the training of the V1 layer 2/3 excitatory and inhibitory cells 
        - *rezeptiv.py* : Python script to plot the receptive fields of the V1 layer 4 excitatory and inhibitory simple cells

Please look at the README-Files in the corresponding subdirectories to obtain more information about requirements and the Python scripts.
