# NETWORK TRAINING
*/Model*

## Requirements
- Python 3 (3.6)
- ANNarchy >=4.6

## Common parameters
- "-j #" Number of threads.
- "learn" Required to start the learning routine.
- "eval" Required to start evaluation routines.
- "load path_to_file" Initializes the network with a save file.
- "MNISTd" Records the responses on the MNIST dataset, considering delays.
- "EMNISTd" Records the responses on the balanced EMNIST dataset, considering delays.
- "SVHNd" Records the responses on the SVHN dataset, considering delays.
- "CIFAR10d" Records the responses on the CIFAR10 dataset, considering delays.
- "ETH80d" Records the responses on the ETH80 dataset, considering delays.
- "CalTech101Subsetd" Records the responses on the subset of the CalTech101 dataset, considering delays.

## Example
*python3 main_Network.py -j 8 learn eval MNISTd EMNISTd SVHNd CIFAR10d ETH80d CalTech101Subsetd*


# EVALUATION
*/Evaluation*

## Requirements
- Matlab or Octave.
- The Matlab interface to liblinear or libsvm.
- The recorded network responses.
- The label files for the datasets.

## Functioning
- "calc_Accuracy.m" reads the recorded responses within the current folder.
- It calculates the accuracies for any specified ("layer_list") network population.
- The results are saved in a .txt and a .mat file.
