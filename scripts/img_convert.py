"""
Convert .mat files to .png files

data from http://ufldl.stanford.edu/housenumbers/
Format 2: Cropped Digits

unoptimized version
"""
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 

def convert_to_png(data_type):
    """
    Convert .mat files to .png files

    Parameters
    ----------
    data_type : str
        'train' or 'test'
    
    Returns
    -------
    None
    """
    # load .mat file
    data = loadmat(f'./raw_data/{data_type}_32x32.mat')
    
    # get data
    img = data['X']
    label = data['y']
    
    labels = np.unique(label)
    name_dic = {}
    for l in labels:
        name_dic[l] = 0
    
    # create directory
    save_path = f'./converted_data/{data_type}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # save as .png
    total_size = img.shape[3]
    for i in tqdm(range(total_size)):
        img_temp = img[:,:,:,i]
        label_temp = label[i][0]
        fig_name = f'{label_temp}_{name_dic[label_temp]}.png'
        
        plt.imshow(img_temp)
        plt.axis('off')
        plt.savefig(os.path.join(save_path, fig_name), 
                    bbox_inches='tight', 
                    pad_inches=0)
        name_dic[label_temp] += 1

def load_img(data_type):
    """
    load .mat file as numpy array

    Parameters
    ----------
    data_type : str
        'train' or 'test'
    
    Returns
    -------
    img : numpy array
        images
    label : numpy array
        labels
    """
    # load .mat file
    data = loadmat(f'./raw_data/{data_type}_32x32.mat')
    
    # get data
    img = data['X']
    label = data['y']
    
    return img, label

if __name__ == '__main__':
    convert_to_png('test')
    print('Done with testing data!')
    convert_to_png('train')
    print('Done with training data!')
    
