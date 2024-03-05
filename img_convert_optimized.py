"""
Convert .mat files to .png files

data from http://ufldl.stanford.edu/housenumbers/
Format 2: Cropped Digits

parallel version
"""
import os
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

def save_image_batch(images, labels, start_index, end_index, name_dic, save_path, lock, error_log):
    for i in range(start_index, end_index):
        img_temp = images[:, :, :, i]
        label_temp = labels[i][0]
        with lock:
            fig_name = f'{label_temp}_{name_dic[label_temp]}.png'
            name_dic[label_temp] += 1
        try:
            plt.imsave(os.path.join(save_path, fig_name), img_temp)
        except Exception as e:
            # Log any errors encountered during image saving
            with lock:
                error_log.append((i, e))

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
    name_dic = {l: 0 for l in labels}
    
    # create directory
    save_path = f'./converted_data/{data_type}'
    os.makedirs(save_path, exist_ok=True)
    
    # save as .png
    total_size = img.shape[3]
    num_threads = min(total_size, os.cpu_count())
    batch_size = total_size // num_threads
    
    lock = threading.Lock()  # Create a lock for thread safety
    error_log = []  # List to store error logs
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size if i < num_threads - 1 else total_size
            futures.append(executor.submit(save_image_batch, img, label, start_index, end_index, name_dic, save_path, lock, error_log))
        
        for future in tqdm(futures, total=len(futures), desc=f'Converting {data_type}'):
            future.result()
    
    # Print any errors encountered during image saving
    if error_log:
        print(f'Errors encountered during image saving in {data_type}:')
        for idx, error in error_log:
            print(f'Error occurred while saving image {idx}: {error}')

if __name__ == '__main__':
    convert_to_png('train')
    convert_to_png('test')