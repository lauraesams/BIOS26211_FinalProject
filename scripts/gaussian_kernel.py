import cv2
import os
from tqdm import tqdm

def gaussian_kernel(source_directory, destination_directory, blur_size=(7, 7)):
    """
    Credit: ChatGPT (modified)

    Applies Gaussian kernel to all images from source directory
    and saves the blurred images to the destination directory.
    
    Parameters:
    - source_directory: Path to the source directory containing the original images 
    - destination_directory: Path to the destination directory where blurred images will be saved.
    - blur_size: Size of the Gaussian kernel to be used
    """
    # Ensure the destination directory exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Retrieve all .png files from the source directory
    file_names = [file for file in os.listdir(source_directory) if file.endswith('.png')]

    # Iterate through each file in the source directory
    for file in tqdm(file_names, desc=f'Processing images in {source_directory}'):
        # Construct the full path to the current file
        source_file_path = os.path.join(source_directory, file)

        # Read the image from file
        image = cv2.imread(source_file_path)

        if image is None:
            print(f"Failed to load the image: {source_file_path}")
        
        # Apply Gaussian Kernel to the image
        blurred_image = cv2.GaussianBlur(image, blur_size, 0)
        
        # Construct the path to save the blurred image
        destination_file_path = os.path.join(destination_directory, file)
        
        # Save the blurred image
        cv2.imwrite(destination_file_path, blurred_image)

# EXAMPLE: apply Gaussian Kernel to images in 'test_gray' and save in 'test_gray_gaussian'
gaussian_kernel('test_gray', 'test_gray_gaussian')

# EXAMPLE: apply Gaussian Kernel to images in 'train_gray' and save in 'train_gray_gaussian'
gaussian_kernel('train_gray', 'train_gray_gaussian')
