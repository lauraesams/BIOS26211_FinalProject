import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import os

#['train_gray','train_gray_protan','train_gray_deutan','test_gray','test_gray_protan','test_gray_deutan'] 

for t in ['train_gray','test_gray']: 
  #setting source images and saving path
  source_dir = f"{t}"
  pca_dir = f"{t}_PCA"

  # Get list of images
  filenames = os.listdir(source_dir) #find path of images

  #loop through images
  for i, file in enumerate(filenames):

    source_path = os.path.join(source_dir, file)
    file_name = file.replace(".png", "") # strip .png extension

    # Loading the image
    img = cv2.imread(source_path) #use preprocessed images
    plt.imshow(img)

    # Splitting the image in R,G,B arrays.
    blue,green,red = cv2.split(img)
    #it will split the original image into Blue, Green and Red arrays.

    #initialize PCA with first 20 principal components
    pca = PCA(20)

    #Applying to red channel and then applying inverse transform to transformed array.
    red_transformed = pca.fit_transform(red)
    red_inverted = pca.inverse_transform(red_transformed)

    #Applying to Green channel and then applying inverse transform to transformed array.
    green_transformed = pca.fit_transform(green)
    green_inverted = pca.inverse_transform(green_transformed)

    #Applying to Blue channel and then applying inverse transform to transformed array.
    blue_transformed = pca.fit_transform(blue)
    blue_inverted = pca.inverse_transform(blue_transformed)


    img_compressed = (np.dstack((red_inverted, red_inverted, red_inverted))).astype(np.uint8)

    #save images
    dest_dir = os.path.join(pca_dir, f"{file_name}_pca.png")
    Image.fromarray(img_compressed).save(dest_dir)
  
  print(f"{t} completed.")

print("PCA_images script completed.")
