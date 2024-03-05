import os
import cv2
from PIL import Image
#import simulate

for t in ["test_gray", "train_gray"]:
    # Set source and destination directories
    source_dir = t
    medblur_dest_dir = f"{t}_medblur"

    # Create destination directory if not exists
    if not os.path.exists(medblur_dest_dir):
        os.makedirs(medblur_dest_dir)

    # Get list of images
    filenames = os.listdir(source_dir)

    # Start simulator using Machado (2009) algorithm
    #simulator = simulate.Simulator_Machado2009()

    # Loop through images
    for i, file in enumerate(filenames):
        source_path = os.path.join(source_dir, file)
        file_name = os.path.splitext(file)[0]  # strip extension
        image = cv2.imread(source_path)
        median = cv2.medianBlur(image, 5)
        dest_path = os.path.join(medblur_dest_dir, f"{file_name}_gr.png")
        cv2.imwrite(dest_path, median)

    print(f"{t} completed.")

print("Prep_images script completed.")
