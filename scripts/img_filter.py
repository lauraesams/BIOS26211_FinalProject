"""
This script prepares training and test images for further preprocessing by 
applying Colorblind simulation filters and converting them to greyscale, then
saving them to their respective folders.

Author: Laura Sams
"""
import os
import numpy as np
from PIL import Image
from daltonlens import simulate, convert, generate

for t in ["test", "train"]:
    # Set source and destination directories
    source_dir = f"{t}"
    gray_dest_dir = f"{t}_gray"
    deutan_dest_dir = f"{t}_gray_deutan"
    protan_dest_dir = f"{t}_gray_protan"

    # Get list of images
    filenames = os.listdir(source_dir)

    # Start simulator using Machado (2009) algorithm
    simulator = simulate.Simulator_Machado2009()

    # Loop through images
    for i, file in enumerate(filenames):
        source_path = os.path.join(source_dir, file)
        file_name = file.replace(".png", "") # strip .png extension

        # Grayscale image
        img = np.asarray(Image.open(source_path).convert('RGB'))
        img_gray = Image.fromarray(img).convert('L')
        dest_dir = os.path.join(gray_dest_dir, f"{file_name}_gr.png")
        img_gray.save(dest_dir)

        # Apply protanopia filter and grayscale
        protan_im = Image.fromarray(simulator.simulate_cvd(img, simulate.Deficiency.PROTAN, severity=1))
        protan_img_gray = protan_im.convert('L')
        dest_dir = os.path.join(protan_dest_dir, f"{file_name}_pr_gr.png")
        protan_img_gray.save(dest_dir)

        # Apply deuteranopia filter and grayscale
        deutan_im = Image.fromarray(simulator.simulate_cvd(img, simulate.Deficiency.DEUTAN, severity=1))
        deutan_img_gray = deutan_im.convert('L')
        dest_dir = os.path.join(deutan_dest_dir, f"{file_name}_de_gr.png")
        deutan_img_gray.save(dest_dir)
    
    print(f"{t} completed.")

print("Prep_images script completed.")
