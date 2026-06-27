import os
import h5py
import numpy as np
import shutil

input_folder = "dataset/BraTS2020_training_data/content/data"
output_folder = "dataset/filtered"

os.makedirs(output_folder, exist_ok=True)

count = 0

for file_name in os.listdir(input_folder):
    if file_name.endswith(".h5"):
        path = os.path.join(input_folder, file_name)

        file = h5py.File(path, "r")
        mask = file["mask"][:]

        # convert mask
        if len(mask.shape) == 3:
            mask = mask[:,:,0]

        # tumor check
        if np.sum(mask) > 0:
            shutil.copy(path, os.path.join(output_folder, file_name))
            count += 1

print("Tumor slices saved:", count)