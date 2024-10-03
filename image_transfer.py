import os
from PIL import Image
import numpy as np
from skimage import io


def convert_filename(input_folder, images_folder, rename_dict):
    with open(rename_dict, 'r') as txt:
        lines = txt.readlines()
        lines = [line.strip() for line in lines]    
    files = [f for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('_context.tif')]
    for file in files:
        img = np.array(Image.open(os.path.join(input_folder, file)))[:, :, :3]
        for line in lines:
            old_name, img_name, gt_name = line.split(', ')
            if old_name == file:
                io.imsave(os.path.join(images_folder, img_name), img)
                break


input_folder = '/input/images/melanoma-wsi'
images_folder = 'nnunetv2/nnunetv2_hist/nnUNet_raw/Dataset526_Mark/imagesTs'
os.makedirs(images_folder, exist_ok=True)
rename_dict = 'convert_dict.txt'
convert_filename(input_folder, images_folder, rename_dict)