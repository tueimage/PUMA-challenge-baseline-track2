import os
import struct

import numpy as np
from PIL import Image
import tifffile
import SimpleITK as sitk
from PIL.TiffTags import TAGS

# Paths for the dictionary file and output folder
rename_dict = 'convert_dict.txt'
pred_folder = 'nnunetv2/nnunetv2_hist/nnUNet_raw/Dataset526_Mark/imagesTs_pred'
output_folder = '/output/images/melanoma-tissue-mask-segmentation'


def recover_filename(pred_folder, output_folder, rename_dict):
    # Create a dictionary to map the new file names (gt_name) to the original image names (img_name)
    rename_mapping = {}

    # create output path
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read and parse the dictionary file
    with open(rename_dict, 'r') as txt:
        lines = txt.readlines()
        for line in lines:
            line = line.strip()
            if line:
                old_name, img_name, gt_name = line.split(', ')
                rename_mapping[gt_name] = old_name.replace('.tiff', '.tif')  # Map the new name (gt_name) to the old name (img_name)
    
    # List all PNG files in the output folder
    files = [f for f in os.listdir(pred_folder) if f.endswith('.png')]

    # Process each file
    for file in files:
        # Check if the file exists in the rename mapping
        if file in rename_mapping:
            img = Image.open(os.path.join(pred_folder, file))
            img_array = np.array(img)

            # Cast voxel values (the entire image array) to int8
            img_array = img_array.astype(np.int8)

            unique_segments = np.unique(img_array)
            print(f'Unique segments: {unique_segments}, Count: {len(unique_segments)}')

            min_val, max_val = img_array.min(), img_array.max()

            # Modify TIFF resolution metadata directly using tifffile
            img_name = rename_mapping[file]  # Get the corresponding old name (img_name)
            new_file_path = os.path.join(output_folder, img_name)


            # Write the image with the correct resolution
            with tifffile.TiffWriter(new_file_path) as tif:
                tif.write(
                    img_array,
                    resolution=(300, 300),  # Set resolution to 300 DPI for both X and Y
                    extratags=[
                        ('MinSampleValue', 'I', 1, int(1)),
                        ('MaxSampleValue', 'I', 1, int(max_val)),
                    ]
                )


            # Verify the new resolution
            with tifffile.TiffFile(new_file_path) as tif:
                for i, page in enumerate(tif.pages):
                    print(f"Page {i} shape: {page.shape}, dtype: {page.dtype}")
                    print(f"XResolution: {page.tags['XResolution'].value}")
                    print(f"YResolution: {page.tags['YResolution'].value}")
                    print(f"ResolutionUnit: {page.tags['ResolutionUnit'].value}")
                    for tag in page.tags.values():
                        name, value = tag.name, tag.value
                        print(f"{name}: {value}")

            print(f'Wrote tissue file at: {new_file_path}')


# Run the function
recover_filename(pred_folder, output_folder, rename_dict)
