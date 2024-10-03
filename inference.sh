#!/usr/bin/env bash

# Step 1: perform nuclei inference with hovernext
python process.py "$@"

# Step 2: perform tissue inference with nnunet

# create the original name and nnunet name list
python create_convert_dict.py
# reload tiff image into png and change name
python image_transfer.py
# inference
cd ./nnunetv2/inference/
python predict_from_raw_data.py
# convert gt json file into png with correspond name
cd ../../
python output_rename.py

