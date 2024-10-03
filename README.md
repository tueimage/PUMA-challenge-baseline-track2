# PUMA Challenge Baseline Track 2
This repository contains a Dockerized environment for running the PUMA Challenge Evaluation track using CUDA 12.1. The container includes all necessary dependencies to execute the model and run inference on input data. 

## Prerequisites
- Docker (Ensure that Docker is installed and supports GPU with CUDA 12.1 or newer)
- NVIDIA Docker Toolkit for GPU support

## Build the container
You can build the Docker image using the `build.sh` script. Ensure GPU support is enabled.

## Running the container.
Use the `test_run.sh` script to run the container.

## Input & Output
One input file will be mounted per container at (algorithm job) `/input/images/melanoma-whole-slide-image/<uuid>.tif`.

Two output files are expected files inside the `/output` directory:
- `melanoma-10-class-nuclei-segmentation.json` contains the nuclei predictions in "Multiple Polygons" format.
- `images/melanoma-tissue-mask-segmentation/<uuid>.tif` contains the tissue predictions, where pixels should be given the following values:
'Background': 0, 'Stroma': 1, 'Blood Vessel': 2, 'Tumor': 3, 'Epidermis': 4, and 'Necrosis': 5

In the `/test` directory, one example input case can be found.

## Saving the container
Use `save.sh` to save the container.
