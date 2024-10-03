#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_TAG="puma-challenge-baseline-track2"

# Input/Output on the host
INPUT_DIR="${SCRIPT_DIR}/test"
INPUT_FILENAME=$(basename $(ls $INPUT_DIR/*.tif | head -n 1) .tif)
OUTPUT_DIR="${SCRIPT_DIR}/output"


echo "=+= (Re)build the container"
docker build "$SCRIPT_DIR" \
  --platform=linux/amd64 \
  --tag $DOCKER_TAG 2>&1


echo "=+= Doing a forward pass"

docker volume create puma-output

docker run \
    --rm \
    --shm-size=1g \
    --memory=4g \
    --platform=linux/amd64 \
    --network none \
    --gpus all \
    -v $SCRIPT_DIR/test/:/input/images/melanoma-wsi/ \
    -v puma-output:/output/ \
    $DOCKER_TAG

# Ensure permissions are set correctly on the output
# This allows the host user (e.g. you) to access and handle these files
docker run --rm \
    --quiet \
    --env HOST_UID=`id --user` \
    --env HOST_GID=`id --group` \
    -v puma-output:/output/ \
    alpine:latest \
    /bin/sh -c 'chown -R ${HOST_UID}:${HOST_GID} /output'


# Check if the output files exist after the container has run
echo "=+= Checking output files..."
EXPECTED_FILE_TISSUE="/output/images/melanoma-tissue-mask-segmentation/${INPUT_FILENAME}.tif"
EXPECTED_FILE_NUCLEI="/output/melanoma-10-class-nuclei-segmentation.json"

docker run --rm \
    -v puma-output:/output/ \
    python:3.7-slim sh -c "ls /output/"

# Check for nuclei segmentation JSON
if docker run --rm -v puma-output:/output/ python:3.7-slim sh -c "[ -f $EXPECTED_FILE_NUCLEI ]"; then
    echo "=+= Expected output for nuclei is correct."

    # Create a temporary container that mounts the volume
    TEMP_CONTAINER_NUCLEI=$(docker create -v puma-output:/output alpine:latest /bin/sh)

    # Copy the nuclei .json file to host for inspection
    docker cp "$TEMP_CONTAINER_NUCLEI:$EXPECTED_FILE_NUCLEI" "$SCRIPT_DIR/output/melanoma-10-class-nuclei-segmentation.json"
    echo "Copied .json file to host: $SCRIPT_DIR/output/melanoma-10-class-nuclei-segmentation.json"

    # Remove the temporary container for nuclei
    docker rm $TEMP_CONTAINER_NUCLEI
else
    echo "=+= Expected output for nuclei not found!"
    exit 1
fi

# Check for the dynamically named tissue .tif file
if docker run --rm -v puma-output:/output/ python:3.7-slim sh -c "[ -f $EXPECTED_FILE_TISSUE ]"; then
    echo "=+= Expected output for tissue is correct."

    # Create a temporary container that mounts the volume
    TEMP_CONTAINER_TISSUE=$(docker create -v puma-output:/output alpine:latest /bin/sh)

    # Copy the .tif file to host for inspection
    docker cp "$TEMP_CONTAINER_TISSUE:$EXPECTED_FILE_TISSUE" "$SCRIPT_DIR/output/${INPUT_FILENAME}.tif"
    echo "Copied .tif file to host: $SCRIPT_DIR/output/${INPUT_FILENAME}.tif"

    # Remove the temporary container for tissue
    docker rm $TEMP_CONTAINER_TISSUE

    # Inspect the .tif file using Python (optional)
    python3 -c "
import tifffile
file_path = '$SCRIPT_DIR/output/${INPUT_FILENAME}.tif'
with tifffile.TiffFile(file_path) as tif:
    for page in tif.pages:
        print(f'Page shape: {page.shape}, dtype: {page.dtype}, metadata: {page.tags}')
"
else
    echo "=+= Expected output for tissue not found!"
    exit 1
fi

echo "=+= Wrote results to ${OUTPUT_DIR}"

echo "=+= Save this image for uploading via save.sh \"${DOCKER_TAG}\""

docker volume rm puma-output
