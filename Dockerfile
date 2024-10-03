FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install git (required for some packages in requirements.txt)
# Install system dependencies and pip
RUN apt-get update \
 && apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*


# create new group (name=user) and create new user (name=user)
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# make 3 directories and set permissions
RUN mkdir -p /opt/app /input /output  \
    && chown -R user:user /opt/app /input /output \
    && chmod -R 777 /output

USER user

WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

# Install Python dependencies as root user
COPY setup.py requirements.txt /opt/app/
RUN pip install -e . && pip install -r requirements.txt

# update pip and install wheel
RUN python -m pip install --user -U pip
RUN python -m pip install --user wheel

# copy all related files
COPY --chown=user:user image_transfer.py create_convert_dict.py output_rename.py inference.sh ./
COPY --chown=user:user nnunetv2/ ./nnunetv2/

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user checkpoint/ /opt/app/checkpoint/
COPY --chown=user:user src/ /opt/app/src/
COPY --chown=user:user process.py /opt/app/

# Set environment variables for data paths
ENV nnUNet_raw="/opt/app/nnunetv2/nnunetv2_hist/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/app/nnunetv2/nnunetv2_hist/nnUNet_preprocessed"
ENV nnUNet_results="/opt/app/nnunetv2/nnunetv2_hist/nnUNet_results"

# entrypoint to run
ENTRYPOINT ["./inference.sh"]

# default arguments, can be overridden
CMD ["--input", "/input/images/melanoma-wsi", "--output", "/output", "--cp", "/checkpoint", "--tta", "4", "--inf_workers", "4", "--pp_tiling", "10", "--pp_workers", "4"]

