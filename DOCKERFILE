# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Start FROM NVIDIA PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:22.04-py3
RUN rm -rf /opt/pytorch  # remove 1.2GB dir

# Downloads to user config dir
# ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Install linux packages
RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

# Install pip packages
RUN python -m pip install --upgrade pip
RUN pip uninstall -y torch torchvision torchtext Pillow
RUN pip install --no-cache albumentations wandb gsutil notebook Pillow>=9.1.0 \
    torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt
# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app


# Copy contents
COPY . /usr/src/app
# RUN git clone https://github.com/ultralytics/yolov5 /usr/src/yolov5

# Set environment variables
ENV OMP_NUM_THREADS=8

ENTRYPOINT ["python", "track.py"]