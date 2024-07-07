# Use the official PyTorch image as the base image
FROM pytorch/pytorch

# Set the working directory in the container
WORKDIR /workspace

# Install the necessary packages
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --upgrade pip && \
    pip install -U torch && \
    pip install pandas transformers accelerate peft trl wandb bitsandbytes huggingface_hub

# Specify the default command to run when starting the container
CMD ["python", "repos/preprocessing/training.py"]
