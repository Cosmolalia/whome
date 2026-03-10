# CHANGE: Use the 'devel' image to get the C++ headers (cuda_fp16.h)
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install Python and minimal tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# We use cupy-cuda12x which matches the container's CUDA version
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    requests \
    cupy-cuda12x

# Copy the Warhead files
COPY w_operator.py .
COPY w_cuda.py .
COPY client.py .

# The command that runs when the container starts
CMD ["python3", "client.py"]
