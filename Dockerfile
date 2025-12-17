
# Use a base image with CUDA and cuDNN
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set non-interactive frontend for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install Python 3.13 and uv
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.13 python3.13-venv python3-pip && \
    pip install uv

# Link python3.13 to python
RUN ln -s /usr/bin/python3.13 /usr/bin/python

# Copy the dependency files
COPY pyproject.toml uv.lock* ./

# Copy verification scripts
COPY tests/ ./tests/

# Copy scripts directory
COPY scripts/ ./scripts/

# Copy src directory
COPY src/ ./src/

# Install dependencies using uv
# Using --system to install in the main python environment
RUN uv pip install --system --no-cache -e .

# Set the python path
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Set the entrypoint
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

# Default command to run the training script
CMD ["python", "-m", "src.training.train"]
