
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy the dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv pip install --system --no-cache -e .



# Set the python path
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Run the training script
CMD ["python", "src/training/train.py"]
