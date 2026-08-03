FROM python:3.10-slim

# Install system dependencies needed for Pillow and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU PyTorch first to leverage Docker build cache layers
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu

# Copy API dependencies and install
COPY aws_deployment/inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy classification source modules and default validation files
COPY classification /app/classification
COPY validate.py /app/validate.py

# Copy the FastAPI app wrapper directly to the root of our application folder
COPY aws_deployment/inference/app.py /app/app.py

# Create checkpoint placeholder directories
RUN mkdir -p /app/output/tiny/finetune

# Port 8000 for serving requests
EXPOSE 8000

# Set environment variables for config loading and Python runtime optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_CONFIG=/app/classification/configs/DAMamba/damamba_tiny.yaml
ENV MODEL_CHECKPOINT=/app/output/tiny/finetune/best_ckpt.pth

# Set path environment so modules can resolve classification imports
ENV PYTHONPATH=/app

# Start FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
