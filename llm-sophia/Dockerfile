# Sophia LLM Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    git \
    wget \
    curl \
    vim \
    htop \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional ML/DL packages
RUN pip install --no-cache-dir \
    flash-attn \
    xformers \
    bitsandbytes

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p \
    /app/data/{raw,processed,datasets,samples} \
    /app/models/{checkpoints,fine_tuned} \
    /app/logs/{training,evaluation} \
    /app/outputs/{generated_text,metrics} \
    /app/cache/{huggingface,datasets}

# Set permissions
RUN chmod +x scripts/*.py

# Environment variables for production
ENV HF_HOME=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/huggingface
ENV HF_DATASETS_CACHE=/app/cache/datasets
ENV SOPHIA_MODEL_PATH=/app/models/fine_tuned/sophia
ENV SOPHIA_API_PORT=8003
ENV LOG_LEVEL=INFO

# Expose API port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Default command (can be overridden)
CMD ["python", "scripts/serve.py"] 