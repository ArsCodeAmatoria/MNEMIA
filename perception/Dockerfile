# Perception Service Dockerfile for MNEMIA (Python AI)
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user and cache directory
RUN groupadd -r app && useradd -r -g app app

# Create cache directories with proper permissions
RUN mkdir -p /app /home/app/.cache && \
    chown -R app:app /app /home/app

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies as app user
USER app
RUN pip install --user --upgrade pip && \
    pip install --user -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Set environment for user installations
ENV PATH="/home/app/.local/bin:$PATH" \
    TRANSFORMERS_CACHE="/home/app/.cache/transformers" \
    HF_HOME="/home/app/.cache/huggingface"

EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["python", "main.py"] 