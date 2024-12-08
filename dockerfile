# Base image for training
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 as training

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Base image for inference
FROM python:3.10-slim as inference

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create workspace directory
WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Copy project files
COPY . .

# Development image
FROM training as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip3 install --no-cache-dir -r requirements-dev.txt

# Install pre-commit hooks
RUN git init && \
    pre-commit install

# Set default command for development container
CMD ["bash"]