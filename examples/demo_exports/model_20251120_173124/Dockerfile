# Production Inference Container
# Generated for: demo-task
# Modality: text

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts and configs
COPY artifacts/ ./artifacts/
COPY configs/ ./configs/
COPY inference.py .

# Create non-root user for security
RUN useradd -m -u 1000 inference && \
    chown -R inference:inference /app
USER inference

# Expose inference port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=artifacts/model.onnx

# Run inference server
# Override this CMD with your preferred serving framework
CMD ["python", "inference.py", "--model", "artifacts/model.onnx", "--host", "0.0.0.0", "--port", "8080"]
