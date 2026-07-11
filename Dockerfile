# -----------------------------------------------------------------------------
# Recommender Demo - Dockerfile
# -----------------------------------------------------------------------------

FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools==69.5.1 wheel Cython && pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Environment variables
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV TOP_K=10
ENV ARTIFACTS_DIR=/app/data
ENV MLFLOW_TRACKING_URI=sqlite:////app/data/mlflow.db

# Start FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]