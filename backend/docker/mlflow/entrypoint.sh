#!/bin/bash
set -e

# Wait for MinIO to be ready
until curl -sf "http://minio:9000/minio/health/live"; do
    echo "Waiting for MinIO to be ready..."
    sleep 5
done

# Start MLflow server with proper artifact configuration
exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --default-artifact-root s3://mlflow-artifacts/ \
    --artifacts-destination s3://mlflow-artifacts/ \
    --serve-artifacts
