# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements files first
COPY backend/requirements.base.txt backend/requirements.streamlit.txt ./

# Install Python dependencies - this layer will be cached unless requirements change
RUN pip install --upgrade pip && \
    pip install -r requirements.base.txt && \
    pip install -r requirements.streamlit.txt

# Create directories for models and data
RUN mkdir -p /app/models /app/data

# Copy source code and other files - these change more frequently
COPY backend/src/ ./src/
COPY backend/configs/ ./configs/
COPY backend/docker/streamlit/entrypoint.sh ./entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app

# Default command to run Streamlit app
ENTRYPOINT ["/app/entrypoint.sh"]
