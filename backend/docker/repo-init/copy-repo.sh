#!/bin/sh

# Create destination if it doesn't exist
mkdir -p /opt/airflow/repo

# Use rsync with exclusion patterns
rsync -av --exclude="logs" \
         --exclude="venv" \
         --exclude=".ipynb_checkpoints" \
         --exclude=".coverage" \
         --exclude="htmlcov" \
         --exclude="dist" \
         --exclude="build" \
         --exclude="*.egg-info" \
         --exclude="*.DS_Store" \
         /app/ /opt/airflow/repo/

# Set proper ownership
chown -R 50000:0 /opt/airflow/repo/
chmod -R 775 /opt/airflow/repo/

echo "Repository files copied successfully to /opt/airflow/repo/"
