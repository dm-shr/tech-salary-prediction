FROM apache/airflow:2.7.3-python3.10

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create directories with correct permissions
RUN mkdir -p /opt/airflow/repo/data /opt/airflow/repo/models && \
    mkdir -p /opt/airflow/repo/.dvc/cache /opt/airflow/repo/.dvc/tmp && \
    chown -R airflow:root /opt/airflow/repo && \
    chmod -R 775 /opt/airflow/repo && \
    chmod -R 777 /opt/airflow/repo/.dvc/cache && \
    chmod -R 777 /opt/airflow/repo/.dvc/tmp && \
    chmod g+s /opt/airflow/repo/.dvc/cache && \
    chmod g+s /opt/airflow/repo/.dvc/tmp

# Install base requirements first for better caching
COPY backend/requirements.base_debug.txt backend/requirements.airflow.txt /opt/airflow/

USER airflow

# Set HOME and PATH environment variables
ENV HOME=/home/airflow \
    PATH="/home/airflow/.local/bin:${PATH}" \
    PYTHONPATH="/home/airflow/.local/lib/python3.10/site-packages:${PYTHONPATH}"

# Create .gitconfig directory with proper permissions
RUN mkdir -p ${HOME}/.config/git && \
    chmod 700 ${HOME}/.config

# Install DVC and other dependencies
RUN pip install --upgrade pip && \
    pip install 'dvc[s3]' && \
    # pip install -r /opt/airflow/requirements.base.txt && \
    pip install -r /opt/airflow/requirements.base_debug.txt && \
    pip install -r /opt/airflow/requirements.airflow.txt && \
    dvc --version

# Add git configuration
RUN git config --global user.email "airflow@example.com" && \
    git config --global user.name "Airflow" && \
    git config --global --add safe.directory /opt/airflow/repo && \
    git config --global init.defaultBranch main
