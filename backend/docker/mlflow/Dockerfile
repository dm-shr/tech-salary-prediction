FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.mlflow.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY docker/mlflow/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]
