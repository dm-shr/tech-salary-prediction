FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.base.txt backend/requirements.fastapi.txt ./

RUN pip install --upgrade pip && \
    pip install -r requirements.base.txt && \
    pip install -r requirements.fastapi.txt

COPY backend/src/ ./src/
COPY backend/configs/ ./configs/
COPY backend/docker/fastapi/entrypoint.sh ./entrypoint.sh

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

ENV PYTHONPATH=/app

ENTRYPOINT ["/app/entrypoint.sh"]
