FROM python:3.10-slim

WORKDIR /app

COPY requirements.base_debug.txt requirements.fastapi.txt ./

RUN pip install --upgrade pip && \
    pip install -r requirements.base_debug.txt && \
    pip install -r requirements.fastapi.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY docker/fastapi/entrypoint.sh ./entrypoint.sh

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

ENV PYTHONPATH=/app

ENTRYPOINT ["/app/entrypoint.sh"]
