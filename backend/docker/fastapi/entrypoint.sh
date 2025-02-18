#!/bin/bash

# Run FastAPI server using Uvicorn
uvicorn src.fastapi_app:app --host 0.0.0.0 --port 8000
