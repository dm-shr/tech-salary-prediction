#!/bin/bash

# Run FastAPI server using Uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000
