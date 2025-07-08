#!/bin/bash

# Create directories for MLflow artifacts and backend store
mkdir -p mlflow_data/artifacts
mkdir -p mlflow_data/db

# Set MLflow server port
MLFLOW_PORT=5000

# Check if port is in use and adjust if needed
while lsof -Pi :$MLFLOW_PORT -sTCP:LISTEN -t >/dev/null ; do
    echo "Port $MLFLOW_PORT is already in use. Incrementing..."
    MLFLOW_PORT=$((MLFLOW_PORT + 1))
done

echo "Starting MLflow server on port $MLFLOW_PORT..."

# Start MLflow server with SQLite as backend store
mlflow server \
    --host 0.0.0.0 \
    --port $MLFLOW_PORT \
    --backend-store-uri sqlite:///mlflow_data/db/mlflow.db \
    --default-artifact-root ./mlflow_data/artifacts \
    --serve-artifacts

# Note: The server will run in the foreground. 
# To run it in the background, add '&' at the end of the mlflow command
# and capture the PID for potential shutdown later.