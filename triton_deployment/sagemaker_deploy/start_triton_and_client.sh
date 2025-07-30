#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# download model from s3
python3 download_model_from_s3.py --source_s3_url ${model_s3_url} --local_dir_path "model_repository/" --working_dir "/opt/program" || { echo "S3 sync failed"; exit 1; }
# aws s3 sync ${model_s3_url} /opt/program/model_repository/

model_repo_path=/opt/program/model_repository

tritonserver --model-repository $model_repo_path \
    --disable-auto-complete-config --grpc-port=8001 > triton.log 2>&1 &

triton_pid=$!

cat triton.log
# Wait for tritonserver to start and check if it's running
timeout=30
triton_started=false

while [ $timeout -gt 0 ]; do
    if kill -0 $triton_pid 2>/dev/null; then
        echo "Triton server started successfully"
        triton_started=true
        break
    fi
    sleep 1
    ((timeout--))
done

if [ "$triton_started" = false ]; then
    echo "Error: Triton server failed to start within 30 seconds"
    echo "Triton server logs:"
    cat triton.log
    exit 1
fi

# Start the main application
echo "Starting uvicorn server..."
uvicorn predictor:app --host 0.0.0.0 --port 8080 --workers 16 || {
    echo "Error: uvicorn server failed to start"
    exit 1
}
