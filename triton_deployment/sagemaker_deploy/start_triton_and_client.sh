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
timeout=15
triton_started=false

echo "Waiting for Triton gRPC server to be ready..."

while [ $timeout -gt 0 ]; do
    if ! kill -0 $triton_pid 2>/dev/null; then
        echo "Error: Triton server process died"
        cat triton.log
        exit 1
    fi
    
    # 使用 grpc_health_probe 或 nc 检查端口
    if nc -z localhost 8001 2>/dev/null; then
        echo "Triton gRPC server is ready"
        triton_ready=true
        break
    fi
    
    echo "Waiting for Triton gRPC server... ($timeout seconds remaining)"
    sleep 1
    ((timeout--))
done

if [ "$triton_ready" = false ]; then
    echo "Error: Triton gRPC server failed to start within 60 seconds"
    echo "Triton server logs:"
    cat triton.log
    exit 1
fi

# Start the main application
echo "Starting uvicorn server..."
uvicorn predictor:app --host 0.0.0.0 --port 8080 --workers 1 || {
    echo "Error: uvicorn server failed to start"
    exit 1
}
