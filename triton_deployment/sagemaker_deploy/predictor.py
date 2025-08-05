# predictor.py

import os
import json
import base64
import tempfile
import traceback
import time
import threading
import multiprocessing as mp
from multiprocessing import Process, Queue, Pool, Manager, Value, Lock
import pickle
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Union
import tritonclient.grpc as grpcclient
import numpy as np
import torch
import torchaudio
from io import BytesIO
import asyncio
from concurrent.futures import ProcessPoolExecutor
import queue

app = FastAPI(title="FireRedASR SageMaker Endpoint", version="1.0.0")

# Triton 客户端配置
TRITON_URL = "127.0.0.1:8001"
MODEL_NAME = "fireredasr_onnx"
SAMPLE_RATE = 16000
PADDING_DURATION = 4  # 4 seconds

class AudioRequest(BaseModel):
    audio: str  # base64 encoded audio data

class PredictionResponse(BaseModel):
    transcription: Union[str, list]
    status: str

class ErrorResponse(BaseModel):
    error: str
    status: str
    
class HealthResponse(BaseModel):
    status: str
    error: Optional[str] = None


def create_triton_client():
    """创建单个 Triton 客户端"""
    try:
        client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)
        if client.is_server_live() and client.is_model_ready(MODEL_NAME):
            return client
        else:
            raise Exception("Client not ready")
    except Exception as e:
        print(f"Failed to create client: {e}")
        raise


def preprocess_audio(waveform):
    """预处理音频数据，包括格式转换和padding"""
    try:
        sample_rate = SAMPLE_RATE
        
        try:
            duration = int(len(waveform) / sample_rate)
        except Exception as e:
            print(f"Error calculating duration: {e}")
            print(f"Length of waveform: {len(waveform)}")
            duration = 0
        
        print(f"Original audio duration: {duration} seconds, length: {len(waveform)}")
        
        # Padding to nearest PADDING_DURATION seconds
        padding_duration = PADDING_DURATION
        target_length = padding_duration * sample_rate * ((duration // padding_duration) + 1)
        
        # 创建零填充的样本数组
        samples = np.zeros((1, target_length), dtype=np.int16)
        samples[0, :len(waveform)] = waveform
        
        print(f"Padded audio shape: {samples.shape}, target_length: {target_length}")
        
        return samples
        
    except Exception as e:
        print(f"Error in audio preprocessing: {e}")
        raise


def worker_process_predict(audio_data):
    """工作进程中的预测函数"""
    client = None
    try:
        # 每个工作进程创建自己的客户端
        client = create_triton_client()
        
        waveform = np.frombuffer(audio_data, dtype=np.int16)
        # 预处理音频数据（包括padding）
        processed_audio = preprocess_audio(waveform)
        
        # 准备输入
        inputs = [
            grpcclient.InferInput(
                "audio_data", 
                processed_audio.shape, 
                "INT16"
            ),
            grpcclient.InferInput(
            "wav_length", [1, 1], "INT32"  # Changed shape to [1, 1]
        ),
        ]
        
        # 设置输入数据
        inputs[0].set_data_from_numpy(processed_audio)
        inputs[1].set_data_from_numpy(np.array([[len(waveform)]], dtype=np.int32))
        # 准备输出
        outputs = [grpcclient.InferRequestedOutput("transcription")]
        
        # print(f"Process {os.getpid()}: Sending inference request with audio shape: {processed_audio.shape}")
        
        # 执行推理
        response = client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs
        )
        
        # 处理结果
        transcription = response.as_numpy("transcription")
        if isinstance(transcription, np.ndarray):
            if transcription.dtype.kind in ['U', 'S']:
                result = transcription.item() if transcription.size == 1 else transcription.tolist()
            else:
                result = transcription.tolist()
        else:
            result = str(transcription)
        
        # print(f"Process {os.getpid()}: Transcription result: {result}")
        
        return {
            "transcription": result,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Process {os.getpid()}: Prediction error: {str(e)}")
        print(traceback.format_exc())
        return {
            "error": str(e),
            "status": "error"
        }
    finally:
        if client:
            try:
                client.close()
            except:
                pass


class AsyncTritonPredictor:
    def __init__(self, max_workers=4):
        """
        初始化多进程预测器
        
        Args:
            max_workers: 最大工作进程数
        """
        self.max_workers = max_workers
        self.executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context('spawn')  # 使用 spawn 方式创建进程
        )
        self.initialized = True
        print(f"Initialized AsyncTritonPredictor with {max_workers} worker processes")
    
    async def predict(self, audio_data):
        """异步预测方法"""
        try:
            # 在进程池中执行预测
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                worker_process_predict,
                audio_data
            )
            return result
        except Exception as e:
            print(f"Async prediction error: {str(e)}")
            print(traceback.format_exc())
            return {
                "error": str(e),
                "status": "error"
            }
    
    def cleanup(self):
        """清理资源"""
        try:
            print("Shutting down process pool...")
            # 关闭进程池
            self.executor.shutdown(wait=True)
            print("Process pool shutdown completed")
        except Exception as e:
            print(f"Cleanup error: {e}")

    def health_check(self):
        """健康检查方法"""
        try:
            # 创建测试客户端检查连接
            test_client = create_triton_client()
            is_healthy = test_client.is_server_live() and test_client.is_model_ready(MODEL_NAME)
            test_client.close()
            return is_healthy
        except Exception as e:
            print(f"Health check error: {e}")
            return False


# 全局预测器实例
_predictor_instance = None
_predictor_lock = threading.Lock()

def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        with _predictor_lock:
            if _predictor_instance is None:
                # 根据CPU核心数设置工作进程数
                max_workers = min(mp.cpu_count(), 8)  # 最多8个进程
                _predictor_instance = AsyncTritonPredictor(max_workers=max_workers)
                print(f"Created predictor with {max_workers} worker processes")
    return _predictor_instance

def cleanup_predictor():
    """清理预测器实例"""
    global _predictor_instance
    if _predictor_instance is not None:
        with _predictor_lock:
            if _predictor_instance is not None:
                _predictor_instance.cleanup()
                _predictor_instance = None

@app.post("/invocations", response_model=Union[PredictionResponse, ErrorResponse])
async def invocations(request: Request):
    try:
        predictor = get_predictor()

        content_type = request.headers.get("content-type", "")

        if content_type == 'application/octet-stream':
            raw_audio_bytes = await request.body()
            audio_data = raw_audio_bytes  # 直接传递字节数据
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported content type: {content_type}")

        # 异步执行推理
        result = await predictor.predict(audio_data)

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result)

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Invocation error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(error=str(e), status="error").dict()
        )
        
@app.get("/ping", response_model=HealthResponse)
async def ping():
    """健康检查端点"""
    try:
        predictor = get_predictor()

        # 检查预测器状态
        if predictor.initialized:
            try:
                # 在单独的线程中执行健康检查，避免阻塞
                loop = asyncio.get_event_loop()
                is_healthy = await loop.run_in_executor(
                    None,  # 使用默认线程池
                    predictor.health_check
                )
                
                if is_healthy:
                    return HealthResponse(status="healthy")
                else:
                    return HealthResponse(status="unhealthy", error="Triton server not accessible")
                    
            except Exception as e:
                return HealthResponse(status="unhealthy", error=f"Health check failed: {str(e)}")
        else:
            return HealthResponse(status="unhealthy", error="Predictor not initialized")

    except Exception as e:
        return HealthResponse(status="unhealthy", error=str(e))

    
@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    print("FastAPI application starting up...")
    print(f"Main process PID: {os.getpid()}")
    print(f"Available CPU cores: {mp.cpu_count()}")
    
    # 预热预测器（可选）
    try:
        predictor = get_predictor()
        print(f"Predictor initialization status: {predictor.initialized}")
        print(f"Worker processes: {predictor.max_workers}")
    except Exception as e:
        print(f"Warning: Failed to initialize predictor during startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    print("FastAPI application shutting down...")
    
    try:
        # 清理预测器资源
        cleanup_predictor()
        print("Predictor cleanup completed")
        
        # 清理 PyTorch 缓存（如果使用了 GPU）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("PyTorch CUDA cache cleared")
            
        print("Application shutdown completed successfully")
        
    except Exception as e:
        print(f"Error during application shutdown: {str(e)}")
        print(traceback.format_exc())


# 为了支持多进程，需要添加主程序保护
if __name__ == "__main__":
    import uvicorn
    
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        workers=1  # FastAPI 应用本身使用单进程，内部使用进程池
    )

