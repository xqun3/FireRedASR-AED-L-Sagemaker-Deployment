# predictor.py

import os
import json
import base64
import tempfile
import traceback
import time
import threading
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Union
import tritonclient.grpc as grpcclient
import numpy as np
import torch
import torchaudio
from io import BytesIO

app = FastAPI(title="FireRedASR SageMaker Endpoint", version="1.0.0")

# Triton 客户端配置
TRITON_URL = "127.0.0.1:8001"
MODEL_NAME = "fireredasr_onnx"

# Pydantic 模型定义
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

class TritonPredictor:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TritonPredictor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.triton_client = None
            self.model_name = MODEL_NAME
            self.initialized = False
            self._initialize_client()

    def _initialize_client(self):
        """初始化 Triton 客户端，带重试机制"""
        try:
            self.triton_client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)

            # 检查服务器健康状态
            if not self.triton_client.is_server_live():
                raise Exception("Triton server is not live")

            if not self.triton_client.is_server_ready():
                raise Exception("Triton server is not ready")

            # 检查模型状态
            if not self.triton_client.is_model_ready(self.model_name):
                raise Exception(f"Model {self.model_name} is not ready")

            print(f"Triton client initialized successfully for model: {self.model_name}")
            self.initialized = True
            return

        except Exception as e:
            raise Exception(f"Failed to initialize Triton client {str(e)}")

    def _ensure_client_ready(self):
        """确保客户端准备就绪"""
        if not self.initialized or self.triton_client is None:
            self._initialize_client()

        # 再次检查连接状态
        try:
            if not self.triton_client.is_server_live():
                print("Triton server connection lost, reinitializing...")
                self._initialize_client()
        except Exception as e:
            print(f"Error checking server status: {e}")
            self._initialize_client()

    def predict(self, audio_data):
        """执行推理"""
        try:
            # 处理音频数据
            if isinstance(audio_data, str):
                # 如果是base64编码的字符串
                try:
                    audio_bytes = base64.b64decode(audio_data)
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                except Exception as e:
                    return {
                        "error": f"Failed to decode base64 audio data: {str(e)}",
                        "status": "error"
                    }
            elif isinstance(audio_data, bytes):
                # 如果是原始字节数据
                audio_data = np.frombuffer(audio_data, dtype=np.int16)
            elif not isinstance(audio_data, np.ndarray):
                return {
                    "error": "Unsupported audio data format",
                    "status": "error"
                }

            # 确保音频数据形状正确
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(1, -1)

            # 准备输入数据
            audio_input = grpcclient.InferInput("audio_data", audio_data.shape, "INT16")
            audio_input.set_data_from_numpy(audio_data)

            # 准备输出
            output = grpcclient.InferRequestedOutput("transcription")

            # 执行推理
            response = self.triton_client.infer(
                model_name=self.model_name,
                inputs=[audio_input],
                outputs=[output]
            )

            # 获取结果
            transcription = response.as_numpy("transcription")

            # 处理结果（根据你的模型输出格式调整）
            if isinstance(transcription, np.ndarray):
                if transcription.dtype.kind in ['U', 'S']:  # Unicode 或字节字符串
                    result = transcription.item() if transcription.size == 1 else transcription.tolist()
                else:
                    result = transcription.tolist()
            else:
                result = str(transcription)

            return {
                "transcription": result,
                "status": "success"
            }

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            print(traceback.format_exc())
            return {
                "error": str(e),
                "status": "error"
            }

    def cleanup(self):
        """清理资源"""
        try:
            if self.triton_client is not None:
                print("Closing Triton client connection...")
                # 关闭 Triton 客户端连接
                self.triton_client.close()
                self.triton_client = None
                print("Triton client connection closed successfully")
            
            self.initialized = False
            print("TritonPredictor cleanup completed")
            
        except Exception as e:
            print(f"Error during TritonPredictor cleanup: {str(e)}")

    def __del__(self):
        """析构函数，确保资源被释放"""
        self.cleanup()

# 全局预测器实例
_predictor_instance = None

def get_predictor():
    """获取预测器实例（延迟初始化）"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = TritonPredictor()
    return _predictor_instance

def cleanup_predictor():
    """清理预测器实例"""
    global _predictor_instance
    if _predictor_instance is not None:
        _predictor_instance.cleanup()
        _predictor_instance = None

@app.get("/ping", response_model=HealthResponse)
async def ping():
    """健康检查端点"""
    try:
        predictor = get_predictor()

        # 检查 Triton 服务器状态
        if predictor.initialized and predictor.triton_client:
            try:
                if predictor.triton_client.is_server_live():
                    return HealthResponse(status="healthy")
                else:
                    return HealthResponse(status="unhealthy", error="Triton server not live")
            except Exception as e:
                return HealthResponse(status="unhealthy", error=f"Triton server check failed: {str(e)}")
        else:
            return HealthResponse(status="unhealthy", error="Triton client not initialized")

    except Exception as e:
        return HealthResponse(status="unhealthy", error=str(e))

@app.post("/invocations", response_model=Union[PredictionResponse, ErrorResponse])
async def invocations(request: Request):
    """SageMaker 推理端点"""
    try:
        # 获取预测器实例
        predictor = get_predictor()

        # 获取请求数据
        content_type = request.headers.get("content-type", "")

        if content_type == 'application/octet-stream':
            # 直接音频数据 - FIX: await the coroutine
            raw_audio_bytes = await request.body()
            audio_data = np.frombuffer(raw_audio_bytes, dtype=np.int16)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported content type: {content_type}")

        # 执行推理
        result = predictor.predict(audio_data)

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


# 启动事件处理
@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    print("FastAPI application starting up...")
    # 预热预测器（可选）
    try:
        predictor = get_predictor()
        print(f"Predictor initialization status: {predictor.initialized}")
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
            
        # 清理临时文件（如果有的话）
        # 这里可以添加其他需要清理的资源
        
        print("Application shutdown completed successfully")
        
    except Exception as e:
        print(f"Error during application shutdown: {str(e)}")
        print(traceback.format_exc())
