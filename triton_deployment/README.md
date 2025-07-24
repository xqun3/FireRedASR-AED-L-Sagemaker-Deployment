# FireRedASR Triton Deployment on SageMaker

## 项目概述

FireRedASR 是一个自动语音识别（ASR）模型，本项目将其转换为 ONNX 格式并部署到 AWS SageMaker 上，通过 Triton Inference Server 提供推理服务。

## 目录结构

- `deploy_and_test.ipynb`: 部署和测试模型的 Jupyter 笔记本
- `Dockerfile.sagemaker`: 用于构建 SageMaker 容器镜像的 Dockerfile
- `sagemaker_deploy/`: 包含 SageMaker 部署所需的脚本和配置
- `model_repository/`: 存储模型文件的目录

## 部署流程

1. **模型准备**
   - 将模型转换为 ONNX 格式
   - 上传模型到 S3 存储桶

2. **构建和推送 Docker 镜像**
   - 使用 `build_and_push.sh` 脚本构建镜像并推送到 ECR
  
## 使用方法

1. 打开 `deploy_and_test.ipynb` 笔记本
2. 按照笔记本中的步骤执行部署流程
3. 使用提供的函数测试部署的模型
 - 使用 SSH-helper 进行部署调试（可选）

## 推理示例

```python
import boto3
import json
import kaldiio

# 准备音频数据
def prepare_audio(audio_file, target_sr=16000):
    _, wav_np = kaldiio.load_mat(audio_file)    
    return wav_np.tobytes()

# 调用端点进行推理
def transcribe_audio(audio_path, endpoint_name):
    audio_data = prepare_audio(audio_path)
    runtime_client = boto3.client('sagemaker-runtime')
    
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/octet-stream',
        Body=audio_data,
    )
    
    result = json.loads(response['Body'].read().decode())
    return result

# 使用示例
audio_path = "your_audio_file.wav"
endpoint_name = "your-endpoint-name"
result = transcribe_audio(audio_path, endpoint_name)
print(json.dumps(result, indent=2, ensure_ascii=False))
```

## 清理资源

使用以下命令删除不再需要的 SageMaker 资源：

```python
sm = boto3.client('sagemaker')
sm.delete_endpoint(EndpointName=endpoint_name)
sm.delete_endpoint_config(EndpointConfigName=endpoint_name)
sm.delete_model(ModelName=endpoint_name)
```

## 环境要求

- AWS 账户和适当的权限
- SageMaker Studio 或具有 AWS CLI 配置的环境
- Python 依赖：boto3, sagemaker, kaldiio
