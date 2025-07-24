#!/usr/bin/env python3

import requests
import argparse
import kaldiio
import time
from pathlib import Path

def prepare_audio(audio_file, target_sr=16000):
    _, wav_np = kaldiio.load_mat(audio_file)    
    return wav_np.tobytes()

class LocalEndpointTester:
    def __init__(self, base_url='http://localhost:8080'):
        self.base_url = base_url
    
    def test_health(self):
        """测试健康检查端点"""
        try:
            response = requests.get(f"{self.base_url}/ping")
            print(f"Health check status: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {str(e)}")
            return False
    
    def test_inference(self, audio_file_path):
        """测试推理端点 - 二进制格式"""
        # audio_path = Path(audio_file_path)
        
        # if not audio_path.exists():
        #     raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        try:
            
            audio_data = prepare_audio(audio_file_path)
            
            print(f"Sending audio file: {audio_file_path} ({len(audio_data)} bytes)")
            
            response = requests.post(
                f"{self.base_url}/invocations",
                data=audio_data,
                headers={"Content-Type": "application/octet-stream"}
            )
            
            print(f"Inference status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success!")
                print(f"Transcription: {result.get('transcription', 'N/A')}")
                print(f"Status: {result.get('status', 'N/A')}")
                return result
            else:
                print(f"❌ Error response ({response.status_code}): {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Inference failed: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Test local FastAPI endpoint with binary audio')
    parser.add_argument('--url', default='http://localhost:8080', help='Base URL of the endpoint')
    parser.add_argument('--audio-file', '-f', required=True, help='Audio file path for testing')
    
    args = parser.parse_args()
    
    tester = LocalEndpointTester(args.url)
    
    print(f"🚀 Testing FastAPI endpoint: {args.url}")
    print(f"🎵 Audio file: {args.audio_file}")
    
    # 健康检查
    print("\n" + "="*60)
    print("🔍 Testing health endpoint...")
    if tester.test_health():
        print("✅ Health check passed - server is ready")
    else:
        print("❌ Health check failed - server may not be ready")
        print("💡 Make sure:")
        print("   - FastAPI server is running: uvicorn predictor:app --host 0.0.0.0 --port 8080")
        print("   - Triton server is running and accessible")
        print("   - Model is loaded in Triton")
        return
    
    # 音频推理测试
    print("\n" + "="*60)
    print("📦 Testing binary inference endpoint (application/octet-stream)...")
    
    st = time.time()
    result = tester.test_inference(args.audio_file)
    print(f"time cost: {time.time()-st}")

if __name__ == '__main__':
    main()
