import json
import numpy as np
import triton_python_backend_utils as pb_utils
import traceback
import io
import os
import sys
import torch
import onnxruntime as ort
import logging

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [TRITON] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    def initialize(self, args):
        """初始化模型"""
        logger.info("Initializing FireRedASR pipeline model...")
        
        # 解析模型配置
        self.model_config = model_config = json.loads(args['model_config'])
        
        # 获取输出配置
        output0_config = pb_utils.get_output_config_by_name(model_config, "transcription")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])
        
        # 设置参数
        self.max_seq_len = 64  # 最大生成序列长度
        self.stop_token = [4]  # 结束token列表
        self.n_mels = 80  # Mel频谱特征数量
        self.sample_rate = 16000  # 采样率
        
        # 初始化特征提取器和tokenizer
        self.feature_extractor = None
        self.tokenizer = None
        
        # 获取当前目录路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fireredasr_path = os.path.join(current_dir, "FireRedASR") 
        
        try:
            # 尝试导入FireRedASR组件
            if fireredasr_path not in sys.path:
                sys.path.append(fireredasr_path)
                
            # 导入FireRedASR的特征提取和分词器
            from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
            from fireredasr.data.asr_feat import ASRFeatExtractor
            
            # 初始化tokenizer和特征提取器
            self.tokenizer = ChineseCharEnglishSpmTokenizer(
                os.path.join(current_dir, "dict.txt"), 
                os.path.join(current_dir, "train_bpe1000.model")
            )
            self.feature_extractor = ASRFeatExtractor(os.path.join(current_dir, "cmvn.ark"))
            logger.info("Successfully loaded FireRedASR feature extractor and tokenizer")
            
        except Exception as e:
            logger.error(f"Could not load FireRedASR components: {e}")
            logger.debug(traceback.format_exc())
            raise RuntimeError("Failed to load FireRedASR components")
        
        # 初始化ONNX Runtime会话
        try:
            # 设置ONNX Runtime提供程序（优先使用GPU）
            providers = []
            if ort.get_device() == 'GPU':
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            # 加载encoder模型
            logger.debug(f"Current directory: {current_dir}")
            encoder_model_path = os.path.join(current_dir, "FireRedASR_AED_L-Encoder-Batch.onnx")
            if not os.path.exists(encoder_model_path):
                raise FileNotFoundError(f"Encoder model not found: {encoder_model_path}")
            
            self.encoder_session = ort.InferenceSession(
                encoder_model_path, 
                providers=providers
            )
            logger.info(f"Loaded encoder model with providers: {self.encoder_session.get_providers()}")
            
            # 加载decoder模型
            decoder_model_path = os.path.join(current_dir, "FireRedASR_AED_L-Decoder-Batch.onnx")
            if not os.path.exists(decoder_model_path):
                raise FileNotFoundError(f"Decoder model not found: {decoder_model_path}")
            
            self.decoder_session = ort.InferenceSession(
                decoder_model_path, 
                providers=providers
            )
            logger.info(f"Loaded decoder model with providers: {self.decoder_session.get_providers()}")
            
            # 获取模型输入输出信息
            self.encoder_input_names = [input.name for input in self.encoder_session.get_inputs()]
            self.encoder_output_names = [output.name for output in self.encoder_session.get_outputs()]
            self.decoder_input_names = [input.name for input in self.decoder_session.get_inputs()]
            self.decoder_output_names = [output.name for output in self.decoder_session.get_outputs()]
            
            logger.debug(f"Encoder inputs: {self.encoder_input_names}")
            logger.debug(f"Encoder outputs: {self.encoder_output_names}")
            logger.debug(f"Decoder inputs: {self.decoder_input_names}")
            logger.debug(f"Decoder outputs: {self.decoder_output_names}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX models: {e}")
            logger.debug(traceback.format_exc())
            raise RuntimeError("Failed to load ONNX models")
        
        logger.info("FireRedASR pipeline model initialized successfully")

    def process_audio_data(self, batch_audio):
        """处理音频字节数据并提取特征"""
        try:
            audios, input_lengths, durs = self.feature_extractor(batch_audio)
                    
            batch_size = len(batch_audio)
            pad_zeros = torch.zeros(batch_size, 6, 80, dtype=torch.float32, device=audios.device)
            padded_input = torch.cat((audios, pad_zeros), dim=1)
            N, T = padded_input.size()[:2]
            padded_input = padded_input.cpu().numpy().astype(np.float32)
            input_lengths = np.array(input_lengths)
            
            # 创建mask
            mask = np.ones((N, 1, T), dtype=np.uint8)
            for i in range(N):
                mask[i, 0, input_lengths[i]:] = 0
            
            logger.debug(f"Extracted features shape: {padded_input.shape}, mask shape: {mask.shape}")
            return padded_input, mask
            
        except Exception as e:
            logger.error(f"FireRedASR feature extraction failed: {e}")
            logger.debug(traceback.format_exc())
            raise e

    def decode_tokens(self, tokens):
        """将token序列解码为文本"""
        try:
            if self.tokenizer is not None:
                # 使用FireRedASR tokenizer解码
                text = ""
                for token_id in tokens:
                    token = self.tokenizer.dict[int(token_id)]
                    if int(token_id) in self.stop_token:
                        break
                    text += token
                # 处理空格
                if hasattr(self.tokenizer, 'SPM_SPACE'):
                    text = text.replace(self.tokenizer.SPM_SPACE, ' ').strip()
                return text
            else:
                # 简单的token到文本映射（备用方案）
                logger.warning("Tokenizer not available, using fallback token mapping")
                return f"tokens_{tokens}"
        except Exception as e:
            logger.error(f"Error decoding tokens: {e}")
            return f"decode_error_{tokens}"

    def run_encoder(self, audio_features, mask):
        """运行encoder推理"""
        try:
            # 准备encoder输入
            encoder_inputs = {
                "audio": audio_features,
                "mask": mask
            }
            
            # 运行encoder推理
            encoder_outputs = self.encoder_session.run(
                self.encoder_output_names, 
                encoder_inputs
            )
            
            # 将输出转换为字典格式
            encoder_output_dict = {}
            for i, output_name in enumerate(self.encoder_output_names):
                encoder_output_dict[output_name] = encoder_outputs[i]
            
            logger.debug(f"Encoder inference completed, got {len(encoder_outputs)} outputs")
            return encoder_output_dict
            
        except Exception as e:
            logger.error(f"Encoder inference failed: {e}")
            logger.debug(traceback.format_exc())
            raise e

    def run_decoder_step(self, decoder_inputs):
        """运行单步decoder推理"""
        try:
            # 检查input_ids的维度，确保是2维
            if "input_ids" in decoder_inputs and len(decoder_inputs["input_ids"].shape) != 2:
                # 如果是3维，则降为2维
                if len(decoder_inputs["input_ids"].shape) == 3:
                    decoder_inputs["input_ids"] = decoder_inputs["input_ids"].reshape(
                        decoder_inputs["input_ids"].shape[0], 
                        decoder_inputs["input_ids"].shape[1]
                    )
                # 如果是1维，则升为2维
                elif len(decoder_inputs["input_ids"].shape) == 1:
                    decoder_inputs["input_ids"] = decoder_inputs["input_ids"].reshape(1, -1)
                
                logger.debug(f"Reshaped input_ids to shape: {decoder_inputs['input_ids'].shape}")
            
            # 运行decoder推理
            decoder_outputs = self.decoder_session.run(
                self.decoder_output_names,
                decoder_inputs
            )
            
            # 将输出转换为字典格式
            decoder_output_dict = {}
            for i, output_name in enumerate(self.decoder_output_names):
                decoder_output_dict[output_name] = decoder_outputs[i]
            
            return decoder_output_dict
            
        except Exception as e:
            logger.error(f"Decoder inference failed: {e}")
            logger.debug(traceback.format_exc())
            raise e

    def execute(self, requests):
        """执行推理请求"""
        responses = []
        for request in requests:
            transcription = ""
            
            try:
                # 获取输入tensor
                audio_tensor = pb_utils.get_input_tensor_by_name(request, "audio_data")
                if audio_tensor is None:
                    raise ValueError("Missing audio_data input")
                
                audio_data = audio_tensor.as_numpy().astype(np.int16)
                logger.debug(f"Audio data length: {len(audio_data)} bytes")

                # 1. 特征提取
                feats, mask = self.process_audio_data(audio_data)

                # 2. 运行encoder
                logger.debug("Running encoder inference...")
                encoder_outputs = self.run_encoder(feats, mask)

                # 3. 解码循环
                logger.debug("Starting decoding loop...")
                
                # 初始化解码器输入
                input_ids = np.array([[3]], dtype=np.int32)  # 保持为二维数组
                history_len = np.array([0], dtype=np.int64)  # 一维数组
                ids_len = np.array([1], dtype=np.int64)  # 一维数组
                attention_mask = np.array([1], dtype=np.int8)  # 一维数组
                
                # 初始化decoder的key-value缓存
                batch_size = 1
                num_layers = 16
                model_dtype = np.float32
                past_keys_B = np.zeros((batch_size, 20, 64, 0), dtype=model_dtype)
                past_values_B = np.zeros((batch_size, 20, 0, 64), dtype=model_dtype)
                
                # 准备decoder输入字典
                decoder_inputs = {
                    "input_ids": input_ids,
                    "history_len": history_len,
                    "ids_len": ids_len,
                    "attention_mask": attention_mask
                }
                
                # 添加past_keys和past_values
                for j in range(num_layers):
                    decoder_inputs[f"in_de_key_{j}"] = past_keys_B
                    decoder_inputs[f"in_de_value_{j}"] = past_values_B
                
                # 添加encoder输出
                for key, value in encoder_outputs.items():
                    decoder_inputs[key] = value
                
                # 初始化解码变量
                num_decode = 0
                save_token = []
                
                # 解码循环
                while num_decode < self.max_seq_len:
                    # 运行decoder推理
                    decoder_outputs = self.run_decoder_step(decoder_inputs)
                    
                    # 获取生成的token
                    max_logit_id = decoder_outputs["max_logit_id"][0]
                    num_decode += 1
                    
                    logger.debug(f"Step {num_decode}: Generated token {max_logit_id}")
                    
                    # 检查是否为结束token
                    if max_logit_id in self.stop_token:
                        logger.debug(f"Encountered stop token: {max_logit_id}")
                        break
                    
                    # 保存生成的token
                    save_token.append(max_logit_id)
                    
                    # 更新decoder输入
                    # 更新past_keys和past_values
                    for j in range(num_layers):
                        decoder_inputs[f"in_de_key_{j}"] = decoder_outputs[f"out_de_key_{j}"]
                        decoder_inputs[f"in_de_value_{j}"] = decoder_outputs[f"out_de_value_{j}"]
                    
                    # 更新input_ids为当前生成的token
                    decoder_inputs["input_ids"] = np.array([[max_logit_id]], dtype=np.int32)  # 保持为二维数组
                    
                    # 更新history_len
                    decoder_inputs["history_len"] = np.array([decoder_inputs["history_len"][0] + 1], dtype=np.int64)  # 一维数组
                    
                    # 更新ids_len，保持为1（每次只生成一个token）
                    decoder_inputs["ids_len"] = np.array([1], dtype=np.int64)  # 一维数组
                    
                    # 更新attention_mask，第一次解码后设为0
                    if num_decode >= 1:
                        decoder_inputs["attention_mask"] = np.array([0], dtype=np.int8)  # 一维数组

                logger.debug(f"Decoding completed. Generated tokens: {save_token}")

                # 4. 解码tokens为文本
                if save_token:
                    transcription = self.decode_tokens(save_token)
                else:
                    transcription = ""
                
                logger.info(f"Final transcription: {transcription}")
                
            except Exception as e:
                error_msg = f"Error during inference: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                transcription = error_msg
            
            # 创建输出tensor
            try:
                transcription_tensor = pb_utils.Tensor(
                    "transcription", 
                    np.array([transcription], dtype=self.output0_dtype)
                )
                response = pb_utils.InferenceResponse(output_tensors=[transcription_tensor])
                responses.append(response)
                
            except Exception as e:
                error_msg = f"Error creating response: {str(e)}"
                logger.error(error_msg)
                # 创建错误响应
                error_tensor = pb_utils.Tensor(
                    "transcription", 
                    np.array([error_msg], dtype=self.output0_dtype)
                )
                error_response = pb_utils.InferenceResponse(output_tensors=[error_tensor])
                responses.append(error_response)
        
        return responses

    def finalize(self):
        """清理资源"""
        logger.info("Finalizing FireRedASR pipeline model...")
        
        # 清理ONNX Runtime会话
        if hasattr(self, 'encoder_session'):
            del self.encoder_session
        if hasattr(self, 'decoder_session'):
            del self.decoder_session
        
        # 清理其他资源
        self.feature_extractor = None
        self.tokenizer = None
        
        logger.info("FireRedASR pipeline model finalized")

