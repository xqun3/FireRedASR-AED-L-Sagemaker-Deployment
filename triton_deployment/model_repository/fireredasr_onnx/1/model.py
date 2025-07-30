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
            batch_size = len(batch_audio)
            audios, input_lengths, durs = self.feature_extractor(batch_audio)
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

    def run_batch_decoder(self, encoder_outputs, batch_size):
        """批量解码器"""
        try:
            # 初始化批量输入
            input_ids = np.array([[3] for _ in range(batch_size)], dtype=np.int32)  # [B, 1]
            history_len = np.zeros(batch_size, dtype=np.int64)  # [B]
            ids_len = np.ones(batch_size, dtype=np.int64)  # [B]
            attention_mask = np.ones(batch_size, dtype=np.int8)  # [B]

            # 初始化分层KV缓存 - 关键修复点1
            num_layers = 16
            past_keys = {}
            past_values = {}
            for j in range(num_layers):
                past_keys[j] = np.zeros((batch_size, 20, 64, 0), dtype=np.float32)
                past_values[j] = np.zeros((batch_size, 20, 0, 64), dtype=np.float32)

            batch_tokens = [[] for _ in range(batch_size)]
            finished = [False] * batch_size

            logger.debug(f"Starting batch decoding for {batch_size} samples")

            # 批量解码循环
            for step in range(self.max_seq_len):
                if all(finished):
                    logger.debug(f"All samples finished at step {step}")
                    break

                logger.debug(f"Decoding step {step}, input_ids shape: {input_ids.shape}")

                # 准备批量输入
                decoder_inputs = {
                    "input_ids": input_ids,
                    "history_len": history_len,
                    "ids_len": ids_len,
                    "attention_mask": attention_mask
                }

                # 添加分层KV缓存 - 关键修复点2
                for j in range(num_layers):
                    decoder_inputs[f"in_de_key_{j}"] = past_keys[j]
                    decoder_inputs[f"in_de_value_{j}"] = past_values[j]

                # 添加encoder输出
                for key, value in encoder_outputs.items():
                    decoder_inputs[key] = value

                # 批量推理
                decoder_outputs = self.run_decoder_step(decoder_inputs)

                # 处理批量输出
                max_logit_ids = decoder_outputs["max_logit_id"]  # [B]
                if len(max_logit_ids.shape) > 1:
                    max_logit_ids = max_logit_ids.flatten()  # 确保是1维

                logger.debug(f"Generated tokens at step {step}: {max_logit_ids}")

                # 更新每个样本的状态 - 关键修复点3
                active_samples = []  # 仍在生成的样本索引
                for i in range(batch_size):
                    if not finished[i]:
                        token = int(max_logit_ids[i])
                        if token in self.stop_token:
                            finished[i] = True
                            logger.debug(f"Sample {i} finished with stop token {token}")
                        else:
                            batch_tokens[i].append(token)
                            active_samples.append(i)

                # 如果所有样本都完成，提前退出
                if not active_samples:
                    logger.debug("All samples finished, breaking early")
                    break

                # 更新批量状态 - 关键修复点4
                # 为下一步准备input_ids（只包含未完成的样本的新token）
                next_input_ids = []
                for i in range(batch_size):
                    if finished[i]:
                        # 已完成的样本，使用padding token或保持当前token
                        next_input_ids.append([max_logit_ids[i]])
                    else:
                        # 未完成的样本，使用新生成的token
                        next_input_ids.append([max_logit_ids[i]])

                input_ids = np.array(next_input_ids, dtype=np.int32)  # [B, 1]

                # 更新历史长度（只对未完成的样本）
                for i in range(batch_size):
                    if not finished[i]:
                        history_len[i] += 1

                # 更新ids_len（保持为1，因为每次只生成一个token）
                ids_len = np.ones(batch_size, dtype=np.int64)

                # 更新attention_mask - 关键修复点5
                # 第一步之后，对于继续生成的样本，attention_mask应该设为0
                if step == 0:
                    attention_mask = np.zeros(batch_size, dtype=np.int8)
                # 后续步骤保持为0

                # 更新分层KV缓存 - 关键修复点6
                for j in range(num_layers):
                    past_keys[j] = decoder_outputs[f"out_de_key_{j}"]
                    past_values[j] = decoder_outputs[f"out_de_value_{j}"]

                logger.debug(f"Step {step} completed, active samples: {len(active_samples)}")

            # 批量解码为文本
            transcriptions = []
            for i, tokens in enumerate(batch_tokens):
                if tokens:
                    transcription = self.decode_tokens(tokens)
                    logger.debug(f"Sample {i} tokens: {tokens[:10]}... -> {transcription[:50]}...")
                else:
                    transcription = ""
                    logger.warning(f"Sample {i} generated no tokens")
                transcriptions.append(transcription)

            logger.info(f"Batch decoding completed, generated {len(transcriptions)} transcriptions")
            return transcriptions

        except Exception as e:
            logger.error(f"Batch decoder failed: {e}")
            logger.debug(traceback.format_exc())
            # 返回错误信息
            return [f"decode_error: {str(e)}"] * batch_size


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
        """批处理版本"""
        responses = []
        try:
            # 1. 收集所有音频数据
            batch_audio_data = []
            for request in requests:
                audio_tensor = pb_utils.get_input_tensor_by_name(request, "audio_data")
                audio_data = audio_tensor.as_numpy().astype(np.int16).squeeze()
                batch_audio_data.append(audio_data)

            batch_size = len(batch_audio_data)
            logger.info(f"Processing batch of {batch_size} requests")

            # 2. 批量特征提取
            feats, mask = self.process_audio_data(batch_audio_data)

            # 3. 批量encoder推理
            encoder_outputs = self.run_encoder(feats, mask)

            # 4. 批量decoder推理
            batch_transcriptions = self.run_batch_decoder(encoder_outputs, batch_size)

            # 5. 创建响应
            
            for transcription in batch_transcriptions:
                transcription_tensor = pb_utils.Tensor(
                    "transcription", 
                    np.array([transcription], dtype=self.output0_dtype)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[transcription_tensor]))

            logger.info(f"Batch processing completed successfully")

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            logger.debug(traceback.format_exc())
            # 为所有请求返回错误信息
            for _ in requests:
                error_tensor = pb_utils.Tensor(
                    "transcription", 
                    np.array(["transcription error"], dtype=self.output0_dtype)
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
