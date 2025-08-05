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
import time

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
        self.max_seq_len = 64
        self.stop_token = [4]
        self.n_mels = 80
        self.sample_rate = 16000
        self.device_id = 0
        
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
        
        # 初始化ONNX Runtime会话 - 优化版本
        try:
            # 优化的会话选项
            session_opts = ort.SessionOptions()
            session_opts.log_severity_level = 4
            session_opts.log_verbosity_level = 4
            session_opts.inter_op_num_threads = 4
            session_opts.intra_op_num_threads = 4
            session_opts.enable_cpu_mem_arena = True
            session_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
            session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
            session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
            session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
            session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
            session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
            session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
            session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
            session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
            
            # 设置提供程序
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.device_type = 'cuda' if 'CUDAExecutionProvider' in providers else 'cpu'
            
            # 加载encoder模型
            encoder_model_path = os.path.join(current_dir, "FireRedASR_AED_L-Encoder-Batch.onnx")
            if not os.path.exists(encoder_model_path):
                raise FileNotFoundError(f"Encoder model not found: {encoder_model_path}")
            
            self.encoder_session = ort.InferenceSession(
                encoder_model_path, 
                sess_options=session_opts,
                providers=providers
            )
            
            # 获取encoder输入输出信息
            in_name_A = self.encoder_session.get_inputs()
            out_name_A = self.encoder_session.get_outputs()
            self.encoder_input_names = [in_name_A[0].name, in_name_A[1].name]
            self.encoder_output_names = [out.name for out in out_name_A]
            
            logger.info(f"Loaded encoder model with providers: {self.encoder_session.get_providers()}")
            
            # 加载decoder模型
            decoder_model_path = os.path.join(current_dir, "FireRedASR_AED_L-Decoder-Batch.onnx")
            if not os.path.exists(decoder_model_path):
                raise FileNotFoundError(f"Decoder model not found: {decoder_model_path}")
            
            self.decoder_session = ort.InferenceSession(
                decoder_model_path, 
                sess_options=session_opts,
                providers=providers
            )
            
            # 获取decoder模型信息
            self.in_name_B = self.decoder_session.get_inputs()
            self.out_name_B = self.decoder_session.get_outputs()
            self.decoder_input_names = [inp.name for inp in self.in_name_B]
            self.decoder_output_names = [out.name for out in self.out_name_B]
            
            # 获取模型数据类型
            model_dtype = self.decoder_session._inputs_meta[0].type
            if 'float16' in model_dtype:
                self.model_dtype = np.float16
            else:
                self.model_dtype = np.float32
            
            # 计算层数和索引
            self.amount_of_outputs = len(self.out_name_B)
            self.generate_limit = self.max_seq_len - 1
            self.num_layers = (self.amount_of_outputs - 2) // 2
            self.num_layers_2 = self.num_layers + self.num_layers
            self.num_layers_4 = self.num_layers_2 + self.num_layers_2
            self.num_layers_2_plus_1 = self.num_layers_2 + 1
            self.num_layers_2_plus_2 = self.num_layers_2 + 2
            self.layer_indices = np.arange(self.num_layers_2, self.num_layers_4, dtype=np.int32) + 3
            
            logger.info(f"Loaded decoder model with providers: {self.decoder_session.get_providers()}")
            logger.debug(f"Model dtype: {self.model_dtype}, num_layers: {self.num_layers}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX models: {e}")
            logger.debug(traceback.format_exc())
            raise RuntimeError("Failed to load ONNX models")
        
        logger.info("FireRedASR pipeline model initialized successfully")

    def process_audio_data(self, batch_audio):
        """处理音频字节数据并提取特征 - 优化版本"""
        start_time = time.perf_counter()
        try:
            batch_size = len(batch_audio)
            logger.debug(f"Processing batch_size: {batch_size}")
            
            # 特征提取
            feature_start = time.perf_counter()
            audios, input_lengths, durs = self.feature_extractor(batch_audio)
            feature_time = time.perf_counter() - feature_start
            logger.debug(f"[TIMING] Feature extraction: {feature_time:.4f}s")
            logger.debug(f"input_lengths: {input_lengths}")
            
            # 使用torch进行padding，然后转换为numpy（减少数据拷贝）
            padding_start = time.perf_counter()
            pad_zeros = torch.zeros(batch_size, 6, 80, dtype=torch.float32, device=audios.device)
            padded_input = torch.cat((audios, pad_zeros), dim=1)
            N, T = padded_input.size()[:2]
            
            # 直接转换为numpy，避免额外的类型转换
            padded_input = padded_input.cpu().numpy().astype(self.model_dtype)
            input_lengths = np.array(input_lengths, dtype=np.int32)
            padding_time = time.perf_counter() - padding_start
            logger.debug(f"[TIMING] Padding and conversion: {padding_time:.4f}s")
            
            # 矢量化创建mask - 优化点1：避免for循环
            mask_start = time.perf_counter()
            mask = np.ones((N, 1, T))
            for i in range(N):
                mask[i, 0, input_lengths[i]:] = 0
            mask = mask.astype(np.uint8)
            mask_time = time.perf_counter() - mask_start
            logger.debug(f"[TIMING] Mask creation: {mask_time:.4f}s")
            
            total_time = time.perf_counter() - start_time
            logger.debug(f"[TIMING] Total audio processing: {total_time:.4f}s")
            logger.debug(f"Extracted features shape: {padded_input.shape}, mask shape: {mask.shape}")
            return padded_input, mask, input_lengths
            
        except Exception as e:
            logger.error(f"FireRedASR feature extraction failed: {e}")
            logger.debug(traceback.format_exc())
            raise e

    def decode_tokens_batch(self, batch_tokens):
        """批量解码tokens - 优化版本"""
        start_time = time.perf_counter()
        try:
            transcriptions = []
            for i, tokens in enumerate(batch_tokens):
                token_start = time.perf_counter()
                if not tokens:
                    transcriptions.append("")
                    continue
                    
                text = ""
                for token_id in tokens:
                    if isinstance(token_id, (list, np.ndarray)):
                        token_id = token_id[0] if len(token_id) > 0 else 0
                    
                    token_id = int(token_id)
                    if token_id in self.stop_token:
                        break
                    
                    if token_id < len(self.tokenizer.dict):
                        token = self.tokenizer.dict[token_id]
                        text += token
                
                # 处理空格
                if hasattr(self.tokenizer, 'SPM_SPACE'):
                    text = text.replace(self.tokenizer.SPM_SPACE, ' ').strip()
                transcriptions.append(text)
                
                token_time = time.perf_counter() - token_start
                logger.debug(f"[TIMING] Token decoding sample {i}: {token_time:.4f}s")
            
            total_time = time.perf_counter() - start_time
            logger.debug(f"[TIMING] Total token decoding: {total_time:.4f}s")
            return transcriptions
            
        except Exception as e:
            logger.error(f"Error in batch token decoding: {e}")
            return [f"decode_error_{i}" for i in range(len(batch_tokens))]

    def run_encoder_optimized(self, audio_features, mask):
        """运行encoder推理 - 优化版本，使用OrtValue"""
        start_time = time.perf_counter()
        try:
            # 使用OrtValue减少内存拷贝
            ortvalue_start = time.perf_counter()
            audio_ortvalue = ort.OrtValue.ortvalue_from_numpy(
                audio_features, self.device_type, self.device_id
            )
            mask_ortvalue = ort.OrtValue.ortvalue_from_numpy(
                mask, self.device_type, self.device_id
            )
            ortvalue_time = time.perf_counter() - ortvalue_start
            logger.debug(f"[TIMING] Encoder OrtValue creation: {ortvalue_time:.4f}s")

            # 运行encoder推理
            inference_start = time.perf_counter()
            encoder_outputs = self.encoder_session.run_with_ort_values(
                self.encoder_output_names,
                {
                    self.encoder_input_names[0]: audio_ortvalue,
                    self.encoder_input_names[1]: mask_ortvalue
                }
            )
            inference_time = time.perf_counter() - inference_start
            logger.debug(f"[TIMING] Encoder inference: {inference_time:.4f}s")

            total_time = time.perf_counter() - start_time
            logger.debug(f"[TIMING] Total encoder processing: {total_time:.4f}s")
            logger.debug(f"Encoder inference completed, got {len(encoder_outputs)} outputs")
            return encoder_outputs  # 这里返回的是OrtValue列表

        except Exception as e:
            logger.error(f"Encoder inference failed: {e}")
            logger.debug(traceback.format_exc())
            raise e



    def run_batch_decoder_optimized(self, encoder_outputs, batch_size):
        start_time = time.perf_counter()
        try:
            logger.debug(f"Starting optimized batch decoding for {batch_size} samples")

            # 初始化输入数据
            init_start = time.perf_counter()
            input_ids = ort.OrtValue.ortvalue_from_numpy(
                np.array([[3]] * batch_size, dtype=np.int32), 
                self.device_type, self.device_id
            )

            ids_len = ort.OrtValue.ortvalue_from_numpy(
                np.array([1] * batch_size, dtype=np.int64), 
                self.device_type, self.device_id
            )

            history_len = ort.OrtValue.ortvalue_from_numpy(
                np.array([0] * batch_size, dtype=np.int64), 
                self.device_type, self.device_id
            )

            attention_mask = ort.OrtValue.ortvalue_from_numpy(
                np.array([1] * batch_size, dtype=np.int8), 
                self.device_type, self.device_id
            )

            key_shape = self.decoder_session._inputs_meta[0].shape
            value_shape = self.decoder_session._inputs_meta[self.num_layers].shape

            past_keys_B = ort.OrtValue.ortvalue_from_numpy(
                np.zeros((batch_size, key_shape[1], key_shape[2], 0), dtype=self.model_dtype), 
                self.device_type, self.device_id
            )

            past_values_B = ort.OrtValue.ortvalue_from_numpy(
                np.zeros((batch_size, value_shape[1], 0, value_shape[3]), dtype=self.model_dtype), 
                self.device_type, self.device_id
            )
            init_time = time.perf_counter() - init_start
            logger.debug(f"[TIMING] Decoder initialization: {init_time:.4f}s")

            # 构建输入字典
            input_prep_start = time.perf_counter()
            input_feed_B = {
                self.decoder_input_names[-1]: attention_mask,
                self.decoder_input_names[self.num_layers_2]: input_ids,
                self.decoder_input_names[self.num_layers_2_plus_1]: history_len,
                self.decoder_input_names[self.num_layers_2_plus_2]: ids_len
            }

            # 添加KV缓存
            for i in range(self.num_layers):
                input_feed_B[self.decoder_input_names[i]] = past_keys_B
                input_feed_B[self.decoder_input_names[i + self.num_layers]] = past_values_B

            # 添加encoder输出 - 修复这里的错误
            logger.debug("添加encoder输出")
            for i in range(self.num_layers_2):  # 修复：使用range而不是enumerate
                input_feed_B[self.in_name_B[self.layer_indices[i]].name] = encoder_outputs[i]
            
            input_prep_time = time.perf_counter() - input_prep_start
            logger.debug(f"[TIMING] Decoder input preparation: {input_prep_time:.4f}s")

            logger.debug("开始解码循环")

            # 存储所有生成的tokens
            all_generated_tokens = [[] for _ in range(batch_size)]
            num_decode = 0
            decode_loop_start = time.perf_counter()
            step_times = []

            # 自回归解码循环
            for step in range(self.generate_limit):
                step_start = time.perf_counter()
                
                # 运行decoder推理
                inference_start = time.perf_counter()
                decoder_outputs = self.decoder_session.run_with_ort_values(
                    self.decoder_output_names, input_feed_B
                )
                inference_time = time.perf_counter() - inference_start

                # 获取生成的token IDs
                token_extract_start = time.perf_counter()
                max_logit_ids = decoder_outputs[-2].numpy()  # 直接调用.numpy()方法
                num_decode += 1
                token_extract_time = time.perf_counter() - token_extract_start

                # 检查是否所有序列都生成了停止token
                if all(id in self.stop_token for id in max_logit_ids):
                    logger.debug(f"All sequences finished at step {step}")
                    break

                # 更新输入 - 使用decoder的输出作为下一步的输入
                update_start = time.perf_counter()
                for i in range(self.amount_of_outputs):
                    input_feed_B[self.in_name_B[i].name] = decoder_outputs[i]

                # 第一步后更新attention_mask
                if num_decode < 2:
                    input_feed_B[self.in_name_B[-1].name] = ort.OrtValue.ortvalue_from_numpy(
                        np.array([0] * batch_size, dtype=np.int8), 
                        self.device_type, self.device_id
                    )

                # 保存生成的tokens
                for j, token_id in enumerate(max_logit_ids):
                    all_generated_tokens[j].append(token_id)
                
                update_time = time.perf_counter() - update_start
                step_time = time.perf_counter() - step_start
                step_times.append(step_time)
                
                logger.debug(f"[TIMING] Step {step}: inference={inference_time:.4f}s, "
                           f"token_extract={token_extract_time:.4f}s, update={update_time:.4f}s, "
                           f"total={step_time:.4f}s")

            decode_loop_time = time.perf_counter() - decode_loop_start
            avg_step_time = sum(step_times) / len(step_times) if step_times else 0
            logger.debug(f"[TIMING] Decode loop: {decode_loop_time:.4f}s, "
                       f"steps={num_decode}, avg_step={avg_step_time:.4f}s")

            total_time = time.perf_counter() - start_time
            logger.debug(f"[TIMING] Total decoder processing: {total_time:.4f}s")
            logger.debug(f"Batch decoding completed after {num_decode} steps")
            return all_generated_tokens

        except Exception as e:
            logger.error(f"Batch decoder inference failed: {e}")
            logger.debug(traceback.format_exc())
            raise e

    def execute(self, requests):
        """执行推理请求 - 优化版本"""
        total_start_time = time.perf_counter()
        responses = []
        
        try:
            # 解析所有请求中的音频数据
            parse_start = time.perf_counter()
            batch_audio_data = []
            for request in requests:
                # 获取音频输入
                in_0 = pb_utils.get_input_tensor_by_name(request, "audio_data")
                in_1 = pb_utils.get_input_tensor_by_name(request, "wav_length")
                if in_0 is None:
                    raise ValueError("Missing required input 'audio_data'")
                
                audio_bytes = in_0.as_numpy()[0]
                wav_length = in_1.as_numpy()[0]
                batch_audio_data.append((audio_bytes, wav_length))
            
            batch_size = len(batch_audio_data)
            parse_time = time.perf_counter() - parse_start
            logger.debug(f"[TIMING] Request parsing: {parse_time:.4f}s")
            logger.info(f"Processing batch of {batch_size} audio samples")
            
            # 1. 批量特征提取
            feature_start = time.perf_counter()
            audio_features, mask, input_lengths = self.process_audio_data(batch_audio_data)
            feature_total_time = time.perf_counter() - feature_start
            logger.debug(f"[TIMING] Total feature processing: {feature_total_time:.4f}s")
            logger.debug(f"Feature extraction completed for batch size {batch_size}")
            
            # 2. Encoder推理
            encoder_start = time.perf_counter()
            encoder_outputs = self.run_encoder_optimized(audio_features, mask)
            encoder_total_time = time.perf_counter() - encoder_start
            logger.debug(f"[TIMING] Total encoder processing: {encoder_total_time:.4f}s")
            logger.debug("Encoder inference completed")
            
            # 3. 批量Decoder推理
            decoder_start = time.perf_counter()
            batch_generated_tokens = self.run_batch_decoder_optimized(encoder_outputs, batch_size)
            decoder_total_time = time.perf_counter() - decoder_start
            logger.debug(f"[TIMING] Total decoder processing: {decoder_total_time:.4f}s")
            logger.debug("Decoder inference completed")
            
            # 4. 批量解码tokens为文本
            token_decode_start = time.perf_counter()
            transcriptions = self.decode_tokens_batch(batch_generated_tokens)
            token_decode_total_time = time.perf_counter() - token_decode_start
            logger.debug(f"[TIMING] Total token decoding: {token_decode_total_time:.4f}s")
            logger.debug("Token decoding completed")
            
            # 5. 构建响应
            response_start = time.perf_counter()
            for i, transcription in enumerate(transcriptions):
                # 创建输出张量
                out_tensor = pb_utils.Tensor("transcription", 
                                           np.array([transcription], dtype=self.output0_dtype))
                
                # 创建推理响应
                inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(inference_response)
                
                logger.info(f"Sample {i}: transcription='{transcription}' "
                           f"(tokens={len(batch_generated_tokens[i])}, "
                           f"audio_len={input_lengths[i]})")
            
            response_time = time.perf_counter() - response_start
            logger.debug(f"[TIMING] Response construction: {response_time:.4f}s")
            
            # 总体耗时统计
            total_time = time.perf_counter() - total_start_time
            logger.debug(f"[TIMING] ===== TOTAL INFERENCE TIME: {total_time:.4f}s =====")
            logger.debug(f"[TIMING] Breakdown - Parse: {parse_time:.4f}s ({parse_time/total_time*100:.1f}%), "
                       f"Feature: {feature_total_time:.4f}s ({feature_total_time/total_time*100:.1f}%), "
                       f"Encoder: {encoder_total_time:.4f}s ({encoder_total_time/total_time*100:.1f}%), "
                       f"Decoder: {decoder_total_time:.4f}s ({decoder_total_time/total_time*100:.1f}%), "
                       f"TokenDecode: {token_decode_total_time:.4f}s ({token_decode_total_time/total_time*100:.1f}%), "
                       f"Response: {response_time:.4f}s ({response_time/total_time*100:.1f}%)")
            
            return responses
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            logger.debug(traceback.format_exc())
            
            # 返回错误响应
            error_responses = []
            for i, request in enumerate(requests):
                error_msg = f"inference_error: {str(e)}"
                out_tensor = pb_utils.Tensor("transcription", 
                                           np.array([error_msg], dtype=self.output0_dtype))
                error_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                error_responses.append(error_response)
            
            return error_responses

    def finalize(self):
        """清理资源"""
        logger.info("Finalizing FireRedASR pipeline model...")
        
        # 清理ONNX会话
        if hasattr(self, 'encoder_session') and self.encoder_session:
            del self.encoder_session
        if hasattr(self, 'decoder_session') and self.decoder_session:
            del self.decoder_session
        
        # 清理其他资源
        if hasattr(self, 'feature_extractor'):
            del self.feature_extractor
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        logger.info("FireRedASR onnx model finalized")

