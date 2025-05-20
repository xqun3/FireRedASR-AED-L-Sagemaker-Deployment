import os
import sys

import torch
import logging
import json
import numpy as np
import torch.nn.functional as F
import uuid


from fireredasr.models.fireredasr import FireRedAsr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_request_id():
    """生成唯一的request_id"""
    return str(uuid.uuid4())

def model_fn(model_dir):
    model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L/")
    return model

def input_fn(request_body, request_content_type):
    audio = np.frombuffer(request_body, dtype=np.int16)
    request_id = generate_request_id()
    return (audio, request_id)


def predict_fn(input_data, model):
    audio_data, request_id = input_data
    results = model.transcribe(
        [request_id],
        audio_data,
        {
            "use_gpu": 1,
            "beam_size": 3,
            "nbest": 1,
            "decode_max_len": 0,
            "softmax_smoothing": 1.0,
            "aed_length_penalty": 0.0,
            "eos_penalty": 1.0
        }
    )
    logger.info(f'ori results: {results}')

    logger.info(
        f'Transcription generated: {results[0]["text"]}') 
    return results


def output_fn(prediction, response_content_type):
    logger.info(
        f'Formatting output with content type: {response_content_type}')
    if response_content_type == 'application/json':
        return json.dumps({'transcription': prediction})
    raise ValueError(f'Unsupported content type: {response_content_type}')
