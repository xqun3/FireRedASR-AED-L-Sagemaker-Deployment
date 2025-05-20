# FireRedASR-AED-L-Sagemaker-Deployment


## 部署步骤
1. 克隆代码库到本地
2. 下载模型权重文件到 FireRedASR-AED-L-Sagemaker-Deployment/code/pretrained_models 路径下，完整文件结构如下
```
.
├── code
│   ├── fireredasr
│   │   ├── data
│   │   │   ├── asr_feat.py
│   │   │   └── token_dict.py
│   │   ├── models
│   │   │   ├── fireredasr_aed.py
│   │   │   ├── fireredasr_llm.py
│   │   │   ├── fireredasr.py
│   │   │   └── module
│   │   │       ├── adapter.py
│   │   │       ├── conformer_encoder.py
│   │   │       └── transformer_decoder.py
│   │   ├── speech2text.py
│   │   ├── tokenizer
│   │   │   ├── aed_tokenizer.py
│   │   │   └── llm_tokenizer.py
│   │   └── utils
│   │       ├── param.py
│   │       └── wer.py
│   ├── inference.py
│   ├── pretrained_models
│   │   ├── FireRedASR-AED-L
│   │   │   ├── cmvn.ark
│   │   │   ├── cmvn.txt
│   │   │   ├── config.yaml
│   │   │   ├── dict.txt
│   │   │   ├── model.pth.tar
│   │   │   ├── README.md
│   │   │   └── train_bpe1000.model
│   │   └── README.md
│   └── requirements.txt
├── examples
│   ├── fireredasr -> ../fireredasr
│   ├── inference_fireredasr_aed.sh
│   ├── inference_fireredasr_llm.sh
│   ├── pretrained_models -> ../pretrained_models
│   └── wav
│       ├── BAC009S0764W0121.wav
│       ├── IT0011W0001.wav
│       ├── TEST_MEETING_T0000000001_S00000.wav
│       ├── TEST_NET_Y0000000000_-KTKHdZ2fb8_S00000.wav
│       ├── text
│       └── wav.scp
├── output.wav
├── README.md
└── sagemaker-notebook.ipynb
```

3. 逐步执行 sagemaker-notebook.ipynb 中的代码部署模型