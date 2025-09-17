"""
配置文件 - NER模型的所有配置參數
Configuration file for NER model parameters
"""

import os
import torch

class Config:
    """NER模型配置類"""
    
    # 模型配置
    MODEL_NAME = "bert-base-chinese"
    MAX_LEN = 75
    BATCH_SIZE = 32
    EPOCHS = 3
    LEARNING_RATE = 3e-5
    MAX_GRAD_NORM = 1.0
    
    # 標籤配置
    TAG_VALUES = ['O', 'B_person_name', 'M_person_name', 'E_person_name', 'PAD']
    TAG2IDX = {tag: idx for idx, tag in enumerate(TAG_VALUES)}
    IDX2TAG = {idx: tag for tag, idx in TAG2IDX.items()}
    NUM_LABELS = len(TAG_VALUES)
    
    # 路徑配置
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "亞洲研究院數據集")
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    
    # 數據文件路徑
    DATA_FILE = os.path.join(DATA_DIR, "asia_institute.csv")
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "ner_model_weights.pth")
    
    # 訓練配置
    TRAIN_TEST_SPLIT = 0.1
    RANDOM_STATE = 38
    WARMUP_STEPS = 0
    
    # 設備配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_GPU = torch.cuda.device_count()
    
    # 優化器配置
    WEIGHT_DECAY_RATE = 0.01
    ADAM_EPSILON = 1e-8
    
    # 其他配置
    DO_LOWER_CASE = False
    FULL_FINETUNING = True
    
    @classmethod
    def ensure_dirs(cls):
        """確保必要的目錄存在"""
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        
    def __str__(self):
        """打印配置信息"""
        config_str = "NER Model Configuration:\n"
        config_str += f"Model Name: {self.MODEL_NAME}\n"
        config_str += f"Max Length: {self.MAX_LEN}\n"
        config_str += f"Batch Size: {self.BATCH_SIZE}\n"
        config_str += f"Epochs: {self.EPOCHS}\n"
        config_str += f"Learning Rate: {self.LEARNING_RATE}\n"
        config_str += f"Device: {self.DEVICE}\n"
        config_str += f"Number of Labels: {self.NUM_LABELS}\n"
        return config_str

# 全局配置實例
config = Config()
