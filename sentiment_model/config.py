"""
情感分析模型配置文件
"""
import torch

class Config:
    """模型配置類"""
    
    # 資料相關配置
    DATA_PATH = 'data/Sentiment_model_data/step2_2_output_train_data_300.csv'
    MODEL_SAVE_PATH = 'model/best_model_state.pth'
    
    # 模型相關配置
    MODEL_NAME = "bert-base-chinese"
    MAX_LEN = 300
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    DROPOUT_RATE = 0.3
    
    # 訓練相關配置
    TEST_SIZE = 0.1
    VAL_SIZE = 0.1  # 從訓練集中分出的驗證集比例
    RANDOM_STATE = 38
    NUM_WORKERS = 0
    
    # 類別標籤
    CLASS_NAMES = ['negative', 'positive']
    N_CLASSES = len(CLASS_NAMES)
    
    # 設備配置
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 早停配置
    PATIENCE = 3  # 如果驗證準確率連續3個epoch沒有提升就停止訓練
    
    # 梯度裁剪
    MAX_GRAD_NORM = 1.0
