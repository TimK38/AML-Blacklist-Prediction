"""
情感分析模型定義
"""
import torch
import torch.nn as nn
from transformers import BertModel
from .config import Config


class SentimentClassifier(nn.Module):
    """BERT情感分析分類器"""
    
    def __init__(self, config=None):
        super(SentimentClassifier, self).__init__()
        self.config = config or Config()
        
        # BERT模型
        self.bert = BertModel.from_pretrained(self.config.MODEL_NAME)
        
        # Dropout層
        self.dropout = nn.Dropout(p=self.config.DROPOUT_RATE)
        
        # 分類層
        self.classifier = nn.Linear(
            self.bert.config.hidden_size, 
            self.config.N_CLASSES
        )
    
    def forward(self, input_ids, attention_mask):
        """前向傳播"""
        # 通過BERT獲取輸出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 取出[CLS]標記的表示（池化輸出）
        pooled_output = outputs.pooler_output
        
        # 應用Dropout
        output = self.dropout(pooled_output)
        
        # 分類
        logits = self.classifier(output)
        
        return logits


def create_model(config=None):
    """建立並初始化模型"""
    config = config or Config()
    model = SentimentClassifier(config)
    model = model.to(config.DEVICE)
    return model


def load_model(model_path, config=None):
    """載入已訓練的模型"""
    config = config or Config()
    model = create_model(config)
    
    # 載入模型權重
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    
    return model
