"""
模型定義模組 - 定義NER模型結構
Model definition module for NER model architecture
"""

import torch
import torch.nn as nn
from transformers import BertForTokenClassification, BertConfig
import os

from .config import config


class NERModel:
    """NER模型封裝類"""
    
    def __init__(self, model_name=None, num_labels=None):
        """
        初始化NER模型
        
        Args:
            model_name (str): 預訓練模型名稱
            num_labels (int): 標籤數量
        """
        self.model_name = model_name or config.MODEL_NAME
        self.num_labels = num_labels or config.NUM_LABELS
        self.model = None
        self.device = config.DEVICE
        
    def create_model(self):
        """創建BERT NER模型"""
        try:
            self.model = BertForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                output_attentions=False,
                output_hidden_states=False
            )
            
            # 移動到指定設備
            self.model.to(self.device)
            
            print(f"成功創建模型: {self.model_name}")
            print(f"標籤數量: {self.num_labels}")
            print(f"設備: {self.device}")
            
            return self.model
            
        except Exception as e:
            raise Exception(f"創建模型失敗: {str(e)}")
    
    def get_model(self):
        """獲取模型實例"""
        if self.model is None:
            self.create_model()
        return self.model
    
    def save_model(self, save_path=None):
        """
        保存模型權重
        
        Args:
            save_path (str): 保存路徑
        """
        if self.model is None:
            raise ValueError("模型尚未創建，無法保存")
        
        save_path = save_path or config.MODEL_SAVE_PATH
        
        # 確保保存目錄存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            torch.save(self.model.state_dict(), save_path)
            print(f"模型權重已保存到: {save_path}")
        except Exception as e:
            raise Exception(f"保存模型失敗: {str(e)}")
    
    def load_model(self, model_path=None):
        """
        加載模型權重
        
        Args:
            model_path (str): 模型權重文件路徑
        """
        if self.model is None:
            self.create_model()
        
        model_path = model_path or config.MODEL_SAVE_PATH
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            print(f"成功加載模型權重: {model_path}")
        except Exception as e:
            raise Exception(f"加載模型失敗: {str(e)}")
    
    def get_model_info(self):
        """獲取模型信息"""
        if self.model is None:
            return "模型尚未創建"
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'model_config': self.model.config.to_dict() if hasattr(self.model, 'config') else None
        }
        
        return info
    
    def set_training_mode(self, training=True):
        """設置模型為訓練或評估模式"""
        if self.model is None:
            raise ValueError("模型尚未創建")
        
        if training:
            self.model.train()
        else:
            self.model.eval()
    
    def get_optimizer_parameters(self, full_finetuning=None):
        """
        獲取優化器參數組
        
        Args:
            full_finetuning (bool): 是否進行全量微調
            
        Returns:
            list: 參數組列表
        """
        if self.model is None:
            raise ValueError("模型尚未創建")
        
        full_finetuning = full_finetuning if full_finetuning is not None else config.FULL_FINETUNING
        
        if full_finetuning:
            # 全量微調：包含所有參數
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    'weight_decay_rate': config.WEIGHT_DECAY_RATE
                },
                {
                    'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.0
                }
            ]
        else:
            # 只微調分類器
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        
        return optimizer_grouped_parameters


class ModelManager:
    """模型管理器 - 提供更高級的模型管理功能"""
    
    def __init__(self):
        self.ner_model = None
    
    def initialize_model(self, model_name=None, num_labels=None):
        """初始化模型"""
        self.ner_model = NERModel(model_name, num_labels)
        return self.ner_model.create_model()
    
    def load_pretrained_model(self, model_path=None):
        """加載預訓練模型"""
        if self.ner_model is None:
            self.initialize_model()
        
        self.ner_model.load_model(model_path)
        return self.ner_model.get_model()
    
    def save_model_checkpoint(self, save_path=None, epoch=None, loss=None, additional_info=None):
        """
        保存模型檢查點（包含額外信息）
        
        Args:
            save_path (str): 保存路徑
            epoch (int): 當前epoch
            loss (float): 當前損失
            additional_info (dict): 額外信息
        """
        if self.ner_model is None:
            raise ValueError("模型尚未初始化")
        
        save_path = save_path or config.MODEL_SAVE_PATH
        
        # 準備檢查點數據
        checkpoint = {
            'model_state_dict': self.ner_model.get_model().state_dict(),
            'model_config': {
                'model_name': self.ner_model.model_name,
                'num_labels': self.ner_model.num_labels,
            }
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        if additional_info:
            checkpoint.update(additional_info)
        
        try:
            torch.save(checkpoint, save_path)
            print(f"模型檢查點已保存: {save_path}")
        except Exception as e:
            raise Exception(f"保存檢查點失敗: {str(e)}")
    
    def get_model(self):
        """獲取當前模型"""
        if self.ner_model is None:
            raise ValueError("模型尚未初始化")
        return self.ner_model.get_model()


def create_ner_model(model_name=None, num_labels=None):
    """
    便捷函數：創建NER模型
    
    Args:
        model_name (str): 模型名稱
        num_labels (int): 標籤數量
        
    Returns:
        torch.nn.Module: NER模型
    """
    ner_model = NERModel(model_name, num_labels)
    return ner_model.create_model()


def load_trained_model(model_path=None):
    """
    便捷函數：加載已訓練的模型
    
    Args:
        model_path (str): 模型權重路徑
        
    Returns:
        torch.nn.Module: 加載權重後的模型
    """
    ner_model = NERModel()
    model = ner_model.create_model()
    ner_model.load_model(model_path)
    return model


if __name__ == "__main__":
    # 測試模型創建
    try:
        print("測試模型創建...")
        model = create_ner_model()
        print("模型創建成功！")
        
        # 測試模型管理器
        print("\n測試模型管理器...")
        manager = ModelManager()
        model = manager.initialize_model()
        info = manager.ner_model.get_model_info()
        print(f"模型信息: {info}")
        
    except Exception as e:
        print(f"模型測試失敗: {str(e)}")
