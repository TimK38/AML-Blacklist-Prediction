"""
訓練模組 - NER模型的訓練邏輯
Training module for NER model training logic
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import trange
import numpy as np
from seqeval.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

from .config import config
from .model import ModelManager
from .utils import EarlyStopping


class NERTrainer:
    """NER模型訓練器"""
    
    def __init__(self, model=None, train_dataloader=None, valid_dataloader=None):
        """
        初始化訓練器
        
        Args:
            model: NER模型
            train_dataloader: 訓練數據加載器
            valid_dataloader: 驗證數據加載器
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = config.DEVICE
        
        # 訓練狀態
        self.optimizer = None
        self.scheduler = None
        self.loss_values = []
        self.validation_loss_values = []
        self.best_f1_score = 0.0
        self.early_stopping = None
        
        # 確保模型在正確的設備上
        if self.model:
            self.model.to(self.device)
    
    def setup_optimizer(self, model=None):
        """設置優化器和學習率調度器"""
        model = model or self.model
        if model is None:
            raise ValueError("模型未設置")
        
        # 獲取模型參數組
        if hasattr(model, 'get_optimizer_parameters'):
            optimizer_grouped_parameters = model.get_optimizer_parameters()
        else:
            # 如果是原始BERT模型，手動設置參數組
            param_optimizer = list(model.named_parameters())
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
        
        # 創建優化器
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=config.LEARNING_RATE,
            eps=config.ADAM_EPSILON
        )
        
        # 創建學習率調度器
        total_steps = len(self.train_dataloader) * config.EPOCHS
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        print(f"優化器設置完成，總訓練步數: {total_steps}")
    
    def train_epoch(self):
        """訓練一個epoch"""
        self.model.train()
        total_loss = 0
        
        for step, batch in enumerate(self.train_dataloader):
            # 將數據移動到設備
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            # 清零梯度
            self.model.zero_grad()
            
            # 前向傳播
            outputs = self.model(
                b_input_ids, 
                token_type_ids=None,
                attention_mask=b_input_mask, 
                labels=b_labels
            )
            
            # 獲取損失
            loss = outputs[0]
            
            # 反向傳播
            loss.backward()
            
            # 累積損失
            total_loss += loss.item()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(), 
                max_norm=config.MAX_GRAD_NORM
            )
            
            # 更新參數
            self.optimizer.step()
            self.scheduler.step()
        
        # 計算平均損失
        avg_train_loss = total_loss / len(self.train_dataloader)
        return avg_train_loss
    
    def validate(self):
        """驗證模型"""
        self.model.eval()
        eval_loss = 0
        predictions = []
        true_labels = []
        
        for batch in self.valid_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            with torch.no_grad():
                outputs = self.model(
                    b_input_ids, 
                    token_type_ids=None,
                    attention_mask=b_input_mask, 
                    labels=b_labels
                )
            
            # 獲取logits和標籤
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # 累積損失
            eval_loss += outputs[0].mean().item()
            
            # 收集預測和真實標籤
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)
        
        # 計算平均驗證損失
        eval_loss = eval_loss / len(self.valid_dataloader)
        
        # 轉換為標籤名稱（去除PAD）
        valid_tags = [
            [config.TAG_VALUES[l_i] for l_i in l if config.TAG_VALUES[l_i] != "PAD"]
            for l in true_labels
        ]
        
        pred_tags = [
            [config.TAG_VALUES[p_i] for p_i, l_i in zip(p, l) 
             if config.TAG_VALUES[l_i] != "PAD"]
            for p, l in zip(predictions, true_labels)
        ]
        
        # 計算指標
        accuracy = accuracy_score(valid_tags, pred_tags)
        f1 = f1_score(valid_tags, pred_tags)
        
        return eval_loss, accuracy, f1
    
    def train(self, epochs=None, save_best_model=True, use_early_stopping=True, patience=3):
        """
        完整的訓練流程
        
        Args:
            epochs (int): 訓練輪數
            save_best_model (bool): 是否保存最佳模型
            use_early_stopping (bool): 是否使用早停
            patience (int): 早停耐心值
        """
        epochs = epochs or config.EPOCHS
        
        if self.optimizer is None:
            self.setup_optimizer()
        
        if use_early_stopping:
            self.early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        print(f"開始訓練，共 {epochs} 個epoch")
        print(f"訓練設備: {self.device}")
        
        for epoch in trange(epochs, desc="Epoch"):
            # 訓練一個epoch
            avg_train_loss = self.train_epoch()
            self.loss_values.append(avg_train_loss)
            
            print(f"Average train loss: {avg_train_loss}")
            
            # 驗證
            eval_loss, accuracy, f1 = self.validate()
            self.validation_loss_values.append(eval_loss)
            
            print(f"Validation loss: {eval_loss}")
            print(f"Validation Accuracy: {accuracy:.4f}")
            print(f"Validation F1-Score: {f1:.4f}")
            print()
            
            # 保存最佳模型
            if save_best_model and f1 > self.best_f1_score:
                self.best_f1_score = f1
                self.save_model()
                print(f"保存最佳模型，F1-Score: {f1:.4f}")
            
            # 早停檢查
            if use_early_stopping:
                self.early_stopping(eval_loss, self.model)
                if self.early_stopping.early_stop:
                    print("早停觸發，停止訓練")
                    break
        
        print(f"訓練完成！最佳F1-Score: {self.best_f1_score:.4f}")
        return self.loss_values, self.validation_loss_values
    
    def save_model(self, save_path=None):
        """保存模型"""
        save_path = save_path or config.MODEL_SAVE_PATH
        
        # 確保目錄存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            torch.save(self.model.state_dict(), save_path)
            print(f"模型已保存到: {save_path}")
        except Exception as e:
            print(f"保存模型失敗: {str(e)}")
    
    def plot_training_history(self, save_path=None):
        """繪製訓練歷史"""
        if not self.loss_values or not self.validation_loss_values:
            print("沒有訓練歷史數據可繪製")
            return
        
        # 設置繪圖樣式
        sns.set(style='darkgrid')
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)
        
        # 繪製學習曲線
        plt.figure(figsize=(12, 6))
        plt.plot(self.loss_values, 'b-o', label="training loss")
        plt.plot(self.validation_loss_values, 'r-o', label="validation loss")
        
        # 設置標籤
        plt.title("Learning Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # 保存圖片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"訓練曲線已保存到: {save_path}")
        
        plt.show()
    
    def get_training_summary(self):
        """獲取訓練摘要"""
        if not self.loss_values:
            return "尚未開始訓練"
        
        summary = {
            'total_epochs': len(self.loss_values),
            'final_train_loss': self.loss_values[-1] if self.loss_values else None,
            'final_valid_loss': self.validation_loss_values[-1] if self.validation_loss_values else None,
            'best_f1_score': self.best_f1_score,
            'min_train_loss': min(self.loss_values) if self.loss_values else None,
            'min_valid_loss': min(self.validation_loss_values) if self.validation_loss_values else None,
        }
        
        return summary


def train_ner_model(train_dataloader, valid_dataloader, model=None, epochs=None, **kwargs):
    """
    便捷函數：訓練NER模型
    
    Args:
        train_dataloader: 訓練數據加載器
        valid_dataloader: 驗證數據加載器
        model: 模型（如果為None則創建新模型）
        epochs: 訓練輪數
        **kwargs: 其他訓練參數
        
    Returns:
        tuple: (trainer, model)
    """
    # 創建模型（如果未提供）
    if model is None:
        from .model import create_ner_model
        model = create_ner_model()
    
    # 創建訓練器
    trainer = NERTrainer(model, train_dataloader, valid_dataloader)
    
    # 開始訓練
    trainer.train(epochs=epochs, **kwargs)
    
    return trainer, model


if __name__ == "__main__":
    # 測試訓練器
    print("訓練器模組測試")
    
    try:
        # 這裡需要實際的數據加載器來測試
        from .data_loader import create_data_loaders
        from .model import create_ner_model
        
        # 創建數據加載器
        train_dl, valid_dl, _ = create_data_loaders()
        
        # 創建模型
        model = create_ner_model()
        
        # 創建訓練器
        trainer = NERTrainer(model, train_dl, valid_dl)
        
        print("訓練器創建成功！")
        print("開始小規模測試訓練...")
        
        # 小規模測試（只訓練1個epoch）
        trainer.train(epochs=1)
        
        print("訓練器測試完成！")
        
    except Exception as e:
        print(f"訓練器測試失敗: {str(e)}")
