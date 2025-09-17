"""
模型訓練器
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from collections import defaultdict
from .config import Config


class SentimentTrainer:
    """情感分析模型訓練器"""
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or Config()
        self.device = self.config.DEVICE
        
        # 設置優化器
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.config.LEARNING_RATE
        )
        
        # 損失函數
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        
        # 訓練歷史記錄
        self.history = defaultdict(list)
        self.best_accuracy = 0.0
        
    def setup_scheduler(self, train_loader):
        """設置學習率調度器"""
        total_steps = len(train_loader) * self.config.EPOCHS
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
    
    def train_epoch(self, train_loader):
        """訓練一個epoch"""
        self.model.train()
        losses = []
        correct_predictions = 0
        total_samples = 0
        
        for batch in train_loader:
            # 移動數據到設備
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["targets"].to(self.device)
            
            # 前向傳播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 計算損失
            loss = self.loss_fn(outputs, targets)
            
            # 計算準確率
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)
            total_samples += targets.size(0)
            
            # 記錄損失
            losses.append(loss.item())
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪
            nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.MAX_GRAD_NORM
            )
            
            # 更新參數
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        accuracy = correct_predictions.double() / total_samples
        avg_loss = np.mean(losses)
        
        return accuracy, avg_loss
    
    def eval_model(self, data_loader):
        """評估模型"""
        self.model.eval()
        losses = []
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # 移動數據到設備
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["targets"].to(self.device)
                
                # 前向傳播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # 計算損失
                loss = self.loss_fn(outputs, targets)
                
                # 計算準確率
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == targets)
                total_samples += targets.size(0)
                
                losses.append(loss.item())
        
        accuracy = correct_predictions.double() / total_samples
        avg_loss = np.mean(losses)
        
        return accuracy, avg_loss
    
    def save_model(self, path):
        """保存模型"""
        # 確保目錄存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到: {path}")
    
    def train(self, train_loader, val_loader):
        """完整的訓練流程"""
        print("開始訓練...")
        print(f"設備: {self.device}")
        print(f"訓練批次數: {len(train_loader)}")
        print(f"驗證批次數: {len(val_loader)}")
        
        # 設置調度器
        self.setup_scheduler(train_loader)
        
        # 早停相關變數
        patience_counter = 0
        
        for epoch in range(self.config.EPOCHS):
            print(f'\nEpoch {epoch + 1}/{self.config.EPOCHS}')
            print('-' * 50)
            
            # 訓練
            train_acc, train_loss = self.train_epoch(train_loader)
            print(f'訓練 - 損失: {train_loss:.4f}, 準確率: {train_acc:.4f}')
            
            # 驗證
            val_acc, val_loss = self.eval_model(val_loader)
            print(f'驗證 - 損失: {val_loss:.4f}, 準確率: {val_acc:.4f}')
            
            # 記錄歷史
            self.history['train_acc'].append(train_acc)
            self.history['train_loss'].append(train_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_loss'].append(val_loss)
            
            # 保存最佳模型
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_model(self.config.MODEL_SAVE_PATH)
                patience_counter = 0
                print(f'新的最佳驗證準確率: {val_acc:.4f}')
            else:
                patience_counter += 1
                print(f'驗證準確率沒有提升 ({patience_counter}/{self.config.PATIENCE})')
            
            # 早停檢查
            if patience_counter >= self.config.PATIENCE:
                print(f'\n早停觸發！連續 {self.config.PATIENCE} 個epoch沒有改善')
                break
        
        print(f'\n訓練完成！最佳驗證準確率: {self.best_accuracy:.4f}')
        return self.history
