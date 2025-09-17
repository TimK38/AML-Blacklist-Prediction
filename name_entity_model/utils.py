"""
工具函數模組 - 基於原始notebook的核心工具函數
Utility functions module based on the original notebook core functions
"""

import torch
import numpy as np
import os
import warnings
from typing import List

from .config import config

warnings.filterwarnings("ignore")

# NumPy 版本兼容性
from .numpy_compat import NUMPY_INF


def tokenize_and_preserve_labels(tokenizer, sentence, text_labels):
    """
    對句子進行分詞並保持標籤對應關係 (來自原始notebook)
    
    Args:
        tokenizer: BERT分詞器
        sentence (list): 詞語列表
        text_labels (list): 對應的標籤列表
        
    Returns:
        tuple: (tokenized_sentence, labels)
    """
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


def cut_tag_encoding(words, tags_encoding):
    """
    剪裁標籤編碼到與詞語相同的長度 (來自原始notebook)
    
    Args:
        words: 詞語列表
        tags_encoding: 標籤編碼列表
        
    Returns:
        list: 剪裁後的標籤編碼
    """
    return tags_encoding[:len(words)]


def extract_entities_from_tags(tokens: List[str], tags: List[str]) -> List[str]:
    """
    從BIO標籤中提取實體 (用於預測結果處理)
    
    Args:
        tokens: 詞語列表
        tags: 標籤列表
        
    Returns:
        list: 實體列表
    """
    entities = []
    current_entity = []
    
    for token, tag in zip(tokens, tags):
        if tag.startswith('B_'):
            if current_entity:
                entities.append(''.join(current_entity))
            current_entity = [token]
        elif tag.startswith('M_') or tag.startswith('I_'):
            if current_entity:
                current_entity.append(token)
        elif tag.startswith('E_'):
            if current_entity:
                current_entity.append(token)
                entities.append(''.join(current_entity))
                current_entity = []
        else:  # O tag
            if current_entity:
                entities.append(''.join(current_entity))
                current_entity = []
    
    # 處理最後一個實體
    if current_entity:
        entities.append(''.join(current_entity))
    
    return entities


def setup_environment():
    """設置訓練環境 (基於原始notebook的設置)"""
    # 確保必要的目錄存在
    config.ensure_dirs()
    
    # 設置隨機種子
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_STATE)
        torch.cuda.manual_seed_all(config.RANDOM_STATE)
    
    print(f"環境設置完成")
    print(f"使用設備: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU設備名稱: {torch.cuda.get_device_name(0)}")


class EarlyStopping:
    """早停機制 (可選的訓練輔助功能)"""
    
    def __init__(self, patience=3, verbose=False, delta=0):
        """
        Args:
            patience (int): 在驗證損失停止改善後等待多少個epoch
            verbose (bool): 是否打印詳細信息
            delta (float): 被視為改善的最小變化量
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = NUMPY_INF
        self.delta = delta

    def __call__(self, val_loss, model=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f'Validation loss: {val_loss:.6f}')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f'Validation loss improved: {val_loss:.6f}')


if __name__ == "__main__":
    # 測試核心工具函數
    print("工具函數模組測試")
    
    # 測試環境設置
    setup_environment()
    
    # 測試標籤剪裁
    words = ['張', '安', '樂']
    tags = [1, 2, 3, 4, 4]  # 多出的標籤
    cut_tags = cut_tag_encoding(words, tags)
    print(f"標籤剪裁測試: {cut_tags}")
    
    print("工具函數模組測試完成！")
