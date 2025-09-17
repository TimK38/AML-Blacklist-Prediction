"""
數據處理模組 - 處理NER訓練數據的加載和預處理
Data processing module for NER training data loading and preprocessing
"""

import pandas as pd
import ast
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import warnings

from .config import config

warnings.filterwarnings("ignore")


class NERDataLoader:
    """NER數據加載器類"""
    
    def __init__(self, data_file=None):
        """
        初始化數據加載器
        
        Args:
            data_file (str): 數據文件路徑，如果為None則使用config中的默認路徑
        """
        self.data_file = data_file or config.DATA_FILE
        self.tokenizer = BertTokenizer.from_pretrained(
            config.MODEL_NAME, 
            do_lower_case=config.DO_LOWER_CASE
        )
        self.data = None
        self.train_dataloader = None
        self.valid_dataloader = None
        
    def load_data(self):
        """加載原始數據"""
        try:
            self.data = pd.read_csv(self.data_file)
            print(f"成功加載數據，共 {len(self.data)} 條記錄")
            
            # 解析字符串格式的列表
            self.data.words = self.data.words.apply(lambda x: ast.literal_eval(x))
            self.data.tags_encoding = self.data.tags_encoding.apply(lambda x: ast.literal_eval(x))
            
            # 剪裁標籤到與words相同長度
            self.data['tags_encoding_cut'] = self.data.apply(
                lambda x: self._cut_tag_encoding(x.words, x.tags_encoding), axis=1
            )
            
            return self.data
        except Exception as e:
            raise Exception(f"加載數據失敗: {str(e)}")
    
    def _cut_tag_encoding(self, words, tags_encoding):
        """剪裁標籤編碼到與詞語相同的長度"""
        return tags_encoding[:len(words)]
    
    def prepare_data(self):
        """準備訓練數據"""
        if self.data is None:
            self.load_data()
        
        # 將詞語轉換為token IDs
        tokenized_texts = [
            self.tokenizer.convert_tokens_to_ids(txt) 
            for txt in self.data['words'].tolist()
        ]
        
        # 填充序列到固定長度
        input_ids = pad_sequences(
            tokenized_texts,
            maxlen=config.MAX_LEN, 
            dtype="long", 
            value=0.0,
            truncating="post", 
            padding="post"
        )
        
        # 創建注意力掩碼
        attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
        
        # 準備標籤
        labels = [
            [config.TAG2IDX.get(l) for l in lab] 
            for lab in self.data['tags_encoding_cut'].tolist()
        ]
        
        # 填充標籤序列
        tags = pad_sequences(
            labels,
            maxlen=config.MAX_LEN, 
            value=config.TAG2IDX["PAD"], 
            padding="post",
            dtype="long", 
            truncating="post"
        )
        
        return input_ids, attention_masks, tags
    
    def create_data_loaders(self):
        """創建訓練和驗證數據加載器"""
        input_ids, attention_masks, tags = self.prepare_data()
        
        # 訓練/驗證分割
        tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, val_masks = train_test_split(
            input_ids, tags, attention_masks,
            random_state=config.RANDOM_STATE,
            test_size=config.TRAIN_TEST_SPLIT
        )
        
        # 轉換為PyTorch張量
        tr_inputs = torch.tensor(tr_inputs, dtype=torch.long)
        val_inputs = torch.tensor(val_inputs, dtype=torch.long)
        tr_tags = torch.tensor(tr_tags, dtype=torch.long)
        val_tags = torch.tensor(val_tags, dtype=torch.long)
        tr_masks = torch.tensor(tr_masks, dtype=torch.long)
        val_masks = torch.tensor(val_masks, dtype=torch.long)
        
        # 創建數據集
        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        valid_data = TensorDataset(val_inputs, val_masks, val_tags)
        
        # 創建數據加載器
        train_sampler = RandomSampler(train_data)
        valid_sampler = SequentialSampler(valid_data)
        
        self.train_dataloader = DataLoader(
            train_data, 
            sampler=train_sampler, 
            batch_size=config.BATCH_SIZE
        )
        
        self.valid_dataloader = DataLoader(
            valid_data, 
            sampler=valid_sampler, 
            batch_size=config.BATCH_SIZE
        )
        
        print(f"訓練數據: {len(train_data)} 條")
        print(f"驗證數據: {len(valid_data)} 條")
        print(f"訓練批次數: {len(self.train_dataloader)}")
        print(f"驗證批次數: {len(self.valid_dataloader)}")
        
        return self.train_dataloader, self.valid_dataloader
    
    def get_data_info(self):
        """獲取數據集信息"""
        if self.data is None:
            self.load_data()
            
        info = {
            'total_samples': len(self.data),
            'sample_sentence': self.data.iloc[0]['words'][:10],
            'sample_tags': self.data.iloc[0]['tags_encoding_cut'][:10],
            'tag_distribution': {}
        }
        
        # 統計標籤分佈
        all_tags = []
        for tags in self.data['tags_encoding_cut']:
            all_tags.extend(tags)
        
        for tag_idx in config.TAG2IDX.values():
            tag_name = config.IDX2TAG[tag_idx]
            info['tag_distribution'][tag_name] = all_tags.count(tag_idx)
        
        return info


def create_data_loaders(data_file=None):
    """
    便捷函數：創建數據加載器
    
    Args:
        data_file (str): 數據文件路徑
        
    Returns:
        tuple: (train_dataloader, valid_dataloader, data_info)
    """
    loader = NERDataLoader(data_file)
    train_dl, valid_dl = loader.create_data_loaders()
    info = loader.get_data_info()
    
    return train_dl, valid_dl, info


if __name__ == "__main__":
    # 測試數據加載器
    try:
        train_dl, valid_dl, info = create_data_loaders()
        print("數據加載器測試成功！")
        print(f"數據集信息: {info}")
    except Exception as e:
        print(f"數據加載器測試失敗: {str(e)}")
