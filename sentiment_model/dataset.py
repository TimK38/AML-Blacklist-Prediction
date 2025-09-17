"""
資料集處理模組
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from .config import Config


class SentimentDataset(Dataset):
    """情感分析資料集"""
    
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class DataProcessor:
    """資料處理器"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.tokenizer = BertTokenizer.from_pretrained(self.config.MODEL_NAME)
    
    def load_data(self):
        """載入資料"""
        df = pd.read_csv(self.config.DATA_PATH)
        print(f"資料集大小: {len(df)}")
        print(f"標籤分布:\n{df.tag.value_counts()}")
        return df
    
    def split_data(self, df):
        """分割資料集"""
        # 先分出測試集
        df_train_val, df_test = train_test_split(
            df,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=df['tag']
        )
        
        # 再從訓練集中分出驗證集
        val_size_adjusted = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
        df_train, df_val = train_test_split(
            df_train_val,
            test_size=val_size_adjusted,
            random_state=self.config.RANDOM_STATE,
            stratify=df_train_val['tag']
        )
        
        print(f"訓練集大小: {len(df_train)}")
        print(f"驗證集大小: {len(df_val)}")
        print(f"測試集大小: {len(df_test)}")
        
        return df_train, df_val, df_test
    
    def create_data_loader(self, df, shuffle=False):
        """建立資料載入器"""
        dataset = SentimentDataset(
            reviews=df.truncated_text.to_numpy(),
            targets=df.tag.to_numpy(),
            tokenizer=self.tokenizer,
            max_len=self.config.MAX_LEN
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=self.config.NUM_WORKERS
        )
    
    def prepare_data(self):
        """準備所有資料"""
        # 載入資料
        df = self.load_data()
        
        # 分割資料
        df_train, df_val, df_test = self.split_data(df)
        
        # 建立資料載入器
        train_loader = self.create_data_loader(df_train, shuffle=True)
        val_loader = self.create_data_loader(df_val, shuffle=False)
        test_loader = self.create_data_loader(df_test, shuffle=False)
        
        return train_loader, val_loader, test_loader
