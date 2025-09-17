"""
預測模組 - NER模型的推理和預測功能
Prediction module for NER model inference and prediction
"""

import torch
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings

from .config import config
from .model import load_trained_model
from .utils import extract_entities_from_tags

warnings.filterwarnings("ignore")


class NERPredictor:
    """NER預測器類"""
    
    def __init__(self, model_path: str = None, model=None):
        """
        初始化預測器
        
        Args:
            model_path (str): 模型權重文件路徑
            model: 已加載的模型實例
        """
        self.model_path = model_path or config.MODEL_SAVE_PATH
        self.model = model
        self.tokenizer = None
        self.device = config.DEVICE
        self.is_loaded = False
        
        # 如果提供了模型，直接設置為已加載
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
        
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """加載分詞器"""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(
                config.MODEL_NAME, 
                do_lower_case=config.DO_LOWER_CASE
            )
            print("分詞器加載成功")
        except Exception as e:
            raise Exception(f"加載分詞器失敗: {str(e)}")
    
    def load_model(self, model_path: str = None):
        """
        加載訓練好的模型
        
        Args:
            model_path (str): 模型權重文件路徑
        """
        model_path = model_path or self.model_path
        
        try:
            self.model = load_trained_model(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"模型加載成功: {model_path}")
        except Exception as e:
            raise Exception(f"加載模型失敗: {str(e)}")
    
    def _ensure_model_loaded(self):
        """確保模型已加載"""
        if not self.is_loaded:
            self.load_model()
    
    def preprocess_text(self, text: str, max_len: int = None) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        預處理輸入文本
        
        Args:
            text (str): 輸入文本
            max_len (int): 最大序列長度
            
        Returns:
            tuple: (input_ids, attention_masks, tokens)
        """
        max_len = max_len or config.MAX_LEN
        
        # 清理文本（簡單清理）
        text = ' '.join(text.split())
        
        # 分詞
        tokens = self.tokenizer.tokenize(text)
        
        # 轉換為IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # 填充序列
        input_ids = pad_sequences(
            [input_ids],
            maxlen=max_len, 
            dtype="long", 
            value=0,
            truncating="post", 
            padding="post"
        )
        
        # 創建注意力掩碼
        attention_masks = [[float(i != 0.0) for i in seq] for seq in input_ids]
        
        # 轉換為張量
        input_ids = torch.tensor(input_ids).to(self.device)
        attention_masks = torch.tensor(attention_masks).to(self.device)
        
        return input_ids, attention_masks, tokens
    
    def predict_tags(self, text: str, return_confidence: bool = False) -> Dict:
        """
        預測文本中的標籤
        
        Args:
            text (str): 輸入文本
            return_confidence (bool): 是否返回置信度
            
        Returns:
            dict: 預測結果
        """
        self._ensure_model_loaded()
        
        # 預處理
        input_ids, attention_masks, tokens = self.preprocess_text(text)
        
        # 預測
        with torch.no_grad():
            outputs = self.model(
                input_ids, 
                token_type_ids=None, 
                attention_mask=attention_masks
            )
        
        # 獲取logits
        logits = outputs[0]
        
        # 獲取預測標籤
        predictions = torch.argmax(logits, dim=2).cpu().numpy()
        pred_tags = [config.TAG_VALUES[p] for p in predictions[0]][:len(tokens)]
        
        result = {
            'text': text,
            'tokens': tokens,
            'predictions': pred_tags
        }
        
        # 如果需要置信度
        if return_confidence:
            probabilities = torch.softmax(logits, dim=2).cpu().numpy()
            confidences = [float(np.max(probabilities[0][i])) for i in range(len(tokens))]
            result['confidences'] = confidences
        
        return result
    
    def extract_entities(self, text: str, return_positions: bool = False) -> List[Dict]:
        """
        從文本中提取命名實體
        
        Args:
            text (str): 輸入文本
            return_positions (bool): 是否返回實體位置
            
        Returns:
            list: 實體列表
        """
        # 獲取標籤預測
        prediction_result = self.predict_tags(text, return_confidence=True)
        tokens = prediction_result['tokens']
        pred_tags = prediction_result['predictions']
        confidences = prediction_result['confidences']
        
        entities = []
        current_entity = {
            'text': '',
            'tokens': [],
            'label': '',
            'confidence': [],
            'start_idx': -1,
            'end_idx': -1
        }
        
        for i, (token, tag, conf) in enumerate(zip(tokens, pred_tags, confidences)):
            if tag.startswith('B_'):  # 實體開始
                # 保存前一個實體（如果存在）
                if current_entity['text']:
                    entities.append(self._finalize_entity(current_entity))
                
                # 開始新實體
                current_entity = {
                    'text': token,
                    'tokens': [token],
                    'label': tag.replace('B_', ''),
                    'confidence': [conf],
                    'start_idx': i,
                    'end_idx': i
                }
                
            elif tag.startswith('M_') or tag.startswith('I_'):  # 實體中間
                if current_entity['text'] and tag.replace('M_', '').replace('I_', '') == current_entity['label']:
                    current_entity['text'] += token
                    current_entity['tokens'].append(token)
                    current_entity['confidence'].append(conf)
                    current_entity['end_idx'] = i
                    
            elif tag.startswith('E_'):  # 實體結束
                if current_entity['text'] and tag.replace('E_', '') == current_entity['label']:
                    current_entity['text'] += token
                    current_entity['tokens'].append(token)
                    current_entity['confidence'].append(conf)
                    current_entity['end_idx'] = i
                    entities.append(self._finalize_entity(current_entity))
                    current_entity = {'text': '', 'tokens': [], 'label': '', 'confidence': [], 'start_idx': -1, 'end_idx': -1}
                    
            else:  # O標籤
                if current_entity['text']:
                    entities.append(self._finalize_entity(current_entity))
                    current_entity = {'text': '', 'tokens': [], 'label': '', 'confidence': [], 'start_idx': -1, 'end_idx': -1}
        
        # 處理最後一個實體
        if current_entity['text']:
            entities.append(self._finalize_entity(current_entity))
        
        # 如果不需要位置信息，簡化輸出
        if not return_positions:
            entities = [
                {
                    'text': entity['text'],
                    'label': entity['label'],
                    'confidence': entity['avg_confidence']
                }
                for entity in entities
            ]
        
        return entities
    
    def _finalize_entity(self, entity: Dict) -> Dict:
        """完善實體信息"""
        if entity['confidence']:
            entity['avg_confidence'] = float(np.mean(entity['confidence']))
            entity['min_confidence'] = float(np.min(entity['confidence']))
            entity['max_confidence'] = float(np.max(entity['confidence']))
        else:
            entity['avg_confidence'] = 0.0
            entity['min_confidence'] = 0.0
            entity['max_confidence'] = 0.0
        
        return entity
    
    def extract_names(self, text: str) -> List[str]:
        """
        簡化的人名提取函數（與原notebook保持一致）
        
        Args:
            text (str): 輸入文本
            
        Returns:
            list: 人名列表
        """
        entities = self.extract_entities(text)
        
        # 只返回人名實體的文本
        names = [entity['text'] for entity in entities if entity['label'] == 'person_name']
        
        return names
    
    def batch_predict(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """
        批量預測
        
        Args:
            texts (list): 文本列表
            batch_size (int): 批次大小
            
        Returns:
            list: 預測結果列表
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                try:
                    result = self.extract_entities(text)
                    results.append({
                        'text': text,
                        'entities': result,
                        'success': True,
                        'error': None
                    })
                except Exception as e:
                    results.append({
                        'text': text,
                        'entities': [],
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    def get_model_info(self) -> Dict:
        """獲取模型信息"""
        info = {
            'model_path': self.model_path,
            'model_loaded': self.is_loaded,
            'device': str(self.device),
            'tag_values': config.TAG_VALUES,
            'max_length': config.MAX_LEN
        }
        
        if self.is_loaded and self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            })
        
        return info


class BatchPredictor:
    """批量預測器 - 用於處理大量文本"""
    
    def __init__(self, predictor: NERPredictor):
        """
        初始化批量預測器
        
        Args:
            predictor: NER預測器實例
        """
        self.predictor = predictor
    
    def predict_from_file(self, input_file: str, output_file: str, 
                         text_column: str = 'text', batch_size: int = 16):
        """
        從文件批量預測並保存結果
        
        Args:
            input_file (str): 輸入文件路徑
            output_file (str): 輸出文件路徑
            text_column (str): 文本列名
            batch_size (int): 批次大小
        """
        import pandas as pd
        import json
        
        # 讀取輸入文件
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.json'):
            df = pd.read_json(input_file)
        else:
            raise ValueError("支持的文件格式: .csv, .json")
        
        if text_column not in df.columns:
            raise ValueError(f"列 '{text_column}' 不存在於輸入文件中")
        
        texts = df[text_column].tolist()
        
        # 批量預測
        results = self.predictor.batch_predict(texts, batch_size)
        
        # 保存結果
        if output_file.endswith('.json'):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            # 轉換為DataFrame格式
            output_data = []
            for i, result in enumerate(results):
                row = {
                    'original_text': result['text'],
                    'entities': result['entities'],
                    'entity_count': len(result['entities']),
                    'success': result['success'],
                    'error': result['error']
                }
                output_data.append(row)
            
            output_df = pd.DataFrame(output_data)
            output_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"批量預測完成，結果已保存到: {output_file}")


def create_predictor(model_path: str = None, model=None) -> NERPredictor:
    """
    便捷函數：創建預測器
    
    Args:
        model_path (str): 模型權重路徑
        model: 模型實例
        
    Returns:
        NERPredictor: 預測器實例
    """
    return NERPredictor(model_path, model)


if __name__ == "__main__":
    # 測試預測器
    print("預測器模組測試")
    
    try:
        # 測試文本
        test_texts = [
            "中華統一促進黨總裁「白狼」張安樂及其子張瑋，遭控收受政治獻金卻未按規定申報",
            "今天早上的部會抽籤，讓我很緊張，還好看到的是游騰林跟燊磊一齊出現在中獎名單上",
            "夜店小王子家賢本月的KPI是載實習生妹妹下班，真的鬼"
        ]
        
        # 創建預測器
        predictor = create_predictor()
        
        print("預測器創建成功！")
        
        # 測試預測（如果模型文件存在）
        import os
        if os.path.exists(config.MODEL_SAVE_PATH):
            print("開始測試預測...")
            
            for text in test_texts:
                names = predictor.extract_names(text)
                print(f"文本: {text}")
                print(f"提取的人名: {names}")
                print("-" * 50)
        else:
            print(f"模型文件不存在: {config.MODEL_SAVE_PATH}")
            print("請先訓練模型")
        
        print("預測器測試完成！")
        
    except Exception as e:
        print(f"預測器測試失敗: {str(e)}")
