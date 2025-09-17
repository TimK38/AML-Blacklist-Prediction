"""
情感分析預測器
"""
import torch
import pandas as pd
from transformers import BertTokenizer
from sklearn.metrics import classification_report, confusion_matrix
from .config import Config
from .model import load_model


class SentimentPredictor:
    """情感分析預測器"""
    
    def __init__(self, model_path=None, config=None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        
        # 載入分詞器
        self.tokenizer = BertTokenizer.from_pretrained(self.config.MODEL_NAME)
        
        # 載入模型
        model_path = model_path or self.config.MODEL_SAVE_PATH
        self.model = load_model(model_path, self.config)
        
        print(f"模型已載入: {model_path}")
        print(f"使用設備: {self.device}")
    
    def predict_single(self, text):
        """預測單個文本的情感"""
        # 編碼文本
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config.MAX_LEN,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # 移動到設備
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # 預測
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            _, prediction = torch.max(outputs, dim=1)
        
        # 返回結果
        pred_class = self.config.CLASS_NAMES[prediction.item()]
        confidence = probabilities[0][prediction.item()].item()
        
        return {
            'text': text,
            'predicted_class': pred_class,
            'confidence': confidence,
            'probabilities': {
                self.config.CLASS_NAMES[i]: probabilities[0][i].item() 
                for i in range(len(self.config.CLASS_NAMES))
            }
        }
    
    def predict_batch(self, texts):
        """批量預測文本情感"""
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        return results
    
    def evaluate_model(self, data_loader):
        """評估模型性能"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_texts = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 移動數據到設備
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["targets"].to(self.device)
                texts = batch["review_text"]
                
                # 預測
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, dim=1)
                
                # 收集結果
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_texts.extend(texts)
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return {
            'predictions': all_predictions,
            'targets': all_targets,
            'texts': all_texts,
            'probabilities': all_probabilities
        }
    
    def print_evaluation_report(self, data_loader):
        """打印評估報告"""
        results = self.evaluate_model(data_loader)
        
        # 分類報告
        report = classification_report(
            results['targets'], 
            results['predictions'], 
            target_names=self.config.CLASS_NAMES,
            digits=4
        )
        print("分類報告:")
        print(report)
        
        # 混淆矩陣
        cm = confusion_matrix(results['targets'], results['predictions'])
        print("\n混淆矩陣:")
        print(f"{'':>10} {'Predicted':>20}")
        print(f"{'Actual':>10} {self.config.CLASS_NAMES[0]:>10} {self.config.CLASS_NAMES[1]:>10}")
        for i, class_name in enumerate(self.config.CLASS_NAMES):
            print(f"{class_name:>10} {cm[i][0]:>10} {cm[i][1]:>10}")
        
        # 準確率
        accuracy = sum(p == t for p, t in zip(results['predictions'], results['targets'])) / len(results['targets'])
        print(f"\n整體準確率: {accuracy:.4f}")
        
        return results
    
    def save_predictions(self, data_loader, output_path):
        """保存預測結果到CSV文件"""
        results = self.evaluate_model(data_loader)
        
        # 創建DataFrame
        df = pd.DataFrame({
            'text': results['texts'],
            'true_label': [self.config.CLASS_NAMES[t] for t in results['targets']],
            'predicted_label': [self.config.CLASS_NAMES[p] for p in results['predictions']],
            'negative_prob': [prob[0] for prob in results['probabilities']],
            'positive_prob': [prob[1] for prob in results['probabilities']],
            'correct': [p == t for p, t in zip(results['predictions'], results['targets'])]
        })
        
        # 保存到CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"預測結果已保存到: {output_path}")
        
        return df
