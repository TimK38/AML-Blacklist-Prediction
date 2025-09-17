# 情感分析模型

基於 BERT 的中文情感分析模型，用於 AML 黑名單預測專案。

## 功能特點

- 基於 BERT-base-chinese 預訓練模型
- 支援二分類情感分析（正面/負面）
- 模組化設計，易於擴展和維護
- 支援批量預測和單文本預測
- 包含完整的訓練和評估流程

## 專案結構

```
sentiment_model/
├── __init__.py          # 套件初始化
├── config.py            # 配置文件
├── dataset.py           # 資料處理模組
├── model.py             # 模型定義
├── trainer.py           # 訓練器
├── predictor.py         # 預測器
├── train.py             # 訓練腳本
├── predict.py           # 預測腳本
├── example.py           # 使用範例
└── README.md            # 說明文件
```

## 安裝依賴

```bash
pip install torch transformers pandas scikit-learn numpy
```

## 快速開始

### 1. 訓練模型

```python
from sentiment_model import train_model, Config

# 使用預設配置訓練
model, history, test_results = train_model()

# 或使用自定義配置
config = Config()
config.EPOCHS = 5
config.BATCH_SIZE = 16
model, history, test_results = train_model(config)
```

### 2. 使用命令行訓練

```bash
python -m sentiment_model.train --epochs 10 --batch_size 16 --learning_rate 2e-5
```

### 3. 預測文本情感

```python
from sentiment_model import SentimentPredictor

# 建立預測器
predictor = SentimentPredictor()

# 預測單個文本
result = predictor.predict_single("這是一個很好的產品")
print(f"預測類別: {result['predicted_class']}")
print(f"信心度: {result['confidence']:.4f}")

# 批量預測
texts = ["文本1", "文本2", "文本3"]
results = predictor.predict_batch(texts)
```

### 4. 使用命令行預測

```bash
# 預測單個文本
python -m sentiment_model.predict --text "要預測的文本"

# 批量預測（從文件讀取）
python -m sentiment_model.predict --input_file texts.txt --output_file results.txt

# 互動模式
python -m sentiment_model.predict
```

## 配置說明

主要配置參數（在 `config.py` 中）：

- `DATA_PATH`: 訓練資料路徑
- `MODEL_SAVE_PATH`: 模型保存路徑
- `MAX_LEN`: 最大序列長度（預設：300）
- `BATCH_SIZE`: 批次大小（預設：16）
- `EPOCHS`: 訓練輪數（預設：10）
- `LEARNING_RATE`: 學習率（預設：2e-5）
- `DROPOUT_RATE`: Dropout 率（預設：0.3）

## 模型架構

- **基礎模型**: BERT-base-chinese
- **分類層**: 線性層 + Dropout
- **損失函數**: CrossEntropyLoss
- **優化器**: AdamW
- **學習率調度**: Linear warmup

## 資料介紹與格式

### 資料集概述

本模型專門針對 **AML（反洗錢）風險評估** 場景設計，主要用於分析與特定人物相關的新聞文章或文本內容，判斷其是否具有潛在的金融風險或負面情感。

### 適用的資料類型

#### ✅ 適合的文本內容：
- **新聞報導**: 關於個人或企業的新聞文章
- **財經資訊**: 股市、金融、投資相關報導  
- **法律案件**: 涉及詐騙、洗錢、金融犯罪的報導
- **企業資訊**: 公司營運、財務狀況相關文本
- **監管公告**: 金管會、檢調單位的相關公告

#### ❌ 不適合的文本內容：
- 一般商品評論（如購物網站評價）
- 電影、餐廳等娛樂評論
- 社交媒體閒聊內容
- 技術文檔或說明書
- 過短的文本（建議至少 20 個字元）

### 資料格式要求

#### CSV 檔案格式
訓練資料必須為 CSV 格式，包含以下必要欄位：

| 欄位名稱 | 資料類型 | 說明 | 範例 |
|---------|---------|------|------|
| `truncated_text` | 字串 | 文本內容（建議 50-300 字） | "富二代網路警察李家賢查獲一間詐騙集團..." |
| `tag` | 整數 | 標籤（0: 負面/有風險, 1: 正面/無風險） | 0 或 1 |

#### 可選欄位：
| 欄位名稱 | 說明 | 用途 |
|---------|------|------|
| `article_index` | 文章編號 | 用於追蹤資料來源 |
| `name` | 相關人名 | 用於分析特定對象 |

### 範例資料

#### 正面樣本（tag = 1）：
```csv
truncated_text,tag,name
"摩根亞太集團董事長李家賢指出，避險基金規模約為5兆美元，採取量化交易的基金規模約1兆美元",1,李家賢
"玉山銀行積極配合政府洗錢防制政策，建立完善的風險管控機制",1,
"該公司財務狀況良好，獲得多家投資機構青睞",1,
```

#### 負面樣本（tag = 0）：
```csv
truncated_text,tag,name  
"檢方認定李家賢以兒子以及嫂子為人頭，詐領每個月8萬元助理費用，共詐領429萬1235元",0,李家賢
"該集團涉嫌透過人頭帳戶進行洗錢活動，金額高達數千萬元",0,
"公司財務造假案遭到檢調搜索，負責人已被約談",0,
```

## 評估指標

模型提供以下評估指標：
- 準確率 (Accuracy)
- 精確率 (Precision)
- 召回率 (Recall)
- F1 分數 (F1-Score)
- 混淆矩陣 (Confusion Matrix)

## 使用範例

查看 `example.py` 文件了解完整的使用範例：

```bash
python sentiment_model/example.py
```

## 注意事項

1. 確保有足夠的 GPU 記憶體進行訓練
2. 訓練時間取決於資料大小和硬體配置
3. 模型會自動保存最佳權重
4. 支援早停機制避免過擬合

## 版本資訊

- 版本: 1.0.0
- Python: 3.7+
- PyTorch: 1.8+
- Transformers: 4.0+
