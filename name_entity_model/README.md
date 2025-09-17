# NER (Named Entity Recognition) 中文命名實體識別模組

## 📋 專案簡介

這是一個基於BERT的中文命名實體識別（NER）模組，專門用於識別中文文本中的人名。該專案將原本的Jupyter Notebook重構為標準的Python模組，提供了完整的訓練和預測管道。

### ✨ 主要特性

- 🤖 基於BERT的深度學習模型
- 🎯 專門針對中文人名識別優化
- 📊 完整的訓練和評估流程
- 🚀 支持批量預測和交互式預測
- 📈 提供訓練可視化和指標監控
- 🔧 模組化設計，易於擴展和維護

## 🏗️ 專案結構

```
name_entity_model/
├── __init__.py           # 模組初始化文件
├── config.py             # 配置文件
├── data_loader.py        # 數據加載和預處理
├── model.py              # 模型定義
├── trainer.py            # 訓練邏輯
├── predictor.py          # 預測邏輯
├── utils.py              # 工具函數
├── train.py              # 訓練主程序
├── predict.py            # 預測主程序
├── requirements.txt      # 依賴文件
└── README.md            # 說明文檔
```

## 🚀 快速開始

### 1. 環境安裝

```bash
# 安裝依賴
pip install -r requirements.txt
```

### 2. 準備數據

確保你的數據文件位於正確的位置：
```
data/亞洲研究院數據集/asia_institute.csv
```

數據格式應包含 `words` 和 `tags_encoding` 列。

### 3. 訓練模型

#### 使用命令行訓練

```bash
# 基本訓練
python train.py

# 自定義參數訓練
python train.py --epochs 5 --batch_size 16 --learning_rate 2e-5 --early_stopping --plot_training
```

#### 使用Python代碼訓練

```python
from name_entity_model import train_ner_model, create_data_loaders, create_ner_model

# 加載數據
train_dl, valid_dl, data_info = create_data_loaders()

# 創建模型
model = create_ner_model()

# 訓練
trainer, trained_model = train_ner_model(
    train_dataloader=train_dl,
    valid_dataloader=valid_dl,
    model=model,
    epochs=3
)
```

### 4. 使用模型預測

#### 命令行預測

```bash
# 單個文本預測
python predict.py --text "張安樂是中華統一促進黨總裁"

# 批量文件預測
python predict.py --input_file input.csv --output_file output.json

# 交互式預測
python predict.py
```

#### Python代碼預測

```python
from name_entity_model import create_predictor

# 創建預測器
predictor = create_predictor("model/ner_model_weights.pth")

# 預測人名
text = "張安樂及其子張瑋，遭控收受政治獻金"
names = predictor.extract_names(text)
print(names)  # ['張安樂', '張瑋']

# 獲取完整實體信息
entities = predictor.extract_entities(text, return_positions=True)
print(entities)
```

## 📖 詳細使用說明

### 配置參數

主要配置參數在 `config.py` 中定義：

```python
# 模型配置
MODEL_NAME = "bert-base-chinese"
MAX_LEN = 75
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 3e-5

# 標籤配置
TAG_VALUES = ['O', 'B_person_name', 'M_person_name', 'E_person_name', 'PAD']
```

### 訓練參數

訓練腳本支持的主要參數：

```bash
python train.py --help

optional arguments:
  --data_file           訓練數據文件路徑
  --model_name          預訓練模型名稱
  --epochs              訓練輪數
  --batch_size          批次大小
  --learning_rate       學習率
  --early_stopping      使用早停機制
  --plot_training       繪製訓練曲線
  --output_dir          模型保存目錄
```

### 預測參數

預測腳本支持的主要參數：

```bash
python predict.py --help

optional arguments:
  --model_path          訓練好的模型權重路徑
  --text                要預測的單個文本
  --input_file          包含多個文本的輸入文件
  --output_file         預測結果輸出文件路徑
  --names_only          只返回人名列表
  --batch_size          批量預測的批次大小
```

## 📊 模型性能

基於亞洲研究院數據集的測試結果：

- **準確率 (Accuracy)**: ~99.8%
- **F1分數**: ~97.2%
- **訓練時間**: ~9分鐘（3個epoch，RTX 4070 SUPER）

## 🐛 故障排除

### 常見問題

1. **CUDA out of memory**
   - 減少 `BATCH_SIZE` 參數
   - 減少 `MAX_LEN` 參數

2. **模型文件不存在**
   - 確保模型已經訓練完成
   - 檢查模型路徑是否正確

3. **數據格式錯誤**
   - 確保CSV文件包含 `words` 和 `tags_encoding` 列
   - 檢查數據是否為正確的列表格式


## 📄 授權

本專案使用 MIT 授權條款。

### 開發環境設置

```bash
# Clone project
git clone <repository-url>
cd name_entity_model

# 安裝開發依賴
pip install -r requirements.txt

```