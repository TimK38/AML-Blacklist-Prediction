# AML-Blacklist-Prediction

玉山人工智慧公開挑戰賽2020夏季賽 - NLP應用挑戰賽解決方案

## 🎯 專案背景

本專案是針對[玉山人工智慧公開挑戰賽2020夏季賽 - NLP應用挑戰賽](https://tbrain.trendmicro.com.tw/Competitions/Details/11)所開發的解決方案。比賽目標是建立一個**AML（反洗錢）黑名單預測系統**，透過分析新聞文章內容，判斷特定人物是否具有潛在的金融風險。

## 🔧 技術方案

本專案採用**雙模型架構**來解決AML風險預測問題：

### 1. 命名實體識別 (NER)
- **目的**: 從新聞文章中準確識別人名
- **技術**: 基於BERT的中文命名實體識別模型
- **功能**: 提取文章中的關鍵人物資訊

### 2. 情感分析 (Sentiment Analysis)  
- **目的**: 分析與特定人物相關的新聞內容情感傾向
- **技術**: 基於BERT-base-chinese的二分類模型
- **功能**: 判斷新聞內容是否暗示潛在風險（正面/負面）

## 📁 專案結構

```
AML-Blacklist-Prediction/
├── name_entity_model/          # NER模型相關代碼
│   ├── README.md              # NER模型詳細說明
│   ├── model.py               # NER模型定義
│   ├── train.py               # NER訓練腳本
│   └── predict.py             # NER預測腳本
├── sentiment_model/            # 情感分析模型相關代碼
│   ├── README.md              # 情感分析模型詳細說明
│   ├── config.py              # 模型配置
│   ├── model.py               # 情感分析模型定義
│   ├── train.py               # 訓練腳本
│   └── predict.py             # 預測腳本
├── data/                       # 訓練資料
│   ├── Sentiment_model_data/   # 情感分析訓練資料
│   └── 亞洲研究院數據集/        # NER訓練資料
├── model/                      # 訓練好的模型權重
├── OriginCode/                 # 原始Jupyter Notebook代碼
└── demo_*.ipynb               # 使用範例
```

## 🚀 快速開始

### 環境需求
- Python 3.7+
- PyTorch 1.8+
- Transformers 4.0+
- 其他依賴請參考各模組的 `requirements.txt`

### 使用流程

#### 1. 命名實體識別 (NER)
```bash
# 查看NER模型詳細說明
cd name_entity_model/
cat README.md

# 訓練NER模型
python train.py

# 使用NER模型預測
python predict.py --text "李家賢涉嫌詐騙案件"
```

#### 2. 情感分析
```bash
# 查看情感分析模型詳細說明  
cd sentiment_model/
cat README.md

# 訓練情感分析模型
python train.py

# 使用情感分析模型預測
python predict.py --text "該公司財務狀況良好"
```

#### 3. 整合使用
```bash
# 查看完整使用範例
jupyter notebook demo_ner.ipynb
jupyter notebook demo_sentiment.ipynb
```

## 📖 詳細文檔

每個模組都有詳細的說明文檔，請參考：

- **[NER模型說明](name_entity_model/README.md)** - 命名實體識別模型的訓練和使用
- **[情感分析模型說明](sentiment_model/README.md)** - 情感分析模型的訓練和使用

## 🎪 Demo範例

- `demo_ner.ipynb` - NER模型使用範例
- `demo_sentiment.ipynb` - 情感分析模型使用範例

## 💡 解決方案亮點

1. **雙模型協同**: NER + 情感分析的組合策略
2. **中文優化**: 針對中文新聞文本進行優化
3. **模組化設計**: 每個模型都可以獨立使用
4. **完整工具鏈**: 從資料預處理到模型部署的完整流程
5. **實戰驗證**: 在玉山競賽中取得優異成績

## 🏆 競賽策略

本解決方案的核心策略：
1. **精準識別**: 使用NER模型準確提取人名
2. **風險評估**: 透過情感分析判斷新聞內容的風險程度  
3. **特徵融合**: 結合實體識別和情感特徵進行綜合判斷
4. **資料增強**: 針對AML領域進行專門的資料處理和增強

---