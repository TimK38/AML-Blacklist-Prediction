"""
情感分析模型使用範例
"""
from sentiment_model import Config, train_model, SentimentPredictor


def example_training():
    """訓練模型的範例"""
    print("=== 訓練模型範例 ===")
    
    # 使用預設配置訓練模型
    config = Config()
    
    # 如果需要，可以修改配置
    config.EPOCHS = 5  # 減少訓練輪數以快速測試
    config.BATCH_SIZE = 8  # 減少批次大小以節省記憶體
    
    # 訓練模型
    model, history, test_results = train_model(config)
    
    print("訓練完成！")
    return model, history


def example_prediction():
    """預測的範例"""
    print("\n=== 預測範例 ===")
    
    # 建立預測器
    predictor = SentimentPredictor()
    
    # 測試文本
    test_texts = [
        "富二代網路警察李家賢查獲一間詐騙集團，協助玉山銀行進行洗錢防制",
        "李家賢涉嫌詐領助理費用，被檢方起訴",
        "這家公司的服務品質很好，我很滿意",
        "股票投資風險很高，需要謹慎評估"
    ]
    
    # 批量預測
    results = predictor.predict_batch(test_texts)
    
    # 顯示結果
    for i, result in enumerate(results, 1):
        print(f"\n{i}. 文本: {result['text']}")
        print(f"   預測: {result['predicted_class']}")
        print(f"   信心度: {result['confidence']:.4f}")
        print(f"   機率分佈: {result['probabilities']}")


def example_custom_config():
    """自定義配置的範例"""
    print("\n=== 自定義配置範例 ===")
    
    # 建立自定義配置
    config = Config()
    config.MAX_LEN = 200  # 修改最大序列長度
    config.BATCH_SIZE = 32  # 修改批次大小
    config.LEARNING_RATE = 1e-5  # 修改學習率
    config.EPOCHS = 3  # 修改訓練輪數
    
    print(f"最大序列長度: {config.MAX_LEN}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print(f"學習率: {config.LEARNING_RATE}")
    print(f"訓練輪數: {config.EPOCHS}")
    
    # 使用自定義配置進行預測
    predictor = SentimentPredictor(config=config)
    
    # 測試單個文本預測
    text = "這是一個測試文本"
    result = predictor.predict_single(text)
    print(f"\n預測結果: {result}")


if __name__ == "__main__":
    print("情感分析模型使用範例")
    print("=" * 50)
    
    # 如果模型還沒有訓練，先訓練一個
    try:
        # 嘗試載入已存在的模型
        predictor = SentimentPredictor()
        print("找到已訓練的模型，直接進行預測範例")
        example_prediction()
    except:
        print("沒有找到已訓練的模型，開始訓練...")
        # 訓練模型
        example_training()
        # 然後進行預測
        example_prediction()
    
    # 顯示自定義配置範例
    example_custom_config()
