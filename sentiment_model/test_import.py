"""
測試模組引入是否正常
"""

def test_imports():
    """測試所有模組是否可以正常引入"""
    try:
        from sentiment_model.config import Config
        print("✓ Config 引入成功")
        
        from sentiment_model.dataset import DataProcessor, SentimentDataset
        print("✓ Dataset 模組引入成功")
        
        from sentiment_model.model import SentimentClassifier, create_model
        print("✓ Model 模組引入成功")
        
        from sentiment_model.trainer import SentimentTrainer
        print("✓ Trainer 模組引入成功")
        
        from sentiment_model.predictor import SentimentPredictor
        print("✓ Predictor 模組引入成功")
        
        from sentiment_model.train import train_model
        print("✓ Train 模組引入成功")
        
        # 測試配置
        config = Config()
        print(f"✓ 配置測試成功 - 設備: {config.DEVICE}")
        print(f"✓ 資料路徑: {config.DATA_PATH}")
        print(f"✓ 模型保存路徑: {config.MODEL_SAVE_PATH}")
        
        print("\n所有模組引入測試通過！")
        return True
        
    except ImportError as e:
        print(f"✗ 引入錯誤: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他錯誤: {e}")
        return False


if __name__ == "__main__":
    print("開始測試模組引入...")
    print("=" * 40)
    test_imports()
