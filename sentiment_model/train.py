"""
訓練情感分析模型的主要腳本
"""
import argparse
import os
from .config import Config
from .dataset import DataProcessor
from .model import create_model
from .trainer import SentimentTrainer
from .predictor import SentimentPredictor


def train_model(config=None):
    """訓練模型的主要函數"""
    config = config or Config()
    
    print("=" * 60)
    print("情感分析模型訓練")
    print("=" * 60)
    
    # 1. 準備資料
    print("\n1. 準備資料...")
    data_processor = DataProcessor(config)
    train_loader, val_loader, test_loader = data_processor.prepare_data()
    
    # 2. 建立模型
    print("\n2. 建立模型...")
    model = create_model(config)
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 訓練模型
    print("\n3. 開始訓練...")
    trainer = SentimentTrainer(model, config)
    history = trainer.train(train_loader, val_loader)
    
    # 4. 評估模型
    print("\n4. 評估模型...")
    predictor = SentimentPredictor(config.MODEL_SAVE_PATH, config)
    
    print("\n=== 測試集評估結果 ===")
    test_results = predictor.print_evaluation_report(test_loader)
    
    print("\n訓練完成！")
    return model, history, test_results


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='訓練情感分析模型')
    parser.add_argument('--data_path', type=str, help='訓練資料路徑')
    parser.add_argument('--model_save_path', type=str, help='模型保存路徑')
    parser.add_argument('--epochs', type=int, help='訓練輪數')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--learning_rate', type=float, help='學習率')
    parser.add_argument('--max_len', type=int, help='最大序列長度')
    
    args = parser.parse_args()
    
    # 建立配置
    config = Config()
    
    # 根據命令行參數更新配置
    if args.data_path:
        config.DATA_PATH = args.data_path
    if args.model_save_path:
        config.MODEL_SAVE_PATH = args.model_save_path
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    if args.max_len:
        config.MAX_LEN = args.max_len
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    
    # 訓練模型
    train_model(config)


if __name__ == "__main__":
    main()
