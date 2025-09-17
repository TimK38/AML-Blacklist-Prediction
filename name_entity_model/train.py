"""
訓練主程序 - NER模型訓練的入口點
Main training script for NER model training
"""

import argparse
import os
import sys
from datetime import datetime

# 添加當前目錄到Python路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from data_loader import create_data_loaders
from model import create_ner_model, ModelManager
from trainer import train_ner_model
from .utils import setup_environment


def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='訓練NER模型')
    
    # 數據相關參數
    parser.add_argument('--data_file', type=str, default=None,
                       help='訓練數據文件路徑')
    
    # 模型相關參數
    parser.add_argument('--model_name', type=str, default=config.MODEL_NAME,
                       help='預訓練模型名稱')
    parser.add_argument('--max_len', type=int, default=config.MAX_LEN,
                       help='最大序列長度')
    
    # 訓練相關參數
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE,
                       help='學習率')
    parser.add_argument('--warmup_steps', type=int, default=config.WARMUP_STEPS,
                       help='預熱步數')
    
    # 保存相關參數
    parser.add_argument('--output_dir', type=str, default=config.MODEL_DIR,
                       help='模型保存目錄')
    parser.add_argument('--save_steps', type=int, default=None,
                       help='每隔多少步保存一次模型')
    parser.add_argument('--save_best_only', action='store_true',
                       help='只保存最佳模型')
    
    # 其他參數
    parser.add_argument('--early_stopping', action='store_true',
                       help='使用早停機制')
    parser.add_argument('--patience', type=int, default=3,
                       help='早停耐心值')
    parser.add_argument('--plot_training', action='store_true',
                       help='繪製訓練曲線')
    parser.add_argument('--log_file', type=str, default=None,
                       help='日誌文件路徑')
    parser.add_argument('--seed', type=int, default=config.RANDOM_STATE,
                       help='隨機種子')
    
    return parser.parse_args()


def update_config_from_args(args):
    """根據命令行參數更新配置"""
    if args.max_len != config.MAX_LEN:
        config.MAX_LEN = args.max_len
    
    if args.batch_size != config.BATCH_SIZE:
        config.BATCH_SIZE = args.batch_size
    
    if args.learning_rate != config.LEARNING_RATE:
        config.LEARNING_RATE = args.learning_rate
    
    if args.epochs != config.EPOCHS:
        config.EPOCHS = args.epochs
    
    if args.warmup_steps != config.WARMUP_STEPS:
        config.WARMUP_STEPS = args.warmup_steps
    
    if args.seed != config.RANDOM_STATE:
        config.RANDOM_STATE = args.seed


def main():
    """主訓練函數"""
    # 解析參數
    args = parse_arguments()
    
    # 更新配置
    update_config_from_args(args)
    
    # 設置環境
    setup_environment()
    
    # 簡單的日誌輸出
    def log_info(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    try:
        # 記錄開始訓練
        log_info("="*50)
        log_info("開始NER模型訓練")
        log_info("="*50)
        
        # 確保輸出目錄存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 1. 加載數據
        log_info("正在加載訓練數據...")
        train_dataloader, valid_dataloader, data_info = create_data_loaders(args.data_file)
        log_info(f"數據加載完成: {data_info}")
        
        # 2. 創建模型
        log_info("正在創建模型...")
        model = create_ner_model(args.model_name)
        log_info("模型創建完成")
        
        # 記錄模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_info(f"模型參數總數: {total_params:,}")
        log_info(f"可訓練參數: {trainable_params:,}")
        
        # 3. 開始訓練
        log_info("開始訓練...")
        trainer, trained_model = train_ner_model(
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            model=model,
            epochs=args.epochs,
            save_best_model=args.save_best_only,
            use_early_stopping=args.early_stopping,
            patience=args.patience
        )
        
        # 4. 保存最終模型
        final_model_path = os.path.join(args.output_dir, "final_model.pth")
        trainer.save_model(final_model_path)
        log_info(f"最終模型已保存到: {final_model_path}")
        
        # 5. 獲取訓練摘要
        training_summary = trainer.get_training_summary()
        log_info(f"訓練摘要: {training_summary}")
        
        # 6. 繪製訓練曲線
        if args.plot_training:
            plot_save_path = os.path.join(args.output_dir, "training_curve.png")
            trainer.plot_training_history(plot_save_path)
            log_info(f"訓練曲線已保存到: {plot_save_path}")
        
        log_info("="*50)
        log_info("訓練完成！")
        log_info(f"最佳F1-Score: {training_summary.get('best_f1_score', 'N/A')}")
        log_info("="*50)
        
    except Exception as e:
        log_info(f"訓練過程中出現錯誤: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
