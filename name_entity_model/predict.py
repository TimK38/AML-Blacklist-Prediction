"""
預測主程序 - NER模型預測的入口點
Main prediction script for NER model inference
"""

import argparse
import os
import sys
import json
from datetime import datetime

# 添加當前目錄到Python路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from predictor import create_predictor, BatchPredictor
# 簡化日誌功能


def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='使用訓練好的NER模型進行預測')
    
    # 模型相關參數
    parser.add_argument('--model_path', type=str, default=config.MODEL_SAVE_PATH,
                       help='訓練好的模型權重路徑')
    
    # 輸入相關參數
    parser.add_argument('--text', type=str, default=None,
                       help='要預測的單個文本')
    parser.add_argument('--input_file', type=str, default=None,
                       help='包含多個文本的輸入文件')
    parser.add_argument('--text_column', type=str, default='text',
                       help='輸入文件中文本列的名稱')
    
    # 輸出相關參數
    parser.add_argument('--output_file', type=str, default=None,
                       help='預測結果輸出文件路徑')
    parser.add_argument('--output_format', type=str, choices=['json', 'csv'], default='json',
                       help='輸出格式')
    
    # 預測相關參數
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批量預測的批次大小')
    parser.add_argument('--return_confidence', action='store_true',
                       help='返回預測置信度')
    parser.add_argument('--return_positions', action='store_true',
                       help='返回實體位置信息')
    parser.add_argument('--names_only', action='store_true',
                       help='只返回人名列表（簡化輸出）')
    
    # 其他參數
    parser.add_argument('--log_file', type=str, default=None,
                       help='日誌文件路徑')
    parser.add_argument('--verbose', action='store_true',
                       help='詳細輸出')
    
    return parser.parse_args()


def predict_single_text(predictor, text, args):
    """預測單個文本"""
    try:
        if args.names_only:
            # 只返回人名列表
            names = predictor.extract_names(text)
            result = {
                'text': text,
                'names': names,
                'count': len(names)
            }
        else:
            # 返回完整實體信息
            entities = predictor.extract_entities(text, return_positions=args.return_positions)
            result = {
                'text': text,
                'entities': entities,
                'entity_count': len(entities)
            }
        
        result['success'] = True
        result['error'] = None
        
        return result
        
    except Exception as e:
        return {
            'text': text,
            'entities': [] if not args.names_only else [],
            'names': [] if args.names_only else None,
            'success': False,
            'error': str(e)
        }


def predict_batch_file(predictor, input_file, output_file, args, log_func):
    """批量預測文件中的文本"""
    try:
        # 創建批量預測器
        batch_predictor = BatchPredictor(predictor)
        
        # 執行批量預測
        log_func(f"開始批量預測文件: {input_file}")
        batch_predictor.predict_from_file(
            input_file=input_file,
            output_file=output_file,
            text_column=args.text_column,
            batch_size=args.batch_size
        )
        
        log_func(f"批量預測完成，結果已保存到: {output_file}")
        return True
        
    except Exception as e:
        print(f"批量預測失敗: {str(e)}")
        return False


def save_result(result, output_file, output_format):
    """保存預測結果"""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if output_format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        elif output_format == 'csv':
            import pandas as pd
            
            # 將結果轉換為DataFrame格式
            if isinstance(result, list):
                df = pd.DataFrame(result)
            else:
                df = pd.DataFrame([result])
            
            df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"預測結果已保存到: {output_file}")
        return True
        
    except Exception as e:
        print(f"保存結果失敗: {str(e)}")
        return False


def main():
    """主預測函數"""
    # 解析參數
    args = parse_arguments()
    
    # 簡單的日誌輸出
    def log_info(message):
        if args.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    try:
        # 記錄開始預測
        log_info("="*50)
        log_info("開始NER模型預測")
        log_info("="*50)
        log_info(f"模型路徑: {args.model_path}")
        
        # 檢查模型文件是否存在
        if not os.path.exists(args.model_path):
            print(f"錯誤: 模型文件不存在: {args.model_path}")
            return
        
        # 創建預測器
        log_info("正在加載模型...")
        
        predictor = create_predictor(args.model_path)
        
        if args.verbose:
            model_info = predictor.get_model_info()
            log_info(f"模型信息: {model_info}")
        
        # 根據輸入類型進行預測
        if args.text:
            # 單個文本預測
            log_info(f"預測單個文本: {args.text}")
            
            result = predict_single_text(predictor, args.text, args)
            
            # 打印結果
            if args.names_only and result['success']:
                print(f"識別的人名: {result['names']}")
            elif result['success']:
                print(f"識別的實體: {result['entities']}")
            else:
                print(f"預測失敗: {result['error']}")
            
            # 保存結果（如果指定了輸出文件）
            if args.output_file:
                save_result(result, args.output_file, args.output_format)
        
        elif args.input_file:
            # 批量文件預測
            if not os.path.exists(args.input_file):
                print(f"錯誤: 輸入文件不存在: {args.input_file}")
                return
            
            if not args.output_file:
                # 自動生成輸出文件名
                base_name = os.path.splitext(args.input_file)[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                args.output_file = f"{base_name}_predictions_{timestamp}.{args.output_format}"
            
            success = predict_batch_file(predictor, args.input_file, args.output_file, args, log_info)
            
            if not success:
                print("錯誤: 批量預測失敗")
                return
        
        else:
            # 交互式預測
            print("進入交互式預測模式（輸入 'quit' 退出）:")
            
            while True:
                try:
                    text = input("\n請輸入要預測的文本: ").strip()
                    
                    if text.lower() in ['quit', 'exit', '退出']:
                        break
                    
                    if not text:
                        continue
                    
                    result = predict_single_text(predictor, text, args)
                    
                    if result['success']:
                        if args.names_only:
                            print(f"識別的人名: {result['names']}")
                        else:
                            print(f"識別的實體:")
                            for entity in result['entities']:
                                print(f"  - {entity}")
                    else:
                        print(f"預測失敗: {result['error']}")
                
                except KeyboardInterrupt:
                    print("\n退出交互式模式")
                    break
                except Exception as e:
                    print(f"預測出錯: {str(e)}")
        
        log_info("="*50)
        log_info("預測完成！")
        log_info("="*50)
    
    except Exception as e:
        print(f"預測過程中出現錯誤: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
