"""
使用訓練好的模型進行預測的腳本
"""
import argparse
from .config import Config
from .predictor import SentimentPredictor


def predict_text(text, model_path=None):
    """預測單個文本的情感"""
    config = Config()
    predictor = SentimentPredictor(model_path, config)
    
    result = predictor.predict_single(text)
    
    print("=" * 60)
    print("情感分析預測結果")
    print("=" * 60)
    print(f"輸入文本: {result['text']}")
    print(f"預測類別: {result['predicted_class']}")
    print(f"信心度: {result['confidence']:.4f}")
    print("\n各類別機率:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")
    
    return result


def predict_batch_from_file(input_file, output_file=None, model_path=None):
    """從文件讀取文本進行批量預測"""
    config = Config()
    predictor = SentimentPredictor(model_path, config)
    
    # 讀取文本
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"從 {input_file} 讀取了 {len(texts)} 條文本")
    
    # 批量預測
    results = predictor.predict_batch(texts)
    
    # 輸出結果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("文本\t預測類別\t信心度\t負面機率\t正面機率\n")
            for result in results:
                f.write(f"{result['text']}\t{result['predicted_class']}\t"
                       f"{result['confidence']:.4f}\t"
                       f"{result['probabilities']['negative']:.4f}\t"
                       f"{result['probabilities']['positive']:.4f}\n")
        print(f"預測結果已保存到: {output_file}")
    else:
        print("\n預測結果:")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['text'][:50]}...")
            print(f"   預測: {result['predicted_class']} (信心度: {result['confidence']:.4f})")
            print()
    
    return results


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='使用情感分析模型進行預測')
    parser.add_argument('--text', type=str, help='要預測的文本')
    parser.add_argument('--input_file', type=str, help='包含要預測文本的文件路径')
    parser.add_argument('--output_file', type=str, help='預測結果輸出文件路径')
    parser.add_argument('--model_path', type=str, help='模型文件路径')
    
    args = parser.parse_args()
    
    if args.text:
        # 預測單個文本
        predict_text(args.text, args.model_path)
    elif args.input_file:
        # 批量預測
        predict_batch_from_file(args.input_file, args.output_file, args.model_path)
    else:
        # 互動模式
        print("情感分析預測器 - 互動模式")
        print("輸入 'quit' 結束程序")
        print("-" * 40)
        
        config = Config()
        predictor = SentimentPredictor(args.model_path, config)
        
        while True:
            text = input("\n請輸入要分析的文本: ").strip()
            if text.lower() == 'quit':
                break
            if text:
                result = predictor.predict_single(text)
                print(f"預測類別: {result['predicted_class']}")
                print(f"信心度: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
