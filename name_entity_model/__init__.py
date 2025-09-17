"""
NER (Named Entity Recognition) 模組
用於中文命名實體識別，特別是人名識別

主要功能：
- 基於BERT的中文NER模型
- 支持訓練和預測
- 提供完整的數據處理管道
- 支持批量處理和交互式預測

使用示例：
    # 訓練模型
    from name_entity_model import train_ner_model, create_data_loaders, create_ner_model
    
    train_dl, valid_dl, _ = create_data_loaders()
    model = create_ner_model()
    trainer, trained_model = train_ner_model(train_dl, valid_dl, model)
    
    # 預測
    from name_entity_model import create_predictor
    
    predictor = create_predictor("path/to/model.pth")
    names = predictor.extract_names("張安樂是中華統一促進黨總裁")
    print(names)  # ['張安樂']

版本信息：
    __version__ = "1.0.0"
    __author__ = "Your Name"
    __email__ = "your.email@example.com"
"""

__version__ = "1.0.0"
__author__ = "NER Team"
__email__ = "ner@example.com"

# 導入主要類和函數
from .config import config, Config
from .data_loader import NERDataLoader, create_data_loaders
from .model import NERModel, ModelManager, create_ner_model, load_trained_model
from .trainer import NERTrainer, train_ner_model
from .predictor import NERPredictor, BatchPredictor, create_predictor
from .utils import (
    EarlyStopping,
    tokenize_and_preserve_labels,
    cut_tag_encoding,
    extract_entities_from_tags,
    setup_environment
)
from . import numpy_compat  # 確保兼容性處理被加載

# 定義公共API
__all__ = [
    # 配置
    'config',
    'Config',
    
    # 數據處理
    'NERDataLoader',
    'create_data_loaders',
    
    # 模型
    'NERModel',
    'ModelManager', 
    'create_ner_model',
    'load_trained_model',
    
    # 訓練
    'NERTrainer',
    'train_ner_model',
    
    # 預測
    'NERPredictor',
    'BatchPredictor',
    'create_predictor',
    
    # 工具
    'EarlyStopping',
    'tokenize_and_preserve_labels',
    'cut_tag_encoding',
    'extract_entities_from_tags',
    'setup_environment',
    
    # 版本信息
    '__version__',
    '__author__',
    '__email__'
]

# 模組級別的便捷函數
def get_version():
    """獲取版本信息"""
    return __version__

def get_info():
    """獲取模組信息"""
    return {
        'name': 'name_entity_model',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': '中文命名實體識別模組，基於BERT實現'
    }

# 初始化時的設置
def _initialize():
    """模組初始化"""
    try:
        # 確保必要的目錄存在
        config.ensure_dirs()
        
        # 可以在這裡添加其他初始化邏輯
        pass
        
    except Exception as e:
        import warnings
        warnings.warn(f"模組初始化時出現警告: {str(e)}", UserWarning)

# 執行初始化
_initialize()
