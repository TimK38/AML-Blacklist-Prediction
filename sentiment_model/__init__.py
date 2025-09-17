"""
情感分析模型套件

這個套件包含了基於BERT的中文情感分析模型的完整實現，
包括資料處理、模型定義、訓練和預測功能。
"""

from .config import Config
from .dataset import DataProcessor, SentimentDataset
from .model import SentimentClassifier, create_model, load_model
from .trainer import SentimentTrainer
from .predictor import SentimentPredictor
from .train import train_model
from .data_validator import DataValidator

__version__ = "1.0.0"
__author__ = "AML Blacklist Prediction Team"

__all__ = [
    'Config',
    'DataProcessor',
    'SentimentDataset',
    'SentimentClassifier',
    'create_model',
    'load_model',
    'SentimentTrainer',
    'SentimentPredictor',
    'train_model',
    'DataValidator'
]
