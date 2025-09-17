"""
NumPy 版本兼容性處理模組
NumPy version compatibility handling module
"""

import numpy as np
import warnings

# 處理 NumPy 版本兼容性問題
def get_numpy_inf():
    """獲取 NumPy 的無窮大值，兼容不同版本"""
    try:
        # NumPy 2.0+ 使用 np.inf
        return np.inf
    except AttributeError:
        # 舊版本 NumPy 使用 np.Inf
        return np.Inf

def get_numpy_nan():
    """獲取 NumPy 的 NaN 值，兼容不同版本"""
    try:
        # NumPy 2.0+ 使用 np.nan
        return np.nan
    except AttributeError:
        # 舊版本 NumPy 使用 np.NaN
        return np.NaN

# 導出兼容性常數
NUMPY_INF = get_numpy_inf()
NUMPY_NAN = get_numpy_nan()

# 抑制相關警告
warnings.filterwarnings("ignore", message=".*np.Inf.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*np.NaN.*", category=DeprecationWarning)
