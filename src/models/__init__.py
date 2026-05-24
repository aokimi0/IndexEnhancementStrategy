"""模型模块。

汇总 alpha 模型与 Stacking 集成框架的公开接口。

注意：在 macOS 上 LightGBM 与 PyTorch 各自携带 OpenMP 运行时，二者顺序错位会引发
``libomp`` 冲突触发段错误。由于 :mod:`src.models.lightgbm_model` 把 ``import lightgbm``
放在函数内部（不能修改源文件），这里在模块加载阶段先显式导入 ``lightgbm`` 让其先初始化
OpenMP 上下文，再加载会引入 :mod:`torch` 的 :mod:`src.models.gru_model`。
"""

import lightgbm as _lightgbm  # noqa: F401  必须先于 torch 导入以避免 OpenMP 冲突

from src.models.base import AlphaModelBase, PredictionResult
from src.models.lightgbm_model import LightgbmAlphaModel, LightgbmPredictionResult
from src.models.xgboost_model import XgboostAlphaModel
from src.models.ridge_model import RidgeAlphaModel
from src.models.gru_model import GRUAlphaModel
from src.models.stacking import StackingAlphaModel

__all__ = [
    "AlphaModelBase",
    "PredictionResult",
    "GRUAlphaModel",
    "LightgbmAlphaModel",
    "LightgbmPredictionResult",
    "RidgeAlphaModel",
    "StackingAlphaModel",
    "XgboostAlphaModel",
]
