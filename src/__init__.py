# 对外暴露核心模型类，便于从 src 包直接 import。
from .model import BranchConfig, DualBranchAPIRec, ExplicitModel, HybridRecSys, ImplicitModel, LightGCNEncoder

__all__ = [
    "BranchConfig",
    "DualBranchAPIRec",
    "ExplicitModel",
    "HybridRecSys",
    "ImplicitModel",
    "LightGCNEncoder",
]
