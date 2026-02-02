"""
配置管理模块

提供统一的配置加载和访问接口，从 YAML 文件读取配置参数。
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional

import yaml

from emolagent.utils.paths import get_project_root, get_resource_path


def _get_config_path() -> str:
    """
    获取配置文件路径。
    
    查找逻辑：
    1. 优先使用环境变量 EMOL_CONFIG_PATH
    2. 使用项目根目录下的 config/settings.yaml
    
    Returns:
        配置文件的绝对路径
    """
    env_path = os.getenv("EMOL_CONFIG_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    
    return os.path.join(get_project_root(), "config", "settings.yaml")


@lru_cache(maxsize=1)
def _load_config() -> Dict[str, Any]:
    """
    加载并缓存配置文件内容。
    
    Returns:
        配置字典
    """
    config_path = _get_config_path()
    
    if not os.path.exists(config_path):
        # 返回空配置，使用默认值
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def get_config(key: str, default: Any = None) -> Any:
    """
    获取配置值，支持点分隔的嵌套键。
    
    Args:
        key: 配置键，支持点分隔（如 "database.solvent_db"）
        default: 默认值
        
    Returns:
        配置值，如果不存在则返回默认值
        
    Example:
        >>> get_config("database.solvent_db")
        "db/cut_10_common.db"
        >>> get_config("gpu.max_tasks_per_gpu", 2)
        2
    """
    config = _load_config()
    
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value


def reload_config() -> None:
    """
    强制重新加载配置文件。
    
    调用此函数后，下一次 get_config 将重新读取文件。
    """
    _load_config.cache_clear()


# ==========================================
# 便捷访问函数
# ==========================================

class DatabaseConfig:
    """数据库配置"""
    
    @staticmethod
    def get_solvent_db_path() -> str:
        """获取溶剂数据库的完整路径"""
        rel_path = get_config("database.solvent_db", "db/cut_10_common.db")
        return get_resource_path(*rel_path.split('/'))
    
    @staticmethod
    def get_anion_db_path() -> str:
        """获取阴离子数据库的完整路径"""
        rel_path = get_config("database.anion_db", "db/anions.db")
        return get_resource_path(*rel_path.split('/'))


class ModelConfig:
    """模型配置"""
    
    @staticmethod
    def get_inference_model_path() -> str:
        """获取推理模型的完整路径（Li+ 团簇用）"""
        rel_path = get_config("models.inference_model", "models/nnenv.ep154.pth")
        return get_resource_path(*rel_path.split('/'))
    
    @staticmethod
    def get_molecule_inference_model_path() -> str:
        """获取中性小分子推理模型的完整路径"""
        rel_path = get_config("models.molecule_inference_model", "models/nnenv.ep125.pth")
        return get_resource_path(*rel_path.split('/'))
    
    @staticmethod
    def get_uma_checkpoint_path() -> str:
        """获取 UMA 模型检查点的完整路径"""
        rel_path = get_config("models.uma_checkpoint", "models/uma-m-1p1.pt")
        return get_resource_path(*rel_path.split('/'))
    
    @staticmethod
    def get_uma_model_name() -> str:
        """获取 UMA 模型名称"""
        return get_config("models.uma_model_name", "uma-m-1p1")


class GPUConfig:
    """GPU 配置"""
    
    @staticmethod
    def get_available_gpus() -> List[int]:
        """获取可用 GPU 列表"""
        return get_config("gpu.available_gpus", [0, 1])
    
    @staticmethod
    def get_max_tasks_per_gpu() -> int:
        """获取每张 GPU 最大任务数"""
        return get_config("gpu.max_tasks_per_gpu", 2)
    
    @staticmethod
    def get_max_concurrent_tasks() -> int:
        """获取总最大并发任务数"""
        gpus = GPUConfig.get_available_gpus()
        max_per_gpu = GPUConfig.get_max_tasks_per_gpu()
        return max_per_gpu * len(gpus)


class LoggingConfig:
    """日志配置"""
    
    @staticmethod
    def get_max_log_size() -> int:
        """获取单个日志文件最大大小（字节）"""
        return get_config("logging.max_log_size", 10 * 1024 * 1024)
    
    @staticmethod
    def get_backup_count() -> int:
        """获取保留的历史日志文件数量"""
        return get_config("logging.backup_count", 5)


class VisualizationConfig:
    """可视化配置"""
    
    @staticmethod
    def get_max_preview_structures() -> int:
        """获取结构预览时最多显示的分子数量"""
        return get_config("visualization.max_preview_structures", 3)


class AuthConfig:
    """认证配置"""
    
    @staticmethod
    def get_admin_users() -> List[str]:
        """获取管理员用户名列表"""
        return get_config("auth.admin_users", ["hayes"])


class KnowledgeConfig:
    """知识库配置"""
    
    @staticmethod
    def get_literature_path() -> str:
        """获取文献资料库路径"""
        return get_config("knowledge.literature_path", "/home/hayes/projects/ai4mol")
    
    @staticmethod
    def get_collection_name() -> str:
        """获取 ChromaDB 集合名称"""
        return get_config("knowledge.collection_name", "ai4mol_literature")


class MoleculeConfig:
    """分子配置"""
    
    @staticmethod
    def get_default_dme_smiles() -> str:
        """获取默认 DME 溶剂 SMILES"""
        return get_config("molecules.default_dme_smiles", "COCCOC:DME")
    
    @staticmethod
    def get_default_fsi_smiles() -> str:
        """获取默认 FSI 阴离子 SMILES"""
        return get_config("molecules.default_fsi_smiles", "F[S](=O)(=O)[N-][S](=O)(=O)F:FSI")


class OutputConfig:
    """输出配置"""
    
    @staticmethod
    def get_uma_workspace() -> str:
        """获取 UMA 优化器默认输出目录"""
        workspace = get_config("output.uma_workspace", "out_li_clusters")
        return os.path.abspath(workspace)
