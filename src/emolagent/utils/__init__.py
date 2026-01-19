"""
工具模块 - 日志、路径处理等通用工具函数
"""

from emolagent.utils.logger import logger, setup_logger
from emolagent.utils.paths import get_project_root, get_resource_path, get_data_path

__all__ = [
    "logger",
    "setup_logger", 
    "get_project_root",
    "get_resource_path",
    "get_data_path",
]
