"""
日志配置模块

提供全局统一的日志配置，支持控制台输出和文件轮转。
"""

import os
import logging
from logging.handlers import RotatingFileHandler

from emolagent.utils.paths import get_project_root


def _get_logging_config():
    """
    获取日志配置参数。
    
    注意：这里不能直接导入 config 模块，因为 logger 是最早被导入的模块之一，
    可能会造成循环导入。因此直接读取 YAML 配置。
    """
    import yaml
    
    config_path = os.path.join(get_project_root(), "config", "settings.yaml")
    
    # 默认值
    defaults = {
        "max_log_size": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5
    }
    
    if not os.path.exists(config_path):
        return defaults
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
            logging_config = config.get("logging", {})
            return {
                "max_log_size": logging_config.get("max_log_size", defaults["max_log_size"]),
                "backup_count": logging_config.get("backup_count", defaults["backup_count"])
            }
    except Exception:
        return defaults


# 日志目录使用项目根目录下的 logs 文件夹
LOG_DIR = os.path.join(get_project_root(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "emol_agent.log")

# 配置参数（从配置文件加载）
_logging_config = _get_logging_config()
MAX_LOG_SIZE = _logging_config["max_log_size"]
BACKUP_COUNT = _logging_config["backup_count"]


def setup_logger(name: str = "EMolAgent") -> logging.Logger:
    """
    设置并返回一个配置好的 logger
    
    特性:
    - 同时输出到控制台和文件
    - 日志文件大小超过 MAX_LOG_SIZE 后自动轮转
    - 最多保留 BACKUP_COUNT 个历史文件
    
    Args:
        name: logger 名称
        
    Returns:
        配置好的 Logger 实例
    """
    _logger = logging.getLogger(name)
    
    # 避免重复添加 handler
    if _logger.handlers:
        return _logger
    
    _logger.setLevel(logging.DEBUG)
    
    # 日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 1. 控制台 Handler (INFO 及以上)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 2. 文件 Handler (带轮转)
    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    
    return _logger


# 全局 logger 实例
logger = setup_logger()
