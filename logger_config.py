import os
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "emol_agent.log")

# 配置参数
MAX_LOG_SIZE = 10 * 1024 * 1024  # 单个日志文件最大 10MB
BACKUP_COUNT = 5  # 保留最多 5 个历史日志文件

def setup_logger(name: str = "EMolAgent") -> logging.Logger:
    """
    设置并返回一个配置好的 logger
    
    特性:
    - 同时输出到控制台和文件
    - 日志文件大小超过 MAX_LOG_SIZE 后自动轮转
    - 最多保留 BACKUP_COUNT 个历史文件
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
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
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


logger = setup_logger()