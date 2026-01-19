"""
路径管理模块

提供项目路径解析的统一接口，确保无论从哪里运行都能正确找到资源文件。
"""

import os
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def get_project_root() -> str:
    """
    获取项目根目录路径。
    
    查找逻辑：
    1. 优先使用环境变量 EMOLAGENT_ROOT
    2. 向上查找包含 pyproject.toml 的目录
    3. 回退到当前工作目录
    
    Returns:
        项目根目录的绝对路径
    """
    # 1. 环境变量优先
    env_root = os.getenv("EMOLAGENT_ROOT")
    if env_root and os.path.exists(env_root):
        return os.path.abspath(env_root)
    
    # 2. 向上查找 pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return str(parent)
    
    # 3. 回退到当前工作目录
    return os.getcwd()


def get_resource_path(*paths: str) -> str:
    """
    获取 resources 目录下文件的绝对路径。
    
    Args:
        *paths: 相对于 resources 目录的路径片段
        
    Returns:
        完整的绝对路径
        
    Example:
        >>> get_resource_path("models", "nnenv.ep154.pth")
        "/path/to/project/resources/models/nnenv.ep154.pth"
    """
    return os.path.join(get_project_root(), "resources", *paths)


def get_data_path(*paths: str) -> str:
    """
    获取 data 目录下文件的绝对路径。
    
    Args:
        *paths: 相对于 data 目录的路径片段
        
    Returns:
        完整的绝对路径
    """
    return os.path.join(get_project_root(), "data", *paths)


def get_user_workspace(username: str, chat_id: str = "temp") -> str:
    """
    获取用户工作目录。
    
    Args:
        username: 用户名
        chat_id: 会话 ID
        
    Returns:
        用户工作目录的绝对路径
    """
    safe_username = "".join([c for c in (username or "guest") if c.isalnum() or c in ("-", "_")])
    safe_chat_id = str(chat_id or "temp")
    workspace = os.path.join(get_project_root(), "users", safe_username, "output", safe_chat_id)
    os.makedirs(workspace, exist_ok=True)
    return workspace
