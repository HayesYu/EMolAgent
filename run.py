#!/usr/bin/env python
"""
EMolAgent 启动脚本

这个脚本提供了一个简单的方式来启动 EMolAgent Streamlit 应用。
"""

import subprocess
import sys
import os

def main():
    """启动 EMolAgent Streamlit 应用。"""
    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 设置环境变量，确保包可以被找到
    os.environ["EMOLAGENT_ROOT"] = project_root
    
    # 添加 src 到 Python 路径
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # 启动 Streamlit
    app_path = os.path.join(src_path, "emolagent", "app.py")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        app_path,
        "--server.port=8501",
        "--server.address=0.0.0.0",
    ]
    
    print(f"🧪 正在启动 EMolAgent...")
    print(f"📁 项目根目录: {project_root}")
    print(f"🌐 访问地址: http://localhost:8501")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 EMolAgent 已停止")
        sys.exit(0)


if __name__ == "__main__":
    main()
