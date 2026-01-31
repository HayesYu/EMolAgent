#!/bin/bash
# EMolAgent Docker 入口脚本
# 负责环境检查、目录初始化和应用启动

set -e

echo "============================================"
echo "  EMolAgent Docker Container Starting..."
echo "============================================"

# 设置 ulimit（Multiwfn 需要）
ulimit -s unlimited 2>/dev/null || echo "Warning: Could not set ulimit"

# 确保运行时目录存在
echo ""
echo "=== Creating runtime directories ==="
mkdir -p /app/data /app/users /app/logs /app/.task_slots /app/data/chroma_db
echo "OK: Runtime directories created"

# ============================================
# 检查必需文件
# ============================================
echo ""
echo "=== Checking required files ==="

# 检查模型文件
MODEL_DIR="/app/resources/models"
MISSING_MODELS=0

if [ ! -f "${MODEL_DIR}/uma-m-1p1.pt" ]; then
    echo "ERROR: Model file not found: ${MODEL_DIR}/uma-m-1p1.pt"
    MISSING_MODELS=1
fi

if [ ! -f "${MODEL_DIR}/nnenv.ep154.pth" ]; then
    echo "WARNING: Model file not found: ${MODEL_DIR}/nnenv.ep154.pth"
    echo "         Electronic structure inference may not work."
fi

if [ $MISSING_MODELS -eq 1 ]; then
    echo ""
    echo "Please mount the models directory with required model files:"
    echo "  docker-compose.yml:"
    echo "    volumes:"
    echo "      - ./resources/models:/app/resources/models"
    echo ""
    echo "Required files:"
    echo "  - uma-m-1p1.pt (UMA optimizer model)"
    echo "  - nnenv.ep154.pth (Electronic structure inference model)"
    exit 1
fi

echo "OK: Required model files found"

# ============================================
# 检查必需环境变量
# ============================================
echo ""
echo "=== Checking environment variables ==="

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "ERROR: GOOGLE_API_KEY environment variable is not set"
    echo ""
    echo "Please set it in docker-compose.yml or export before running:"
    echo "  export GOOGLE_API_KEY='your-api-key'"
    exit 1
fi
echo "OK: GOOGLE_API_KEY is set"

# ============================================
# 检查可选组件
# ============================================
echo ""
echo "=== Checking optional components ==="

# 检查 Multiwfn（可选，ESP 可视化需要）
if [ -d "/opt/Multiwfn" ] && [ -x "/opt/Multiwfn/Multiwfn" ]; then
    echo "OK: Multiwfn found at /opt/Multiwfn"
    export PATH="/opt/Multiwfn:$PATH"
    
    # 检查并修改 settings.ini 中的 nthreads（如果需要）
    if [ -f "/opt/Multiwfn/settings.ini" ]; then
        echo "    Multiwfn settings.ini found"
    fi
else
    echo "SKIP: Multiwfn not mounted"
    echo "      ESP visualization will be disabled."
    echo "      To enable, set MULTIWFN_PATH and restart:"
    echo "        export MULTIWFN_PATH=/path/to/Multiwfn"
fi

# 检查文献库（可选，RAG 功能需要）
if [ -d "/app/literature" ] && [ "$(ls -A /app/literature 2>/dev/null)" ]; then
    LITERATURE_COUNT=$(find /app/literature -type f \( -name "*.pdf" -o -name "*.txt" -o -name "*.md" \) 2>/dev/null | wc -l)
    echo "OK: Literature directory mounted at /app/literature"
    echo "    Found ${LITERATURE_COUNT} document(s)"
    echo "    NOTE: Update config/settings.yaml to set knowledge.literature_path"
else
    echo "SKIP: Literature directory not mounted or empty"
    echo "      RAG functionality will use default settings."
    echo "      To enable, set LITERATURE_PATH and restart:"
    echo "        export LITERATURE_PATH=/path/to/your/literature"
fi

# ============================================
# 显示配置摘要
# ============================================
echo ""
echo "=== Configuration Summary ==="
echo "EMOLAGENT_ROOT: ${EMOLAGENT_ROOT:-/app}"
echo "CHROMA_DB_PATH: ${EMOL_CHROMA_DB_PATH:-/app/data/chroma_db}"
echo "Multiwfn:       $([ -x /opt/Multiwfn/Multiwfn ] && echo 'Enabled' || echo 'Disabled')"
echo "Literature:     $([ -d /app/literature ] && [ \"$(ls -A /app/literature 2>/dev/null)\" ] && echo 'Mounted' || echo 'Not mounted')"

# ============================================
# 启动应用
# ============================================
echo ""
echo "============================================"
echo "  Starting EMolAgent Application..."
echo "  Access URL: http://localhost:8501"
echo "============================================"
echo ""

# 执行传入的命令（默认是 python run.py）
exec "$@"
