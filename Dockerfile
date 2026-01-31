# EMolAgent Dockerfile
# 基于 NVIDIA CUDA 12.8 镜像，包含完整的计算化学 AI 环境

# ============================================
# Stage 1: 基础镜像
# ============================================
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS base

# 避免交互式安装提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    vim \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Stage 2: 安装 Miniconda
# ============================================
ENV CONDA_DIR=/opt/conda
ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    conda init bash && \
    conda update -n base -c defaults conda -y

# ============================================
# Stage 3: 创建 Conda 环境并安装基础依赖
# ============================================
RUN conda create -n EMolAgent python=3.10 -y

# 激活环境的 shell 设置
SHELL ["conda", "run", "-n", "EMolAgent", "/bin/bash", "-c"]

# 安装 conda 依赖（MOKIT、RDKit）
RUN conda install -n EMolAgent mokit -c mokit -c conda-forge -y && \
    conda install -n EMolAgent -c conda-forge rdkit -y

# ============================================
# Stage 4: 安装 PyTorch 和 CUDA 相关依赖
# ============================================
RUN pip install fairchem-core==2.12.0 && \
    pip install torch_geometric && \
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv pyg_lib \
        -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# ============================================
# Stage 5: 安装 GitHub 仓库依赖
# ============================================
WORKDIR /tmp/deps

# DeePTB - 切换 onehot 分支 + 取消 scipy/lmdb 版本限制
RUN git clone https://github.com/Franklalalala/DeePTB.git && \
    cd DeePTB && \
    git checkout onehot && \
    sed -i 's/"scipy[<>=!][^"]*"/"scipy"/g' pyproject.toml && \
    sed -i 's/"lmdb[<>=!][^"]*"/"lmdb"/g' pyproject.toml && \
    pip install . && \
    cd .. && rm -rf DeePTB

# EMolES - 切换 dev 分支（使用可编辑安装，保留源码）
RUN git clone https://github.com/Franklalalala/EMolES.git /opt/EMolES && \
    cd /opt/EMolES && \
    git checkout dev && \
    pip install -e .

# learn_qh9
RUN git clone https://github.com/Franklalalala/learn_qh9.git && \
    cd learn_qh9 && \
    pip install . && \
    cd .. && rm -rf learn_qh9

# dftio - 取消多个依赖版本限制
RUN git clone https://github.com/deepmodeling/dftio.git && \
    cd dftio && \
    sed -i 's/"scipy[<>=!][^"]*"/"scipy"/g' pyproject.toml && \
    sed -i 's/"torch[<>=!][^"]*"/"torch"/g' pyproject.toml && \
    sed -i 's/"lmdb[<>=!][^"]*"/"lmdb"/g' pyproject.toml && \
    sed -i 's/"torch-scatter[<>=!][^"]*"/"torch-scatter"/g' pyproject.toml && \
    pip install . && \
    cd .. && rm -rf dftio

# ai4mol - 仅克隆，不安装（作为资源使用）
RUN git clone https://github.com/Franklalalala/ai4mol.git /opt/ai4mol

# 清理临时目录
WORKDIR /
RUN rm -rf /tmp/deps

# ============================================
# Stage 6: 安装 EMolAgent
# ============================================
WORKDIR /app

# 复制项目文件
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY config/ ./config/
COPY resources/db/ ./resources/db/
COPY run.py ./

# 安装 EMolAgent
RUN pip install -e .

# ============================================
# Stage 7: 配置运行环境
# ============================================

# 设置环境变量
ENV EMOLAGENT_ROOT=/app
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Multiwfn 环境变量（运行时挂载）
ENV Multiwfnpath=/opt/Multiwfn
ENV PATH="${Multiwfnpath}:${PATH}"

# 创建运行时目录
RUN mkdir -p /app/data /app/users /app/logs /app/.task_slots /app/resources/models

# 复制 entrypoint 脚本
COPY scripts/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# 暴露端口
EXPOSE 8501

# 设置入口点和默认命令
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "EMolAgent", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "run.py"]
