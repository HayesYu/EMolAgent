# EMolAgent

EMolAgent 是一个基于大语言模型的分子模拟智能代理系统，集成了多种分子计算工具和 RAG（检索增强生成）功能。

## 目录

- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
  - [1. 创建 Conda 环境](#1-创建-conda-环境)
  - [2. 安装基础依赖](#2-安装基础依赖)
  - [3. 克隆并安装 EMolAgent](#3-克隆并安装-emolagent)
  - [4. 安装 DeePTB](#4-安装-deeptb)
  - [5. 安装 EMolES](#5-安装-emoles)
  - [6. 安装 learn_qh9](#6-安装-learn_qh9)
  - [7. 安装 dftio](#7-安装-dftio)
  - [8. 克隆 ai4mol](#8-克隆-ai4mol)
  - [9. 安装 Multiwfn](#9-安装-multiwfn)
  - [10. 安装其他依赖](#10-安装其他依赖)
  - [11. 下载模型文件](#11-下载模型文件)
- [配置](#配置)
- [使用方法](#使用方法)
  - [启动应用](#启动应用)
  - [使用 RAG 功能](#使用-rag-功能)

## 环境要求

- Python 3.10
- CUDA 12.8（用于 GPU 加速）
- Conda 包管理器
- Linux 64bit 系统

## 安装步骤

### 1. 创建 Conda 环境

```bash
conda create -n EMolAgent python=3.10
conda activate EMolAgent
```

### 2. 安装基础依赖

```bash
pip install fairchem-core==2.12.0
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv pyg_lib -f https://data.pyg.org/whl/torch-2.8.0+cu128. html
```

### 3. 克隆并安装 EMolAgent

```bash
git clone https://github.com/HayesYu/EMolAgent.git
```

### 4. 安装 DeePTB

```bash
git clone https://github.com/Franklalalala/DeePTB.git
cd DeePTB/
git checkout onehot
```

打开 `pyproject.toml`，取消 `scipy` 和 `lmdb` 的版本限制，然后安装：

```bash
pip install . 
cd ..
```

### 5. 安装 EMolES

```bash
git clone https://github.com/Franklalalala/EMolES.git
cd EMolES/
git checkout dev
pip install -e .
cd ..
```

### 6. 安装 learn_qh9

```bash
git clone https://github.com/Franklalalala/learn_qh9.git
cd learn_qh9/
pip install . 
cd ..
```

### 7. 安装 dftio

```bash
git clone https://github.com/deepmodeling/dftio. git
cd dftio/
```

打开 `pyproject.toml`，取消 `scipy`、`torch`、`lmdb`、`torch-scatter` 的版本限制，然后安装：

```bash
pip install .
cd .. 
```

### 8. 克隆 ai4mol

```bash
git clone https://github.com/Franklalalala/ai4mol.git
```

### 9. 安装 Multiwfn

1. 前往 [Multiwfn 官网](http://sobereva.com/multiwfn/) 下载 Linux 64bit (noGUI version)

2. 解压并配置：

```bash
unzip Multiwfn_2026.1. 12_bin_Linux_noGUI.zip
cd Multiwfn_2026.1.12_bin_Linux_noGUI/
```

3. 配置环境变量，编辑 `~/.bashrc`：

```bash
vim ~/.bashrc
```

添加以下内容：

```bash
ulimit -s unlimited
export Multiwfnpath=/path/to/Multiwfn_2026.1.12_bin_Linux_noGUI
export PATH="$Multiwfnpath:$HOME/bin:$PATH"
```

4. 使配置生效并设置权限：

```bash
source ~/.bashrc
conda activate EMolAgent
chmod +x /path/to/Multiwfn_2026.1.12_bin_Linux_noGUI/Multiwfn
```

5. 修改 `settings.ini` 配置：

```bash
vim settings.ini
```

将 `nthreads` 修改为 `64`（或根据您的 CPU 核心数调整）：

```ini
nthreads=64
```

```bash
cd ..
```

### 10. 安装其他依赖

```bash
# 安装 MOKIT
conda install mokit -c mokit -c conda-forge -y

# 安装系统依赖
sudo apt install -y libcairo2-dev pkg-config python3-dev

# 安装 Python 包
pip install deeprep
pip install streamlit
pip install extra-streamlit-components
pip install pyjwt
pip install langchain==1.2.0 langchain-chroma==1.1.0 langchain-classic==1.0.1 langchain-community==0.4.1 langchain-core==1.2.6 langchain-google-genai==4.1.3 langchain-text-splitters==1.1.0

# 安装 RDKit
conda install -c conda-forge rdkit

# 安装其他依赖
pip install h5py==3.11.0 numpy==2.2.5
pip install openequivariance==0.4.1
pip install -U "huggingface_hub"
```

### 11. 下载模型文件

1. 前往 [Hugging Face UMA](https://huggingface.co/facebook/UMA) 申请获得 `uma-m-1p1. pt` 模型使用权

2. 将 `uma-m-1p1.pt` 放置于 `EMolAgent/resources/models/` 目录下

3. 将其他模型权重文件放置于 `EMolAgent/resources/models/`
   > **注意**：该文件暂时未给出，请等后续更新

## 配置

1. 根据实际需求，修改 `app.py` 和 `knowledge_base.py` 顶部的文件路径配置

2. 前往 [Google AI Studio](https://aistudio.google. com/app/api-keys) 注册您的 Google API Key

## 使用方法

### 启动应用

```bash
cd /path/to/EMolAgent/
export GOOGLE_API_KEY="Your Google API KEY"
streamlit run app.py
```

### 使用 RAG 功能

1. 修改 `knowledge_base.py` 顶部的文献路径配置

2. 在 `app.py` 顶部的 `ADMIN_USERS` 中添加您的用户名

3. 启动应用：

```bash
cd path/to/EMolAgent/
streamlit run app.py
```

4. 在左侧栏中点击 **"重建索引"** 即可使用 RAG 功能

## 许可证

本项目采用 [MIT 许可证](LICENSE) 进行许可。

## 致谢

感谢以下开源项目的支持：
- [DeePTB](https://github.com/Franklalalala/DeePTB)
- [EMolES](https://github.com/Franklalalala/EMolES)
- [learn_qh9](https://github.com/Franklalalala/learn_qh9)
- [dftio](https://github.com/deepmodeling/dftio)
- [ai4mol](https://github.com/Franklalalala/ai4mol)
- [Multiwfn](http://sobereva.com/multiwfn/)
- [MOKIT](https://github.com/1234zou/MOKIT)
- [UMA](https://huggingface.co/facebook/UMA)
