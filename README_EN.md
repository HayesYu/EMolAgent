# EMolAgent

> [中文](README.md) | English

EMolAgent is an LLM-based computational chemistry AI assistant that integrates molecular cluster computation (structure construction & optimization + electronic property prediction) and RAG literature Q&A.

## Table of Contents

- [EMolAgent](#emolagent)
  - [Table of Contents](#table-of-contents)
  - [Chinese/English Bilingual Support](#chineseenglish-bilingual-support)
  - [Project Structure](#project-structure)
  - [Requirements](#requirements)
  - [Docker Quick Deploy](#docker-quick-deploy)
  - [Manual Installation (Recommended)](#manual-installation-recommended)
    - [1. Create Conda Environment](#1-create-conda-environment)
    - [2. Install Base Dependencies](#2-install-base-dependencies)
    - [3. Clone EMolAgent](#3-clone-emolagent)
    - [4. Install DeePTB](#4-install-deeptb)
    - [5. Install EMolES](#5-install-emoles)
    - [6. Install learn_qh9](#6-install-learn_qh9)
    - [7. Install dftio](#7-install-dftio)
    - [8. Clone ai4mol](#8-clone-ai4mol)
    - [9. Install Multiwfn](#9-install-multiwfn)
    - [10. Install Other Dependencies](#10-install-other-dependencies)
    - [11. Download Model Files](#11-download-model-files)
  - [Configuration](#configuration)
    - [Config File](#config-file)
    - [API Key Configuration](#api-key-configuration)
  - [Usage](#usage)
    - [Launch Application](#launch-application)
    - [Using RAG](#using-rag)
    - [Using ESP Visualization](#using-esp-visualization)
    - [Custom Configuration](#custom-configuration)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Chinese/English Bilingual Support

EMolAgent supports **Chinese** and **English** interfaces. You can switch languages anytime within the app:

- **How to switch**: Use the language selector (🌐) in the login page or the top-right corner of the sidebar
- **Preference persistence**: Your language choice is saved in cookies and restored on your next visit
- **Coverage**: UI text, model system prompts, and visualization components all support both languages

## Project Structure

```
EMolAgent/
├── src/emolagent/           # Main package
│   ├── app.py               # Streamlit main application
│   ├── core/                # Core modules
│   │   ├── cluster_factory.py   # Molecular cluster construction
│   │   ├── uma_optimizer.py     # UMA structure optimization
│   │   └── tools.py             # LangChain tools
│   ├── database/            # Database module
│   │   └── db.py                # User and session management
│   ├── knowledge/           # RAG knowledge base module
│   │   └── knowledge_base.py    # Literature Q&A system
│   ├── visualization/       # Visualization module
│   │   └── mol_viewer.py        # 3D molecular visualization
│   └── utils/               # Utility modules
│       ├── config.py            # Configuration management
│       ├── i18n.py              # Internationalization (Chinese/English)
│       ├── logger.py            # Logging configuration
│       └── paths.py             # Path management
├── config/                  # Configuration directory
│   └── settings.yaml            # Main config file
├── resources/               # Resource files
│   ├── models/              # Model weights
│   └── db/                  # Database files
├── data/                    # ChromaDB vector database
├── users/                   # User data directory
├── run.py                   # Launch script
└── pyproject.toml           # Project configuration
```

## Requirements

- Python 3.10
- CUDA 12.8 (for GPU acceleration)
- Conda package manager
- Linux 64bit

## Docker Quick Deploy

Docker allows quick deployment of EMolAgent without manually installing complex dependencies.

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU driver
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# Install NVIDIA Container Toolkit (Ubuntu)
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Quick Start

#### 1. Clone the Project

```bash
git clone https://github.com/HayesYu/EMolAgent.git
cd EMolAgent
```

#### 2. Prepare Model Files

```bash
# Create models directory
mkdir -p resources/models

# Place model files in resources/models/:
# - uma-m-1p1.pt (apply at https://huggingface.co/facebook/UMA)
# - Electronic structure inference weight file (You can obtain it from https://drive.google.com/drive/folders/17u1Ex9FNi0lza2Kc0vjP4fU-NssIlO-2?usp=drive_link)
# This version is for testing purposes only; the latest version has not yet been released.
```

#### 3. Configure Environment Variables

```bash
# Required: Google API Key
export GOOGLE_API_KEY="your-google-api-key"

# Required: Multiwfn
# Download Linux 64bit noGUI version from http://sobereva.com/multiwfn/ (See the manual installation section for details.)
export MULTIWFN_PATH="/path/to/Multiwfn_2026.1.12_bin_Linux_noGUI"

# Optional: RAG literature directory
export LITERATURE_PATH="/path/to/your/literature"
```

#### 4. Build and Start

```bash
# Build image (takes time on first run)
docker-compose build

# Start service
docker-compose up -d

# View logs
docker-compose logs -f
```

#### 5. Access Application

Open http://localhost:8501 in your browser

### Docker Volume Mounts

| Mount Path | Purpose | Required |
|------------|---------|----------|
| `./data` | ChromaDB + SQLite data | Yes (auto-created) |
| `./users` | User task output | Yes (auto-created) |
| `./logs` | Log files | Yes (auto-created) |
| `./resources/models` | Model files | Yes (must be pre-placed) |
| `./config/settings.yaml` | Custom config | No |
| `$MULTIWFN_PATH` | Multiwfn binary | Yes (must be pre-placed) |
| `$LITERATURE_PATH` | RAG literature | No |

### Common Docker Commands

```bash
# Stop service
docker-compose down

# Restart service
docker-compose restart

# Check container status
docker-compose ps

# Enter container for debugging
docker-compose exec emolagent bash
```

---

## Manual Installation (Recommended)

### 1. Create Conda Environment

```bash
conda create -n EMolAgent python=3.10
conda activate EMolAgent
```

### 2. Install Base Dependencies

```bash
pip install fairchem-core==2.12.0
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv pyg_lib -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

### 3. Clone EMolAgent

```bash
git clone https://github.com/HayesYu/EMolAgent.git
```

### 4. Install DeePTB

```bash
git clone https://github.com/Franklalalala/DeePTB.git
cd DeePTB/
git checkout onehot
```

Remove version constraints for `scipy` and `lmdb` in `pyproject.toml`, then install:

```bash
pip install .
cd ..
```

### 5. Install EMolES

```bash
git clone https://github.com/Franklalalala/EMolES.git
cd EMolES/
git checkout dev
pip install -e .
cd ..
```

### 6. Install learn_qh9

```bash
git clone https://github.com/Franklalalala/learn_qh9.git
cd learn_qh9/
pip install .
cd ..
```

### 7. Install dftio

```bash
git clone https://github.com/deepmodeling/dftio.git
cd dftio/
```

Remove version constraints for `scipy`, `torch`, `lmdb`, `torch-scatter` in `pyproject.toml`, then install:

```bash
pip install .
cd ..
```

### 8. Clone ai4mol

```bash
git clone https://github.com/Franklalalala/ai4mol.git
```

### 9. Install Multiwfn

1. Download Linux 64bit (noGUI version) from [Multiwfn official site](http://sobereva.com/multiwfn/)

2. Extract and configure:

```bash
unzip Multiwfn_2026.1.12_bin_Linux_noGUI.zip
cd Multiwfn_2026.1.12_bin_Linux_noGUI/
```

3. Configure environment variables. Edit `~/.bashrc`:

```bash
vim ~/.bashrc
```

Add the following:

```bash
ulimit -s unlimited
export Multiwfnpath=/path/to/Multiwfn_2026.1.12_bin_Linux_noGUI
export PATH="$Multiwfnpath:$HOME/bin:$PATH"
```

4. Apply configuration and set permissions:

```bash
source ~/.bashrc
conda activate EMolAgent
chmod +x /path/to/Multiwfn_2026.1.12_bin_Linux_noGUI/Multiwfn
```

5. Edit `settings.ini`:

```bash
vim settings.ini
```

Set `nthreads` to `64` (or adjust based on your CPU cores):

```ini
nthreads=64
```

```bash
cd ..
```

### 10. Install Other Dependencies

```bash
# Install MOKIT
conda install mokit -c mokit -c conda-forge -y

# Install system dependencies
sudo apt install -y libcairo2-dev pkg-config python3-dev

# Install RDKit
conda install -c conda-forge rdkit

# Install EMolAgent and its Python dependencies
cd EMolAgent/
pip install -e .
```

### 11. Download Model Files

1. Apply for `uma-m-1p1.pt` model access at [Hugging Face UMA](https://huggingface.co/facebook/UMA)

2. Place `uma-m-1p1.pt` in `EMolAgent/resources/models/`

3. Apply for electronic structure inference weight file access at [Google Drive nnenv](https://drive.google.com/drive/folders/17u1Ex9FNi0lza2Kc0vjP4fU-NssIlO-2?usp=drive_link)
   > **Note**: This file is for testing purposes only; the latest version has not yet been released.

4. Place electronic structure inference weight file in `EMolAgent/resources/models/`

## Configuration

### Config File

The project uses a YAML config file for centralized parameter management at `config/settings.yaml`.

Main configuration items:

| Category | Key | Description |
|----------|-----|-------------|
| `database` | `solvent_db`, `anion_db` | Molecular database file paths |
| `models` | `inference_model`, `uma_checkpoint`, `uma_model_name` | Model paths and names |
| `gpu` | `available_gpus`, `max_tasks_per_gpu` | GPU device list and concurrent task limits |
| `logging` | `max_log_size`, `backup_count` | Log file size and backup count |
| `visualization` | `max_preview_structures` | Max number of structures in preview |
| `auth` | `admin_users` | Admin usernames |
| `knowledge` | `literature_path`, `collection_name` | Literature path and collection name |
| `molecules` | `default_dme_smiles`, `default_fsi_smiles` | Default molecule SMILES |
| `output` | `uma_workspace` | Optimizer output directory |

Restart the application after config changes. You can also set `EMOL_CONFIG_PATH` to use a custom config path.

### API Key Configuration

Register your Google API Key at [Google AI Studio](https://aistudio.google.com/app/api-keys)

## Usage

### Launch Application

```bash
cd /path/to/EMolAgent/
export GOOGLE_API_KEY="Your Google API KEY"

# Option 1: Use launch script (recommended)
python run.py

# Option 2: Run Streamlit directly
streamlit run src/emolagent/app.py
```

### Using RAG

1. Place PDF files in the configured literature directory (`knowledge.literature_path` in `config/settings.yaml`)

2. Add your username to `auth.admin_users` in `config/settings.yaml`

3. After launching, click **"Rebuild Index"** in the sidebar to use RAG

### Using ESP Visualization

ESP visualization requires cube files generated by Multiwfn. To enable:

1. Change `gen_esp_cube_flag: bool = False` to `True` in `EMolES/src/emoles/inference/infer_entry.py`

2. Ensure Multiwfn is installed and environment variables are configured

### Custom Configuration

To modify default parameters (e.g., GPU concurrency, admin list, model paths), edit `config/settings.yaml`.

Example:

```yaml
# Modify GPU config
gpu:
  available_gpus: [0, 1, 2, 3]  # Use 4 GPUs
  max_tasks_per_gpu: 3          # Max 3 concurrent tasks per GPU

# Add admins
auth:
  admin_users:
    - "hayes"
    - "your_username"

# Modify literature path
knowledge:
  literature_path: "/your/custom/path/to/literature"
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Thanks to the following open-source projects:
- [DeePTB](https://github.com/Franklalalala/DeePTB)
- [EMolES](https://github.com/Franklalalala/EMolES)
- [learn_qh9](https://github.com/Franklalalala/learn_qh9)
- [dftio](https://github.com/deepmodeling/dftio)
- [ai4mol](https://github.com/Franklalalala/ai4mol)
- [Multiwfn](http://sobereva.com/multiwfn/)
- [MOKIT](https://github.com/1234zou/MOKIT)
- [UMA](https://huggingface.co/facebook/UMA)
