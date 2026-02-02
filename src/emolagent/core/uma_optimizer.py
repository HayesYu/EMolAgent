"""
UMA (Universal Machine Learning Atomistic) 优化器模块

使用 FAIRChem 的 UMA 模型进行分子结构优化。
"""

import os
import re
import shutil
import argparse
import threading
import numpy as np
from ase import Atoms
from ase.io import write
from ase.db import connect
from ase.optimize import LBFGS
from tqdm import tqdm

from emolagent.utils.logger import logger
from emolagent.utils.paths import get_resource_path
from emolagent.utils.config import ModelConfig, OutputConfig

# 全局锁，用于保护 FAIRChem 模型加载（防止并发懒加载冲突）
_MODEL_LOAD_LOCK = threading.Lock()

# Optional import for SMILES processing
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.calculate.pretrained_mlip import get_isolated_atomic_energies

# ==========================================
# Default Configuration（从配置文件加载）
# ==========================================
DEFAULT_CHECKPOINT = ModelConfig.get_uma_checkpoint_path()
DEFAULT_WORKSPACE = OutputConfig.get_uma_workspace()
DEFAULT_MODEL_NAME = ModelConfig.get_uma_model_name()


def sanitize_name(s: str) -> str:
    """清理字符串，使其适合作为文件名。"""
    s = str(s) if s is not None else ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "unnamed"


def _coerce_int(x, default=None):
    """
    安全的整数转换，处理 int/float/np scalars/strings 等各种类型。
    
    Args:
        x: 要转换的值
        default: 转换失败时返回的默认值
        
    Returns:
        转换后的整数，或默认值
    """
    if x is None:
        return default
    try:
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return int(float(x))
        return int(float(x))
    except Exception:
        return default


def smiles_to_atoms(smiles: str) -> Atoms:
    """将 SMILES 字符串转换为 ASE Atoms 对象。"""
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required to process SMILES strings.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, randomSeed=42)
    if res == -1:
        # Try random coordinates if embedding fails
        AllChem.EmbedMolecule(mol, useRandomCoords=True)

    # Convert RDKit to ASE
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    return Atoms(symbols=symbols, positions=positions)


def prepare_input_source(workspace: str, input_db: str = None, smiles_list: list = None) -> str:
    """
    确定输入数据库。
    如果提供了 SMILES，则创建临时数据库并返回其路径。
    否则返回现有的 input_db 路径。
    """
    # Case 1: SMILES provided -> Generate temp DB
    if smiles_list:
        temp_db_path = os.path.join(workspace, 'temp_smiles_input.db')
        logger.info(f"Generating 3D structures from {len(smiles_list)} SMILES -> {temp_db_path} ...")

        # Clean previous temp file if exists
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

        with connect(temp_db_path) as db:
            for i, smi in enumerate(smiles_list):
                try:
                    atoms = smiles_to_atoms(smi)
                    name = f"smiles_{i}_{sanitize_name(smi[:10])}"
                    db.write(atoms, name=name, smiles=smi)
                except Exception as e:
                    logger.warning(f"Failed to convert SMILES '{smi}': {e}")

        return temp_db_path

    # Case 2: Use existing DB
    if not input_db:
        return os.path.join(workspace, 'all.db')

    return input_db


def _merge_row_metadata(row, atoms: Atoms) -> dict:
    """
    合并 ASE DB row 的元数据。
    
    ASE DB rows 可以在多个位置存储元数据：
      - row.key_value_pairs
      - row.data (dict)
      - atoms.info
    
    重要：structures.db 中 data['charge'] 是正确的，但 key_value_pairs['charge'] 
    可能是错误的（常为1）。因此合并时必须让 row.data 覆盖 row.key_value_pairs。
    
    合并顺序：
      1. key_value_pairs first
      2. row.data second (override kvp; especially for 'charge')
      3. atoms.info last fallback (only fill missing keys)
    """
    meta = {}

    # 1. key_value_pairs first
    try:
        if getattr(row, "key_value_pairs", None):
            meta.update(row.key_value_pairs)
    except Exception:
        pass

    # 2. row.data second (override kvp; especially for 'charge')
    try:
        if getattr(row, "data", None):
            meta.update(row.data)
    except Exception:
        pass

    # 3. atoms.info last fallback (only fill missing keys)
    try:
        if isinstance(getattr(atoms, "info", None), dict):
            for k, v in atoms.info.items():
                if k not in meta:
                    meta[k] = v
    except Exception:
        pass

    return meta


def _infer_charge(atoms: Atoms, meta: dict) -> int:
    """
    推断分子/团簇的电荷。
    
    电荷优先级：
      1) meta['charge'] if present (现在正确优先使用 row.data['charge'])
      2) 从阴离子数量键推断: n_anion, n_anion_total, n_anions, n_anion_tot
    
    推断规则（单体 vs 团簇）：
      - 如果结构包含 Li: charge = 1 - n_anion (Li+ 团簇惯例)
      - 否则: charge = 0 - n_anion (单体惯例: 溶剂 0, 阴离子 -1)
    
    回退（如果没有 n_anion 信息）：
      - 包含 Li -> +1
      - 否则 -> 0
    """
    # 1) 如果 meta 中有 charge，直接使用
    if "charge" in meta and meta["charge"] is not None:
        ch = _coerce_int(meta["charge"], default=None)
        if ch is not None:
            return int(ch)

    # 2) 尝试从阴离子数量推断
    n_anion = None
    for key in ("n_anion", "n_anion_total", "n_anions", "n_anion_tot"):
        if key in meta and meta[key] is not None:
            n_anion = _coerce_int(meta[key], default=None)
            if n_anion is not None:
                break

    has_li = ("Li" in atoms.get_chemical_symbols())

    if n_anion is not None:
        # Li+ 团簇: charge = 1 - n_anion
        # 单体: charge = 0 - n_anion
        return int((1 - n_anion) if has_li else (0 - n_anion))

    # 3) 回退: Li -> +1, 否则 -> 0
    return int(1 if has_li else 0)


def entry(
        input_db: str = None,
        smiles: list = None,
        workspace: str = DEFAULT_WORKSPACE,
        checkpoint_path: str = None,
        device: str = "cuda",
        fmax: float = 0.05,
        max_steps: int = 200,
        verbose: bool = False,
        show_progress: bool = True
) -> str:
    """
    主优化入口函数。
    
    Args:
        input_db: 输入的 ASE 数据库路径
        smiles: SMILES 字符串列表
        workspace: 工作目录
        checkpoint_path: 模型检查点路径
        device: 计算设备 (cuda/cpu)
        fmax: 力收敛标准
        max_steps: 最大优化步数
        verbose: 是否输出详细日志
        show_progress: 是否显示进度条
        
    Returns:
        输出数据库的路径
    """
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT
        
    # 1. Path Setup & Cleanup
    traj_dir = os.path.join(workspace, 'traj')
    out_xyz_dir = os.path.join(workspace, 'optimized_xyz_all')
    out_db_path = os.path.join(workspace, 'optimized_all.db')

    if os.path.exists(out_db_path):
        if verbose:
            logger.debug(f"Removing existing output DB: {out_db_path}")
        os.remove(out_db_path)

    for d in [traj_dir, out_xyz_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 2. Prepare Input Source (DB or SMILES->DB)
    active_input_db = prepare_input_source(workspace, input_db, smiles)

    if verbose:
        logger.info(f"Workdir: {workspace}\nInput: {active_input_db}\nOutput: {out_db_path}")

    # 3. Model Loading (使用锁防止并发懒加载冲突)
    if verbose:
        logger.info("Loading FAIRChem model...")
    
    with _MODEL_LOAD_LOCK:
        logger.debug(f"[UMA] Acquired model load lock for workspace: {workspace}")
        atom_refs = get_isolated_atomic_energies(DEFAULT_MODEL_NAME, workspace)
        predictor = load_predict_unit(checkpoint_path, "default", None, device, atom_refs)
        calc = FAIRChemCalculator(predictor, task_name="omol")
        logger.debug(f"[UMA] Released model load lock, model ready")

    # 4. Optimization Loop
    if not os.path.exists(active_input_db):
        raise FileNotFoundError(f"DB not found: {active_input_db}")

    with connect(active_input_db) as src_db, connect(out_db_path) as tgt_db:
        total = src_db.count()
        
        rows = src_db.select()
        if show_progress:
            rows = tqdm(rows, total=total, desc="Optimizing", unit="mol")

        def _log(msg: str):
            """日志输出，兼容进度条"""
            if show_progress:
                tqdm.write(msg)
            else:
                logger.info(msg)

        for row in rows:
            atoms = row.toatoms()
            
            # --- Core Logic: Charge Calculation (优先使用 row.data['charge']) ---
            # 关键修复：row.data['charge'] 是正确的，row.key_value_pairs['charge'] 可能是错误的
            charge = None
            try:
                # 尝试从 row.data 获取 charge（优先级最高）
                if hasattr(row, 'data') and row.data:
                    charge = row.data.get('charge', None)
            except Exception:
                pass
            
            if charge is None:
                # row.data 中没有 charge，使用完整的推断逻辑
                logger.debug(f"Row {row.id}: No charge in row.data, inferring...")
                meta = _merge_row_metadata(row, atoms)
                charge = _infer_charge(atoms, meta)
            else:
                # row.data 中有 charge，安全转换为 int
                charge = _coerce_int(charge, default=None)
                if charge is None:
                    meta = _merge_row_metadata(row, atoms)
                    charge = _infer_charge(atoms, meta)
                else:
                    meta = _merge_row_metadata(row, atoms)
            
            # spin 永远是 1，不从任何来源读取/改变
            spin = 1
            
            # 确保 charge 和 spin 是整数类型
            charge = int(charge)
            spin = int(spin)
            
            # 设置属性到 atoms.info 和 meta
            atoms.info['charge'] = charge
            atoms.info['spin'] = spin
            meta['charge'] = charge
            meta['spin'] = spin

            # --- Determine name early ---
            raw_name = meta.get('xyz_file', None) or meta.get('name', None) or f"id_{row.id}"
            raw_name = os.path.splitext(str(raw_name))[0]
            base_name = sanitize_name(raw_name)

            # 每个结构优化前打印 charge/spin
            _log(f"[UMA] Optimizing: {base_name} | charge={charge} spin={spin}")

            # --- Optimization ---
            atoms.calc = calc
            traj_path = os.path.join(traj_dir, f"{base_name}.traj")

            try:
                logfile = '-' if (verbose and not show_progress) else None
                opt = LBFGS(atoms, trajectory=traj_path, logfile=logfile)
                opt.run(fmax=fmax, steps=max_steps)

                # Save Results
                out_xyz = os.path.join(out_xyz_dir, f"{base_name}.xyz")
                write(out_xyz, atoms)

                atoms.calc = None  # detach calculator before storing
                tgt_db.write(atoms, data=meta, **meta)

            except Exception as e:
                _log(f"[Error] {base_name}: {e}")

    return out_db_path


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="FAIRChem Optimization Script")
    parser.add_argument('--workspace', default=DEFAULT_WORKSPACE, help='Root output directory')
    parser.add_argument('--input-db', default=None, help='Input ASE database')
    parser.add_argument('--smiles', nargs='*', default=None, help='Input SMILES string(s) to optimize')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='Model checkpoint path')
    parser.add_argument('--device', default='cuda', help='Compute device')
    parser.add_argument('--fmax', type=float, default=0.05, help='Force convergence criteria')
    parser.add_argument('--steps', type=int, default=200, dest='max_steps', help='Max optimization steps')
    parser.add_argument('--verbose', action='store_true', help='Show optimization logs')
    parser.add_argument('--no-progress', action='store_false', dest='progress', help='Disable progress bar')
    parser.set_defaults(progress=True)

    args = parser.parse_args()

    out_path = entry(
        input_db=args.input_db,
        smiles=args.smiles,
        workspace=args.workspace,
        checkpoint_path=args.checkpoint,
        device=args.device,
        fmax=args.fmax,
        max_steps=args.max_steps,
        verbose=args.verbose,
        show_progress=args.progress
    )

    logger.info(f"\nOptimization finished. Output DB: {out_path}")


if __name__ == "__main__":
    main()
