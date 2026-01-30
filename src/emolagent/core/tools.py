"""
核心工具函数模块

提供数据库搜索、团簇构建、电子结构推断等工具函数。
"""

import os
import shutil
import json
import time
import re
import multiprocessing
import pandas as pd
import torch
import numpy as np
from ase.db import connect

from emoles.py3Dmol import cube_2_html, cubes_2_htmls
from emoles.inference import infer_entry

from emolagent.utils.logger import logger
from emolagent.utils.paths import get_resource_path
from emolagent.utils.config import DatabaseConfig
from emolagent.core import cluster_factory

# ==========================================
# 常量定义（从配置文件加载）
# ==========================================
SOLVENT_DB_PATH = DatabaseConfig.get_solvent_db_path()
ANION_DB_PATH = DatabaseConfig.get_anion_db_path()


class NumpyEncoder(json.JSONEncoder):
    """JSON 编码器，支持 NumPy 类型。"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# ==========================================
# 数据库搜索工具 (Search)
# ==========================================

def search_molecule_in_db(query_name: str, mol_type: str, output_dir: str) -> str:
    """
    在指定数据库中搜索分子。
    
    Args:
        query_name: 分子名称 (如 "DME", "FSI")
        mol_type: "solvent" 或 "anion"
        output_dir: 结果输出目录
        
    Returns:
        结果信息的 JSON 字符串 (包含是否找到，以及生成的 temp_db 路径)
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    target_db = SOLVENT_DB_PATH if mol_type == "solvent" else ANION_DB_PATH
    
    if not os.path.exists(target_db):
        return json.dumps({"found": False, "msg": f"Database file not found: {target_db}"})

    query_norm = query_name.strip()
    found_row = None
    exact_match = False
    
    with connect(target_db) as src_db:
        for row in src_db.select():
            a_name = row.get('name', '')
            if not a_name and row.key_value_pairs:
                a_name = row.key_value_pairs.get('name', '')
            
            if not a_name:
                continue

            if a_name == query_norm:
                found_row = row
                exact_match = True
                break
            
            short_name = a_name.split('(')[0].strip()
            if short_name == query_norm:
                found_row = row
                if not exact_match: 
                    exact_match = False

        if found_row:
            safe_name = re.sub(r'[^A-Za-z0-9]', '_', query_name)
            unique = f"{time.time_ns()}_{os.getpid()}"
            temp_db_name = f"found_{mol_type}_{safe_name}_{unique}.db"
            temp_db_path = os.path.abspath(os.path.join(output_dir, temp_db_name))

            atoms = found_row.toatoms()
            kvp = (found_row.key_value_pairs or {}).copy()
            data = (found_row.data or {}).copy()
            kvp.pop("name", None)
                
            with connect(temp_db_path) as tmp_db:
                tmp_db.write(atoms, name=query_name, data=data, **kvp)
            
            return json.dumps({
                "found": True,
                "name": found_row.get('name'),
                "db_path": temp_db_path,
                "msg": f"Success: Found '{query_name}' in {mol_type} DB."
            })
        else:
            return json.dumps({
                "found": False,
                "msg": f"'{query_name}' not found in {mol_type} DB."
            })


# ==========================================
# 建模与优化工具 (Cluster Factory + UMA)
# ==========================================

def build_and_optimize_cluster(
    ion_name: str,
    solvents_config: list,
    anions_config: list,
    output_dir: str
) -> str:
    """
    调用 Cluster Factory 进行建模，并自动进行 UMA 优化。
    
    Args:
        ion_name: 离子名称 (如 "Li")
        solvents_config: 溶剂配置列表 [{"name": "DME", "path": "...", "count": 3}]
        anions_config: 阴离子配置列表 [{"name": "FSI", "path": "...", "count": 1}]
        output_dir: 输出目录
        
    Returns:
        JSON 格式的结果字符串
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 1. 准备 Solvents 参数
        solvent_args = []
        s_counts = set()
        
        if solvents_config:
            for s_item in solvents_config:
                if s_item.get('path') and os.path.exists(s_item.get('path')):
                    solvent_args.append(s_item['path'])
                elif s_item.get('smiles'):
                    solvent_args.append(s_item['smiles'])
                
                if s_item.get('count'):
                    try:
                        s_counts.add(int(s_item['count']))
                    except ValueError:
                        pass

        if not solvent_args:
            return "Error: No solvent information provided."

        # 2. 准备 Anions 参数
        anion_args = []
        a_counts_agg = set()
        has_cip_request = False
        
        if anions_config:
            for a_item in anions_config:
                if a_item.get("path"):
                    p = os.path.abspath(a_item["path"])
                    if os.path.exists(p):
                        anion_args.append(p)
                elif a_item.get('smiles'):
                    anion_args.append(a_item['smiles'])
                
                if a_item.get('count'):
                    try:
                        c = int(a_item['count'])
                        if c == 1:
                            has_cip_request = True
                        elif c > 1:
                            a_counts_agg.add(c)
                    except ValueError:
                        pass

        # 3. 确定 Categories 和 Counts
        cats = []
        if not anion_args:
            cats.append("SSIP")
        else:
            if has_cip_request:
                cats.append("CIP")
            if a_counts_agg:
                cats.append("AGG")
            if not cats: 
                cats.append("CIP")

        final_s_counts = tuple(sorted(list(s_counts))) if s_counts else (4,)
        final_a_counts = tuple(sorted(list(a_counts_agg))) if a_counts_agg else (2,)

        max_s = max(final_s_counts) if final_s_counts else 0
        max_a = max(final_a_counts) if "AGG" in cats else (1 if "CIP" in cats else 0)
        max_total = max_s + max_a

        logger.info(f"Calling Cluster Factory: Solv={solvent_args}, Anion={anion_args}")
        logger.info(f"Plan: Cats={cats}, S_Counts={final_s_counts}, A_Counts(AGG)={final_a_counts}")
        
        # 4. 调用 cluster_factory.entry
        stats = cluster_factory.entry(
            solvents=solvent_args,
            anions=anion_args,
            out_dir=output_dir,
            ion=ion_name,
            solv_counts=final_s_counts,
            agg_anion_counts=final_a_counts,
            max_total_ligands=max_total,
            categories=tuple(cats),
            optimize_result=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False,
            show_progress=False
        )
        
        # 5. 寻找结果文件
        final_opt_dir = os.path.join(output_dir, "final_optimized")
        result_db = None
        
        if os.path.exists(final_opt_dir):
            preferred = os.path.join(final_opt_dir, "optimized_all.db")
            if os.path.exists(preferred):
                result_db = preferred
            else:
                dbs = [os.path.join(final_opt_dir, f) for f in os.listdir(final_opt_dir) if f.endswith(".db")]
                if dbs:
                    result_db = max(dbs, key=os.path.getmtime)
        
        if not result_db:
            raw_db = os.path.join(output_dir, "all.db")
            if os.path.exists(raw_db):
                result_db = raw_db
        
        if result_db:
            return json.dumps({
                "success": True, 
                "optimized_db": result_db,
                "stats": stats,
                "msg": f"Cluster built successfully. Stats: {stats}"
            })
        else:
            return json.dumps({"success": False, "msg": "Cluster building failed, no output DB found."})

    except Exception as e:
        logger.exception("Error occurred")
        return f"Error in build_and_optimize_cluster: {str(e)}"


def build_multiple_clusters(
    ion_name: str,
    recipes: list,
    output_dir: str
) -> str:
    """
    批量构建多个不同配方的团簇，逐个配方精确构建，然后统一进行 UMA 优化。
    
    与 build_and_optimize_cluster 的区别：这个函数会精确构建用户指定的每个配方，
    而不是做笛卡尔积组合。
    
    Args:
        ion_name: 离子名称 (如 "Li")
        recipes: 配方列表，每个配方包含:
            [
                {
                    "solvents": [{"name": "DME", "path": "...", "count": 3}],
                    "anions": [{"name": "FSI", "path": "...", "count": 1}]
                },
                {
                    "solvents": [{"name": "DME", "path": "...", "count": 2}],
                    "anions": [{"name": "FSI", "path": "...", "count": 2}]
                }
            ]
        output_dir: 输出目录
        
    Returns:
        JSON 格式的结果字符串
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if not recipes or len(recipes) == 0:
            return json.dumps({"success": False, "msg": "No recipes provided."})

        logger.info(f"[Batch Build] Processing {len(recipes)} recipes separately...")
        
        # 为每个配方单独构建，不做优化，最后统一优化
        all_clusters_db_path = os.path.join(output_dir, "all_clusters_raw.db")
        if os.path.exists(all_clusters_db_path):
            os.remove(all_clusters_db_path)
        
        total_built = 0
        total_failed = 0
        
        for idx, recipe in enumerate(recipes):
            solvents_config = recipe.get("solvents", [])
            anions_config = recipe.get("anions", [])
            
            # 解析溶剂参数
            solvent_args = []
            s_count = 1
            for s_item in solvents_config:
                if s_item.get('path') and os.path.exists(s_item.get('path')):
                    solvent_args.append(s_item['path'])
                elif s_item.get('smiles'):
                    solvent_args.append(s_item['smiles'])
                if s_item.get('count'):
                    try:
                        s_count = int(s_item['count'])
                    except ValueError:
                        pass
            
            # 解析阴离子参数
            anion_args = []
            a_count = 0
            for a_item in anions_config:
                if a_item.get("path"):
                    p = os.path.abspath(a_item["path"])
                    if os.path.exists(p):
                        anion_args.append(p)
                elif a_item.get('smiles'):
                    anion_args.append(a_item['smiles'])
                if a_item.get('count'):
                    try:
                        a_count = int(a_item['count'])
                    except ValueError:
                        pass
            
            if not solvent_args:
                logger.warning(f"[Batch Build] Recipe {idx+1}: No solvent, skipping")
                total_failed += 1
                continue
            
            # 确定 category
            if a_count == 0:
                cat = "SSIP"
            elif a_count == 1:
                cat = "CIP"
            else:
                cat = "AGG"
            
            logger.info(f"[Batch Build] Recipe {idx+1}/{len(recipes)}: {s_count} solvent + {a_count} anion ({cat})")
            
            # 为这个配方创建临时目录
            recipe_dir = os.path.join(output_dir, f"recipe_{idx+1}")
            
            # 调用 cluster_factory.entry，精确匹配这个配方
            # solv_counts 只包含这一个数量，agg_anion_counts 也只包含这一个
            try:
                stats = cluster_factory.entry(
                    solvents=solvent_args,
                    anions=anion_args if anion_args else None,
                    out_dir=recipe_dir,
                    ion=ion_name,
                    solv_counts=(s_count,),  # 精确指定溶剂数量
                    agg_anion_counts=(a_count,) if a_count > 1 else (2,),  # AGG 时精确指定
                    max_total_ligands=s_count + a_count,
                    categories=(cat,),  # 只构建这一个类型
                    optimize_result=False,  # 先不优化，最后统一优化
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    verbose=False,
                    show_progress=False
                )
                
                # 将结果合并到总数据库
                recipe_db = os.path.join(recipe_dir, "all.db")
                if os.path.exists(recipe_db):
                    with connect(recipe_db) as src_db:
                        src_count = src_db.count()
                        logger.info(f"[Batch Build] Recipe {idx+1}: Source DB has {src_count} structure(s)")
                        
                        with connect(all_clusters_db_path) as dst_db:
                            for row in src_db.select():
                                atoms = row.toatoms()
                                kvp = row.key_value_pairs.copy() if row.key_value_pairs else {}
                                data = row.data.copy() if row.data else {}
                                # 添加配方索引信息
                                kvp['recipe_index'] = idx + 1
                                dst_db.write(atoms, data=data, **kvp)
                    
                    # 使用实际从源数据库读取的数量
                    total_built += src_count
                    logger.info(f"[Batch Build] Recipe {idx+1}: Merged {src_count} cluster(s), total now: {total_built}")
                else:
                    total_failed += 1
                    logger.warning(f"[Batch Build] Recipe {idx+1}: No output DB found")
                    
            except Exception as e:
                total_failed += 1
                logger.error(f"[Batch Build] Recipe {idx+1} failed: {e}")
        
        if total_built == 0:
            return json.dumps({"success": False, "msg": "No clusters were built from any recipe."})
        
        # 验证合并后的数据库实际结构数量
        with connect(all_clusters_db_path) as check_db:
            actual_count = check_db.count()
            logger.info(f"[Batch Build] Merged DB verification: {actual_count} structures in {all_clusters_db_path}")
            if actual_count != total_built:
                logger.warning(f"[Batch Build] Mismatch! Expected {total_built}, got {actual_count}")
        
        # 统一进行 UMA 优化
        logger.info(f"[Batch Build] Starting UMA optimization for {actual_count} clusters...")
        
        from emolagent.core import uma_optimizer
        
        final_opt_dir = os.path.join(output_dir, "final_optimized")
        optimized_db_path = uma_optimizer.entry(
            input_db=all_clusters_db_path,
            workspace=final_opt_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False,
            show_progress=False
        )
        
        logger.info(f"[Batch Build] Optimization complete. Output: {optimized_db_path}")
        
        if optimized_db_path and os.path.exists(optimized_db_path):
            return json.dumps({
                "success": True, 
                "optimized_db": optimized_db_path,
                "stats": {"built": total_built, "failed": total_failed},
                "recipes_count": len(recipes),
                "msg": f"Successfully built {total_built} clusters from {len(recipes)} recipes."
            })
        else:
            # 如果优化失败，返回原始数据库
            return json.dumps({
                "success": True,
                "optimized_db": all_clusters_db_path,
                "stats": {"built": total_built, "failed": total_failed},
                "recipes_count": len(recipes),
                "msg": f"Built {total_built} clusters (optimization may have failed)."
            })

    except Exception as e:
        logger.exception("Error occurred in build_multiple_clusters")
        return json.dumps({"success": False, "msg": f"Error: {str(e)}"})


# ==========================================
# 电子结构推断 (Infer + DM Analysis)
# ==========================================

# ==========================================
# GPU 任务槽管理器 (限制并发推理任务数，带文件锁)
# ==========================================

import fcntl
import threading

from emolagent.utils.config import GPUConfig

# GPU 配置（从配置文件加载）
MAX_TASKS_PER_GPU = GPUConfig.get_max_tasks_per_gpu()
AVAILABLE_GPUS = GPUConfig.get_available_gpus()
MAX_CONCURRENT_INFER_TASKS = GPUConfig.get_max_concurrent_tasks()

# 任务槽目录（用于存放锁文件）
_TASK_SLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".task_slots")

# 全局锁文件路径（用于原子性操作）
_GLOBAL_LOCK_FILE = os.path.join(_TASK_SLOT_DIR, ".global.lock")

# 线程本地存储，用于跟踪当前任务的上下文
_task_context = threading.local()


def set_task_context(user_id: str = None, task_id: str = None):
    """设置当前线程的任务上下文，用于日志追踪。"""
    _task_context.user_id = user_id
    _task_context.task_id = task_id


def get_task_context() -> tuple[str | None, str | None]:
    """获取当前线程的任务上下文。"""
    return (
        getattr(_task_context, 'user_id', None),
        getattr(_task_context, 'task_id', None)
    )


def _log_with_context(level: str, message: str):
    """带上下文的日志记录。"""
    user_id, task_id = get_task_context()
    context_prefix = ""
    if user_id or task_id:
        parts = []
        if user_id:
            parts.append(f"user={user_id}")
        if task_id:
            parts.append(f"task={task_id[:16]}")  # 截断 task_id
        context_prefix = f"[{' '.join(parts)}] "
    
    full_message = f"{context_prefix}{message}"
    
    if level == "debug":
        logger.debug(full_message)
    elif level == "info":
        logger.info(full_message)
    elif level == "warning":
        logger.warning(full_message)
    elif level == "error":
        logger.error(full_message)


def _get_task_slot_dir() -> str:
    """获取任务槽目录，如果不存在则创建。"""
    if not os.path.exists(_TASK_SLOT_DIR):
        os.makedirs(_TASK_SLOT_DIR, exist_ok=True)
    return _TASK_SLOT_DIR


def _get_global_lock() -> int:
    """
    获取全局锁文件句柄。
    
    Returns:
        锁文件的文件描述符
    """
    slot_dir = _get_task_slot_dir()
    lock_path = os.path.join(slot_dir, ".global.lock")
    
    # 使用 os.open 确保文件存在
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    return fd


def _cleanup_stale_slots_locked(slot_dir: str) -> None:
    """
    清理过期的任务槽（进程已不存在的槽）。
    注意：调用此函数前必须已经持有全局锁！
    """
    for slot_file in os.listdir(slot_dir):
        if slot_file.startswith("slot_") and slot_file.endswith(".lock"):
            slot_path = os.path.join(slot_dir, slot_file)
            try:
                with open(slot_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        parts = content.split('|')
                        pid = int(parts[0])
                        # 检查进程是否还存在
                        try:
                            os.kill(pid, 0)  # 发送信号 0 来检查进程是否存在
                        except OSError:
                            # 进程不存在，删除这个槽
                            os.remove(slot_path)
                            _log_with_context("info", f"[TaskSlot] Cleaned up stale slot: {slot_file} (PID {pid} no longer exists)")
            except Exception as e:
                # 无法读取或解析，尝试删除
                try:
                    os.remove(slot_path)
                    _log_with_context("warning", f"[TaskSlot] Removed corrupted slot file: {slot_file}")
                except Exception:
                    pass


def _count_slots_by_gpu_locked(slot_dir: str) -> dict[int, int]:
    """
    统计每张 GPU 上的活跃任务槽数量。
    注意：调用此函数前必须已经持有全局锁！
    
    Returns:
        {gpu_id: count} 的字典
    """
    gpu_counts = {gpu_id: 0 for gpu_id in AVAILABLE_GPUS}
    
    for slot_file in os.listdir(slot_dir):
        if slot_file.startswith("slot_") and slot_file.endswith(".lock"):
            slot_path = os.path.join(slot_dir, slot_file)
            try:
                with open(slot_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        parts = content.split('|')
                        if len(parts) >= 4:
                            gpu_id = int(parts[3])
                            if gpu_id in gpu_counts:
                                gpu_counts[gpu_id] += 1
            except Exception:
                pass
    
    return gpu_counts


def _select_gpu_locked(slot_dir: str) -> int | None:
    """
    选择负载最低的 GPU。
    注意：调用此函数前必须已经持有全局锁！
    
    Returns:
        选中的 GPU ID，如果所有 GPU 都满载则返回 None
    """
    gpu_counts = _count_slots_by_gpu_locked(slot_dir)
    
    # 找到负载最低的 GPU
    min_count = float('inf')
    selected_gpu = None
    
    for gpu_id in AVAILABLE_GPUS:
        count = gpu_counts.get(gpu_id, 0)
        if count < MAX_TASKS_PER_GPU and count < min_count:
            min_count = count
            selected_gpu = gpu_id
    
    return selected_gpu


def _acquire_task_slot(task_id: str, user_id: str = None) -> tuple[str | None, int | None]:
    """
    尝试获取一个任务槽（原子操作，带文件锁）。
    
    Args:
        task_id: 任务标识符
        user_id: 用户标识符（用于日志）
        
    Returns:
        (槽文件路径, 分配的GPU ID)，失败返回 (None, None)
    """
    slot_dir = _get_task_slot_dir()
    lock_fd = _get_global_lock()
    
    try:
        # 获取排他锁（阻塞）
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        
        try:
            # 在锁内执行所有操作
            _cleanup_stale_slots_locked(slot_dir)
            
            # 选择 GPU
            selected_gpu = _select_gpu_locked(slot_dir)
            
            if selected_gpu is None:
                gpu_counts = _count_slots_by_gpu_locked(slot_dir)
                total_active = sum(gpu_counts.values())
                _log_with_context("warning", 
                    f"[TaskSlot] Cannot acquire slot: all GPUs at capacity "
                    f"({total_active}/{MAX_CONCURRENT_INFER_TASKS} tasks, "
                    f"GPU loads: {gpu_counts})")
                return None, None
            
            # 创建新的槽文件
            pid = os.getpid()
            timestamp = time.time()
            slot_filename = f"slot_{pid}_{int(timestamp * 1000)}.lock"
            slot_path = os.path.join(slot_dir, slot_filename)
            
            # 写入槽信息：PID|task_id|timestamp|gpu_id|user_id
            with open(slot_path, 'w') as f:
                f.write(f"{pid}|{task_id}|{timestamp}|{selected_gpu}|{user_id or 'unknown'}")
            
            gpu_counts = _count_slots_by_gpu_locked(slot_dir)
            total_active = sum(gpu_counts.values())
            _log_with_context("info", 
                f"[TaskSlot] Acquired slot on GPU {selected_gpu}: {slot_filename} "
                f"(total {total_active}/{MAX_CONCURRENT_INFER_TASKS}, GPU loads: {gpu_counts})")
            
            return slot_path, selected_gpu
            
        finally:
            # 释放锁
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def _release_task_slot(slot_path: str) -> None:
    """
    释放一个任务槽（原子操作，带文件锁）。
    
    Args:
        slot_path: 槽文件路径
    """
    if not slot_path or not os.path.exists(slot_path):
        return
        
    slot_dir = _get_task_slot_dir()
    lock_fd = _get_global_lock()
    
    try:
        # 获取排他锁
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        
        try:
            if os.path.exists(slot_path):
                os.remove(slot_path)
            
            gpu_counts = _count_slots_by_gpu_locked(slot_dir)
            total_remaining = sum(gpu_counts.values())
            _log_with_context("info", 
                f"[TaskSlot] Released slot: {os.path.basename(slot_path)} "
                f"(now {total_remaining}/{MAX_CONCURRENT_INFER_TASKS}, GPU loads: {gpu_counts})")
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    except Exception as e:
        _log_with_context("warning", f"[TaskSlot] Failed to release slot: {e}")
    finally:
        try:
            os.close(lock_fd)
        except Exception:
            pass


def get_task_queue_status() -> dict:
    """
    获取当前任务队列状态（原子操作，带文件锁）。
    
    Returns:
        包含当前活跃任务数和最大任务数的字典
    """
    slot_dir = _get_task_slot_dir()
    lock_fd = _get_global_lock()
    
    try:
        # 获取共享锁（允许并发读取）
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        
        try:
            _cleanup_stale_slots_locked(slot_dir)
            gpu_counts = _count_slots_by_gpu_locked(slot_dir)
            total_active = sum(gpu_counts.values())
            
            return {
                "active_tasks": total_active,
                "max_tasks": MAX_CONCURRENT_INFER_TASKS,
                "available_slots": MAX_CONCURRENT_INFER_TASKS - total_active,
                "can_accept": total_active < MAX_CONCURRENT_INFER_TASKS,
                "gpu_loads": gpu_counts,
                "gpus": AVAILABLE_GPUS,
            }
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        try:
            os.close(lock_fd)
        except Exception:
            pass


def _dm_infer_worker(ase_db_path, model_path, output_dir, gpu_id, user_id, task_id, queue):
    """
    子进程 worker，用于 DM 推断。
    隔离 os.chdir 操作以防止主进程的线程安全问题。
    
    Args:
        ase_db_path: ASE 数据库路径
        model_path: 模型检查点路径
        output_dir: 输出目录
        gpu_id: 分配的 GPU ID
        user_id: 用户 ID（用于日志）
        task_id: 任务 ID（用于日志）
        queue: 用于返回结果的队列
    """
    # ============================================================
    # 关键：必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES！
    # 否则 CUDA 上下文会绑定到默认设备（通常是 GPU 0）
    # ============================================================
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["LC_ALL"] = "C"
    os.environ["LANG"] = "C"
    
    # 现在才能安全地导入 torch 和其他 CUDA 相关模块
    import json
    import torch
    import pandas as pd
    from emoles.py3Dmol import cubes_2_htmls
    from emoles.inference import infer_entry
    
    # 导入本模块的工具函数（这些不会触发 CUDA 初始化）
    from emolagent.core.tools import set_task_context, _log_with_context, NumpyEncoder
    from emolagent.utils.logger import logger
    
    try:
        # 设置任务上下文用于日志
        set_task_context(user_id=user_id, task_id=task_id)

        ase_db_path = os.path.abspath(ase_db_path)
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        os.chdir(output_dir)
        
        # 验证 CUDA 设备
        if torch.cuda.is_available():
            _log_with_context("info", f">>> CUDA device count: {torch.cuda.device_count()}, using device: {torch.cuda.current_device()} (physical GPU {gpu_id})")
        
        # Step 1: DPTB Inference
        _log_with_context("info", f">>> Starting DPTB Inference on GPU {gpu_id}...")
        infer_entry.dptb_infer_from_ase_db(
            ase_db_path=ase_db_path,
            out_path=output_dir,
            checkpoint_path=model_path,
            limit=50,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        possible_npy_dirs = [
            os.path.join(output_dir, 'inference', 'npy'),
            os.path.join(output_dir, 'results')
        ]
        npy_results_dir = os.path.join(output_dir, 'results')
        for d in possible_npy_dirs:
            if os.path.exists(d):
                npy_results_dir = d
                break
        
        _log_with_context("info", f"Using NPY results dir: {npy_results_dir}")

        # Step 2: DM Inference
        _log_with_context("info", ">>> Starting DM Property Inference (via infer_entry)...")
        
        summary_data = infer_entry.dm_infer_entry(
            abs_ase_path=ase_db_path,
            results_folder_path=npy_results_dir,
            dm_filename="predicted.npy",
            convention="def2svp",
            calc_esp_flag=True,
            calc_electronic_flag=True,
            save_cube_info=True,
            n_save_cube_items=5,
            cube_grid=40,
            summary_filename="inference_summary.npz"
        )
        
        _log_with_context("info", ">>> Generating HTML visualizations for Cube files...")
        try:
            cubes_2_htmls(output_dir, iso_value=0.03)
        except Exception as e:
            _log_with_context("warning", f"Warning: HTML generation failed: {e}")

        # Step 3: Format Output
        csv_path = os.path.join(output_dir, "analysis_summary.csv")
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(csv_path, index=False)
        
        _log_with_context("info", f">>> Inference completed successfully on GPU {gpu_id}")
            
        result = json.dumps({
            "success": True,
            "csv_path": csv_path,
            "results_dir": npy_results_dir,
            "output_dir": output_dir,
            "gpu_id": gpu_id,
            "data_preview": summary_data[:3] if summary_data else []
        }, cls=NumpyEncoder)
        
        queue.put({"status": "success", "data": result})

    except Exception as e:
        _log_with_context("error", f"Error occurred: {e}")
        logger.exception("Error occurred")
        queue.put({"status": "error", "message": f"Pipeline Failed (GPU {gpu_id}): {str(e)}"})


def run_dm_infer_pipeline(
    ase_db_path: str, 
    model_path: str, 
    output_dir: str, 
    wait_timeout: float = 300.0,
    user_id: str = None
) -> str:
    """
    运行完整的推断流水线（带并发控制和 GPU 轮询分配）。
    
    Args:
        ase_db_path: ASE 数据库路径
        model_path: 模型检查点路径
        output_dir: 输出目录
        wait_timeout: 等待任务槽的超时时间（秒），默认 300 秒
        user_id: 用户 ID（用于日志追踪）
        
    Returns:
        JSON 格式的结果字符串
    """
    task_id = f"{os.path.basename(output_dir)}_{time.time_ns()}"
    
    # 设置任务上下文用于日志
    set_task_context(user_id=user_id, task_id=task_id)
    
    # 尝试获取任务槽（带等待机制）
    slot_path = None
    gpu_id = None
    wait_start = time.time()
    wait_interval = 5.0  # 每 5 秒检查一次
    
    while slot_path is None:
        slot_path, gpu_id = _acquire_task_slot(task_id, user_id)
        
        if slot_path is None:
            elapsed = time.time() - wait_start
            if elapsed >= wait_timeout:
                status = get_task_queue_status()
                _log_with_context("error", 
                    f"Task queue full, timeout after {wait_timeout}s. "
                    f"GPU loads: {status.get('gpu_loads', {})}")
                return json.dumps({
                    "success": False,
                    "msg": f"任务队列已满，等待超时（{wait_timeout}秒）。当前有 {status['active_tasks']} 个任务正在运行，最大并发数为 {status['max_tasks']}。GPU 负载: {status.get('gpu_loads', {})}。请稍后再试。"
                })
            
            # 记录等待状态
            status = get_task_queue_status()
            remaining_wait = wait_timeout - elapsed
            _log_with_context("info", 
                f"[TaskSlot] Waiting for slot... "
                f"({status['active_tasks']}/{status['max_tasks']} in use, "
                f"GPU loads: {status.get('gpu_loads', {})}, "
                f"waited {elapsed:.1f}s, timeout in {remaining_wait:.1f}s)")
            time.sleep(wait_interval)
    
    _log_with_context("info", f"Starting inference pipeline on GPU {gpu_id}")
    
    # 成功获取槽，开始执行任务
    try:
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        
        p = ctx.Process(
            target=_dm_infer_worker, 
            args=(ase_db_path, model_path, output_dir, gpu_id, user_id, task_id, queue)
        )
        p.start()
        p.join()
        
        if not queue.empty():
            res = queue.get()
            if res['status'] == 'success':
                _log_with_context("info", f"Inference completed successfully on GPU {gpu_id}")
                return res['data']
            else:
                _log_with_context("error", f"Inference failed: {res['message']}")
                return res['message']
        else:
            _log_with_context("error", "Subprocess did not return any data")
            return "Pipeline Failed: Subprocess did not return any data."
    finally:
        # 无论成功还是失败，都要释放任务槽
        _release_task_slot(slot_path)


def compress_directory(dir_path: str, output_path_base: str) -> str:
    """
    压缩文件夹。
    
    Args:
        dir_path: 要压缩的文件夹路径
        output_path_base: 输出文件路径（不含后缀）
        
    Returns:
        压缩文件的完整路径
    """
    return shutil.make_archive(output_path_base, 'zip', dir_path)
