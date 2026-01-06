import os
import shutil
import json
import time
import traceback
import re
import multiprocessing
import pandas as pd
import torch
import numpy as np
from ase.db import connect
from emoles.py3Dmol import cube_2_html, cubes_2_htmls
import cluster_factory
from emoles.inference import infer_entry

# ==========================================
# 常量定义
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVENT_DB_PATH = os.path.join(BASE_DIR, "resources/db/cut_10_common.db")
ANION_DB_PATH = os.path.join(BASE_DIR, "resources/db/anions.db")

class NumpyEncoder(json.JSONEncoder):
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
    :param query_name: 分子名称 (如 "DME", "FSI")
    :param mol_type: "solvent" 或 "anion"
    :param output_dir: 结果输出目录
    :return: 结果信息的 JSON 字符串 (包含是否找到，以及生成的 temp_db 路径)
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
        # 第一次遍历：寻找
        for row in src_db.select():
            # 获取名字，通常在 key_value_pairs['name'] 或 row.name
            a_name = row.get('name', '')
            if not a_name and row.key_value_pairs:
                a_name = row.key_value_pairs.get('name', '')
            
            if not a_name:
                continue

            # 逻辑 1: 精确匹配
            if a_name == query_norm:
                found_row = row
                exact_match = True
                break
            
            # 逻辑 2: 去掉括号后的模糊匹配 (例如 "DME(123)" -> "DME")
            short_name = a_name.split('(')[0].strip()
            if short_name == query_norm:
                found_row = row
                if not exact_match: 
                    exact_match = False

        if found_row:
            # 提取分子并保存到一个临时的单分子 DB 中，供 cluster_factory 使用
            safe_name = re.sub(r'[^A-Za-z0-9]', '_', query_name)
            unique = f"{time.time_ns()}_{os.getpid()}"
            temp_db_name = f"found_{mol_type}_{safe_name}_{unique}.db"
            temp_db_path = os.path.abspath(os.path.join(output_dir, temp_db_name))

            atoms = found_row.toatoms()
            kvp = (found_row.key_value_pairs or {}).copy()
            data = (found_row.data or {}).copy()
            kvp.pop("name", None)
                
            with connect(temp_db_path) as tmp_db:
                tmp_db.write(atoms, name=query_name, data=data, **kvp) # 写入时重命名为用户查询的名字，方便后续识别
            
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
    solvents_config: list, # [{"name": "DME", "path": "path/to/dme.db", "count": 3, "smiles": ""}]
    anions_config: list,   # [{"name": "FSI", "path": "path/to/fsi.db", "count": 1, "smiles": ""}]
    output_dir: str
):
    """
    调用 Cluster Factory 进行建模，并自动进行 UMA 优化。
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 1. 准备 Solvents 参数 (收集所有溶剂输入)
        solvent_args = []
        s_counts = set()
        
        if solvents_config:
            for s_item in solvents_config:
                # 收集路径或SMILES
                if s_item.get('path') and os.path.exists(s_item.get('path')):
                    solvent_args.append(s_item['path'])
                elif s_item.get('smiles'):
                    solvent_args.append(s_item['smiles'])
                
                # 收集配位数
                if s_item.get('count'):
                    try:
                        s_counts.add(int(s_item['count']))
                    except ValueError:
                        pass

        if not solvent_args:
            return "Error: No solvent information provided."

        # 2. 准备 Anions 参数 (收集所有阴离子输入)
        anion_args = []
        a_counts_agg = set() # 用于 AGG 的阴离子数量 (>1)
        has_cip_request = False # 是否有 count=1 的请求
        
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
            # 没有阴离子 -> SSIP
            cats.append("SSIP")
        else:
            # 有阴离子，根据数量决定
            if has_cip_request:
                cats.append("CIP")
            if a_counts_agg:
                cats.append("AGG")
            
            # 如果用户提供了阴离子但没指定 count，或者 count 解析失败，默认给个 CIP
            if not cats: 
                cats.append("CIP")

        # 转换 counts 为 tuple，如果为空则使用默认值或报错
        final_s_counts = tuple(sorted(list(s_counts))) if s_counts else (4,) # 默认配位4
        final_a_counts = tuple(sorted(list(a_counts_agg))) if a_counts_agg else (2,)

        # 计算最大配位数用于剪枝 (取最大的溶剂数 + 最大的阴离子数)
        max_s = max(final_s_counts) if final_s_counts else 0
        max_a = max(final_a_counts) if "AGG" in cats else (1 if "CIP" in cats else 0)
        max_total = max_s + max_a

        print(f"Calling Cluster Factory: Solv={solvent_args}, Anion={anion_args}")
        print(f"Plan: Cats={cats}, S_Counts={final_s_counts}, A_Counts(AGG)={final_a_counts}")
        
        # 4. 调用 cluster_factory.entry
        # 直接利用 factory 的逻辑来处理列表组合
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
            # 找里面最新的 db
            preferred = os.path.join(final_opt_dir, "optimized_all.db")
            if os.path.exists(preferred):
                result_db = preferred
            else:
                dbs = [os.path.join(final_opt_dir, f) for f in os.listdir(final_opt_dir) if f.endswith(".db")]
                if dbs:
                    result_db = max(dbs, key=os.path.getmtime)
        
        # 如果没有优化结果 (比如 failed)，尝试找未优化的 all.db
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
        traceback.print_exc()
        return f"Error in build_and_optimize_cluster: {str(e)}"

# ==========================================
# 电子结构推断 (Infer + DM Analysis)
# ==========================================

def _dm_infer_worker(ase_db_path, model_path, output_dir, queue):
    """
    Subprocess worker for DM inference pipeline.
    Isolates os.chdir operations to prevent thread-safety issues in main process.
    """
    try:
        os.environ["LC_ALL"] = "C"
        os.environ["LANG"] = "C"

        # 转换为绝对路径，防止 chdir 后失效
        ase_db_path = os.path.abspath(ase_db_path)
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 在子进程中切换目录，安全
        os.chdir(output_dir)
        if os.getenv("EMOL_DEBUG_CHARGE") == "1":
            try:
                with connect(ase_db_path) as _db:
                    sample = []
                    for i, row in enumerate(_db.select()):
                        kvp = row.key_value_pairs or {}
                        sample.append((row.id, kvp.get("xyz_file"), kvp.get("charge"), kvp.get("n_anion")))
                        if i >= 4:
                            break
                print("[ChargeDebug] first rows:", sample)
            except Exception as e:
                print("[ChargeDebug] failed:", e)

        # ---------------------------
        # Step 1: DPTB Inference
        # ---------------------------
        print(">>> Starting DPTB Inference (Subprocess)...")
        # 调用 infer_entry 中的实现
        infer_entry.dptb_infer_from_ase_db(
            ase_db_path=ase_db_path,
            out_path=output_dir,
            checkpoint_path=model_path,
            limit=50, # 限制数量，避免太慢
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        possible_npy_dirs = [
            os.path.join(output_dir, 'inference', 'npy'),
            os.path.join(output_dir, 'results')
        ]
        npy_results_dir = os.path.join(output_dir, 'results') # Default fallback
        for d in possible_npy_dirs:
            if os.path.exists(d):
                npy_results_dir = d
                break
        
        print(f"Using NPY results dir: {npy_results_dir}")

        # ---------------------------
        # Step 2: DM Inference (Properties & Cubes)
        # ---------------------------
        print(">>> Starting DM Property Inference (via infer_entry)...")
        
        # 使用 infer_entry.dm_infer_entry 进行一站式属性计算
        summary_data = infer_entry.dm_infer_entry(
            abs_ase_path=ase_db_path,
            results_folder_path=npy_results_dir,
            dm_filename="predicted.npy",
            convention="def2svp",
            calc_esp_flag=True,
            calc_electronic_flag=True,
            save_cube_info=True,
            n_save_cube_items=5,   # 只生成前5个Cube
            cube_grid=40,          # 网格精度
            summary_filename="inference_summary.npz"
        )
        
        print(">>> Generating HTML visualizations for Cube files...")
        try:
            cubes_2_htmls(output_dir, iso_value=0.03)
        except Exception as e:
            print(f"Warning: HTML generation failed: {e}")
            traceback.print_exc()

        # ---------------------------
        # Step 3: Format Output
        # ---------------------------
        # 将结果保存为 CSV 方便 Agent 读取
        csv_path = os.path.join(output_dir, "analysis_summary.csv")
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(csv_path, index=False)
            
        result = json.dumps({
            "success": True,
            "csv_path": csv_path,
            "results_dir": npy_results_dir,
            "output_dir": output_dir, # 明确返回根输出目录
            "data_preview": summary_data[:3] if summary_data else []
        }, cls=NumpyEncoder)
        
        queue.put({"status": "success", "data": result})

    except Exception as e:
        traceback.print_exc()
        queue.put({"status": "error", "message": f"Pipeline Failed: {str(e)}"})

def run_dm_infer_pipeline(ase_db_path: str, model_path: str, output_dir: str):
    """
    运行完整的推断流水线：
    1. DPTB 推断 -> 生成 Hamiltonian/DM NPY
    2. DM Infer -> 计算 Dipole, ESP, etc.
    3. Hamiltonian Analysis -> 生成 Cube, HOMO/LUMO
    
    Updated: Uses multiprocessing to avoid os.chdir thread-safety issues.
    """
    # 使用 spawn 方法启动子进程，避免 CUDA 上下文在 fork 时崩溃
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    
    p = ctx.Process(target=_dm_infer_worker, args=(ase_db_path, model_path, output_dir, queue))
    p.start()
    p.join()
    
    if not queue.empty():
        res = queue.get()
        if res['status'] == 'success':
            return res['data']
        else:
            return res['message']
    else:
        return "Pipeline Failed: Subprocess did not return any data."

def compress_directory(dir_path: str, output_path_base: str) -> str:
    """
    压缩文件夹
    :param dir_path: 要压缩的文件夹路径
    :param output_path_base: 输出文件路径（不含后缀，例如 /tmp/res），函数会自动添加 .zip
    :return: 压缩文件的完整路径
    """
    return shutil.make_archive(output_path_base, 'zip', dir_path)