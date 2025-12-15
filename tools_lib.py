import os
import torch
import json
import re
import shutil
from ase.db import connect
from emoles.dptb import process_dataset, setup_dataset, setup_db_path, setup_output_directory
from dptb.data.dataloader import DataLoader
from dptb.data import build_dataset
from dptb.nn import build_model
import e3nn
from emoles.loss import get_mae_from_npy
from emoles.pyscf import generate_cube_files
from emoles.py3Dmol import cubes_2_htmls
from pprint import pprint

e3nn.set_optimization_defaults(jit_script_fx=False)

# --- 工具 1: 推理 (Inference) ---
def run_dptb_inference(data_root, model_path, output_dir="output", db_name="dump.db"):
    """
    运行 DPTB 模型推理。
    Args:
        data_root: LMDB 数据文件夹路径
        model_path: .pth 模型路径
        output_dir: 输出文件夹
        db_name: 输出的数据库文件名
    """
    try:
        
        # 路径处理
        setup_output_directory(output_dir)
        abs_db_path = setup_db_path(db_name)

        # 构建数据集
        dataset = build_dataset(
            root=data_root,
            type="LMDBDataset",
            prefix="data.0",
            get_overlap=False,
            train_w_charge=True,
            get_Hamiltonian=True,
            basis={
                "Li": "3s2p",
                "C": "3s2p1d",
                "N": "3s2p1d",
                "S": "4s3p1d",
                "O": "3s2p1d",
                "F": "3s2p1d",
                "H": "2s1p",
                "B": "3s2p1d",
                "P": "4s3p1d"
            },
            r_max={
                "Li": 12,
                "C": 10,
                "N": 10,
                "S": 8,
                "O": 9,
                "F": 8,
                "H": 8.5,
                "B": 9,
                "P": 8
                },
        )
        
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        model = build_model(model_path, common_options={"device": "cuda"})

        # 执行处理
        time_per_item, loss_arr = process_dataset(
            data_loader=data_loader,
            dataset=dataset,
            model=model,
            device=device,
            output_dir=output_dir,
            db_path=abs_db_path,
            save_data_flag=True,
            save_npy_flag=True,
            save_db_predicted=False,
            save_db_original=False,
            max_items=None
        )
        
        return f"推理完成。DB保存至: {abs_db_path}, NPY保存至: {output_dir}, 平均耗时: {time_per_item:.4f}s"

    except Exception as e:
        return f"Error in run_dptb_inference: {str(e)}"

# --- 工具 2: 更新元数据 (Update Metadata) ---
# def update_db_metadata(input_db, input_paths_file, output_db="updated.db"):
#     """
#     根据路径文件更新数据库中的 spin 和 charge。
#     """
#     try:
#         path_pattern = re.compile(r'spin_(\d+)_charge_([+-]?\d+)')
#         count = 0
        
#         with connect(input_db) as src_db, connect(output_db) as dst_db, open(input_paths_file) as f:
#             for row, path in zip(src_db.select(), f):
#                 match = path_pattern.search(path)
#                 if not match:
#                     continue # 或者记录警告

#                 spin = int(match.group(1))
#                 charge = int(match.group(2))
                
#                 data = row.data.copy()
#                 data.update({"spin": spin, "charge": charge, "path": path.strip()})
#                 dst_db.write(row.toatoms(), data=data)
#                 count += 1
                
#         return f"元数据更新完成。已写入 {output_db}, 共处理 {count} 条记录。"
    
#     except Exception as e:
#         return f"Error in update_db_metadata: {str(e)}"

# --- 工具 2: 可视化与分析 (Viz & Analysis) ---
def generate_viz_report(abs_ase_path, npy_folder_path):
    """
    生成 Cube 文件、HTML 可视化以及 MAE 报告。
    """
    try:
        n_grid = 75
        convention = 'def2svp'
        
        # 临时路径
        os.makedirs(npy_folder_path, exist_ok=True)
        temp_data_file = os.path.abspath('temp_data.npz')
        cube_dump_place = os.path.abspath('cubes')
        if os.path.exists(cube_dump_place):
            shutil.rmtree(cube_dump_place)
        # os.makedirs(cube_dump_place, exist_ok=True)
        if os.path.exists(temp_data_file):
            os.remove(temp_data_file)
        os.makedirs(cube_dump_place)

        # 1. 计算 MAE
        total_error_dict = get_mae_from_npy(
            abs_ase_path=abs_ase_path, 
            npy_folder_path=npy_folder_path,
            temp_data_file=temp_data_file, 
            save_summary=True,
            united_overlap_flag=True, 
            convention=convention
        )

        # 2. 保存 JSON 报告
        pprint(total_error_dict)
        json_path = 'test_results.json'
        with open(json_path, 'w') as f:
            json.dump(total_error_dict, f, indent=2)

        # 3. 生成 Cube
        generate_cube_files(temp_data_file, n_grid, cube_dump_place, 5)

        # 4. 生成 HTML
        cubes_2_htmls(cube_dump_place, 0.03)

        return f"分析完成。MAE 报告: {json_path}, Cube/HTML 文件位于: {cube_dump_place}"

    except Exception as e:
        return f"Error in generate_viz_report: {str(e)}"
