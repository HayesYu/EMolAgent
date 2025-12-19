import os
import shutil
import json
import time
import pickle
import traceback

import numpy as np
import pandas as pd
import lmdb
import torch
from ase.db import connect
from tqdm import tqdm
from ase.units import Hartree

import pyscf
from pyscf import gto, dft, tools
from pyscf.scf.hf import dip_moment, make_rdm1

from dftio.data import _keys
from dptb.nn.hr2hk import HR2HK, HR2HK_Gamma_Only
from dptb.data import AtomicDataset, DataLoader, AtomicData, AtomicDataDict
from dptb.data.build import build_dataset
from dptb.nn.build import build_model
from dptb.utils.tools import j_loader
from dptb.utils.argcheck import collect_cutoffs

from emoles.utils import (
    matrix_transform,
    get_mo_occ,
)
from emoles.inference.common_tools import atom_2_smile, calculate_esp_from_dm, extract_model_params
from emoles.py3Dmol import cubes_2_htmls

def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
    """
    Solve generalized eigenvalue problem HC = SCE.
    Returns energies and coefficients.
    """
    if overlap_matrix.ndim == 2:
        overlap_matrix = overlap_matrix[None, ...]
    if full_hamiltonian.ndim == 2:
        full_hamiltonian = full_hamiltonian[None, ...]

    eigvals, eigvecs = np.linalg.eigh(overlap_matrix)
    eps = 1e-8 * np.ones_like(eigvals)
    eigvals = np.where(eigvals > 1e-8, eigvals, eps)
    frac_overlap = eigvecs / np.sqrt(eigvals[:, np.newaxis])

    Fs = np.matmul(
        np.matmul(np.transpose(frac_overlap, (0, 2, 1)), full_hamiltonian),
        frac_overlap,
    )
    orbital_energies, orbital_coefficients = np.linalg.eigh(Fs)
    orbital_coefficients = frac_overlap @ orbital_coefficients

    return orbital_energies[0], orbital_coefficients[0]


def get_electronic_properties(mol, ham=None, overlap=None, dm=None):
    """
    Extract electronic properties (Energies, Orbitals, Gap) from Ham+Overlap.
    """
    if ham is None:
        if dm is None:
            raise ValueError("Must provide either Hamiltonian or Density Matrix")
        mf = dft.RKS(mol)
        mf.xc = "b3lyp"
        ham = mf.get_fock(dm=dm)
        if overlap is None:
            overlap = mf.get_ovlp()

    if overlap is None:
        overlap = mol.intor("int1e_ovlp")

    ham_2d = ham[0] if ham.ndim == 3 else ham
    ov_2d = overlap[0] if overlap.ndim == 3 else overlap

    energies, coeffs = cal_orbital_and_energies(
        overlap_matrix=ov_2d, full_hamiltonian=ham_2d
    )

    n_electrons = mol.tot_electrons()
    homo_idx = int(n_electrons / 2) - 1
    lumo_idx = homo_idx + 1

    mo_occ = get_mo_occ(full_len=len(energies), occ_len=homo_idx + 1)

    results = {
        "HOMO": energies[homo_idx],
        "LUMO": energies[lumo_idx],
        "GAP": energies[lumo_idx] - energies[homo_idx],
        "hamiltonian": ham_2d,
        "overlap": ov_2d,
        "mo_occ": mo_occ,
        "orbital_coefficients": coeffs,
        "HOMO_coefficients": coeffs[:, homo_idx],
        "LUMO_coefficients": coeffs[:, lumo_idx],
        "occupied_orbital_energy": energies[: homo_idx + 1],
        "all_energies": energies,
        "homo_idx": homo_idx,
        "lumo_idx": lumo_idx
    }
    return results


def _load_npy_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return np.load(path)

# DPTB Inference Logic
def ase_db_2_dummy_dptb_lmdb(ase_db_path: str, dptb_lmdb_path: str):
    dptb_lmdb_path = os.path.join(dptb_lmdb_path, "data.{}.lmdb".format(os.getpid()))
    os.makedirs(dptb_lmdb_path, exist_ok=True)
    lmdb_env = lmdb.open(dptb_lmdb_path, map_size=1048576000000, lock=True)
    with connect(ase_db_path) as src_db:
        with lmdb_env.begin(write=True) as txn:
            entries = 0
            for idx, a_row in enumerate(src_db.select()):
                an_atoms = a_row.toatoms()
                data_dict = {
                    _keys.ATOMIC_NUMBERS_KEY: an_atoms.numbers,
                    _keys.PBC_KEY: np.array([False, False, False]),
                    _keys.POSITIONS_KEY: an_atoms.positions.reshape(1, -1, 3).astype(np.float32),
                    _keys.CELL_KEY: an_atoms.cell.reshape(1, 3, 3).astype(np.float32),
                    "charge": a_row.data.get('charge', 0),
                    "idx": idx,
                    "nf": 0
                }
                data_bytes = pickle.dumps(data_dict)
                txn.put(entries.to_bytes(length=4, byteorder='big'), data_bytes)
                entries += 1
    lmdb_env.close()


def save_info_2_npy(folder_path, idx, batch_info, model, device, has_overlap):
    cwd_ = os.getcwd()
    os.chdir(folder_path)
    if not os.path.exists(str(idx)):
        os.makedirs(str(idx))
    os.chdir(str(idx))

    batch_info['kpoint'] = torch.tensor([0.0, 0.0, 0.0], device=device)

    a_ham_hr2hk = HR2HK_Gamma_Only(
        idp=model.idp,
        edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
        node_field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field=AtomicDataDict.HAMILTONIAN_KEY,
        overlap=True,
        device=device
    )
    ham_out_data = a_ham_hr2hk.forward(batch_info)
    a_ham = ham_out_data[AtomicDataDict.HAMILTONIAN_KEY]
    ham_ndarray = a_ham.real.cpu().numpy()
    np.save('predicted.npy', ham_ndarray)

    if has_overlap:
        an_overlap_hr2hk = HR2HK(
            idp=model.idp,
            edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
            node_field=AtomicDataDict.NODE_OVERLAP_KEY,
            out_field=AtomicDataDict.OVERLAP_KEY,
            overlap=True,
            device=device
        )
        overlap_out_data = an_overlap_hr2hk.forward(batch_info)
        an_overlap = overlap_out_data[AtomicDataDict.OVERLAP_KEY]
        overlap_ndarray = an_overlap.real.cpu().numpy()
        np.save('predicted_overlap.npy', overlap_ndarray)

    os.chdir(cwd_)


def dptb_infer_from_ase_db(ase_db_path: str, out_path: str,
                           checkpoint_path: str,
                           limit: int = 200, device: str = 'cuda'):
    import e3nn
    e3nn.set_optimization_defaults(jit_script_fx=False)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = build_model(checkpoint=checkpoint_path)
    model.to(device)
    basis, r_max = extract_model_params(model)
    abs_out_path = os.path.abspath(out_path)
    ase_db_path = os.path.abspath(ase_db_path)

    if os.path.exists(abs_out_path):
        shutil.rmtree(abs_out_path)
    os.makedirs(abs_out_path)

    lmdb_path = os.path.join(abs_out_path, 'lmdb')
    npy_path = os.path.join(abs_out_path, 'npy')
    os.makedirs(npy_path)

    ase_db_2_dummy_dptb_lmdb(ase_db_path, lmdb_path)

    reference_info = {
        "root": lmdb_path,
        "prefix": "data",
        "type": "LMDBDataset",
        "get_DM": False,
        "get_Hamiltonian": False,
        "get_overlap": False
    }
    reference_datasets = build_dataset(basis=basis, r_max=r_max, train_w_charge=True, **reference_info)
    reference_loader = DataLoader(dataset=reference_datasets, batch_size=1, shuffle=False)

    start_time = time.time()
    count = 0
    for idx, a_ref_batch in enumerate(reference_loader):
        if idx >= limit:
            break
        batch = a_ref_batch.to(device)
        batch = AtomicData.to_AtomicDataDict(batch)
        with torch.no_grad():
            predicted_data = model(batch)
        
        save_info_2_npy(folder_path=npy_path, idx=idx, batch_info=predicted_data, model=model, device=device, has_overlap=False)
        count = idx

    end_time = time.time()
    second_per_item = (end_time - start_time) / max(1, count + 1)
    return f"推理完成。NPY文件已保存至: {npy_path}, 平均耗时: {second_per_item:.4f} s/item"

# Post-Processing: Hamiltonian -> Energy/Cube
def get_ham_info_from_npy(ase_db_path,
                          npy_folder_path,
                          output_base_dir=None,
                          convert_smiles_flag=False,
                          convention='def2svp',
                          mol_charge=0,
                          pred_ham_filename='predicted.npy',
                          max_items: int = 300,
                          max_cube_save: int = 5,
                          cube_grid: int = 75):
    """
    Read Hamiltonian from npy -> Rotate -> Solve with PySCF Overlap ->
    Get Energies/Coeffs -> Draw HOMO/LUMO -> Save Summary.
    """
    print('Start Hamiltonian postprocess')

    if convention == '6311gdp':
        basis = '6-311+g(d,p)'
        back_convention = 'back_2_thu_pyscf'
    else:
        basis = 'def2svp'
        back_convention = 'back2pyscf'

    npy_folder_path = os.path.abspath(npy_folder_path)
    cwd_ = os.getcwd()

    if output_base_dir is None:
        output_base_dir = cwd_
    else:
        output_base_dir = os.path.abspath(output_base_dir)

    all_ham_info = []
    start_time = time.time()

    with connect(ase_db_path) as db:
        total = db.count()
        limit = min(total, max_items)
        for idx, a_row in tqdm(enumerate(db.select()), total=limit):
            if idx >= limit:
                break

            npy_sub_dir = os.path.join(npy_folder_path, f'{idx}')
            if not os.path.exists(npy_sub_dir):
                continue
            
            os.chdir(npy_sub_dir)

            try:
                atom_nums = a_row.numbers
                an_atoms = a_row.toatoms()
                if convert_smiles_flag:
                    smiles = atom_2_smile(an_atoms)
                else:
                    smiles = "N/A"

                # --- Charge & Spin Logic ---
                current_mol_charge = a_row.data.get("charge", mol_charge)
                sum_of_atomic_numbers = an_atoms.get_atomic_numbers().sum()
                total_electrons = sum_of_atomic_numbers - current_mol_charge
                mol_spin = total_electrons % 2

                # 1. Load Hamiltonian
                pred_ham = _load_npy_safe(pred_ham_filename)

                # 2. Transform Basis (Rotation)
                pred_ham_pyscf = matrix_transform(pred_ham, atom_nums, convention=back_convention)

                # 3. Build PySCF Mole & Get Overlap
                mol = pyscf.gto.Mole()
                t = [[atom_nums[atom_idx], an_atom.position]
                     for atom_idx, an_atom in enumerate(an_atoms)]
                mol.charge = current_mol_charge
                mol.spin = mol_spin
                mol.build(verbose=0, atom=t, basis=basis, unit='ang')

                overlap = mol.intor("int1e_ovlp")

                # 4. Solve for Energies and Coefficients
                props = get_electronic_properties(
                    mol, ham=pred_ham_pyscf, overlap=overlap, dm=None
                )

                homo_ev = props["HOMO"] * 27.2114
                lumo_ev = props["LUMO"] * 27.2114
                gap_ev = props["GAP"] * 27.2114

                # 5. Draw HOMO / LUMO Cubes
                cube_html_status = "Not Generated"
                if idx < max_cube_save:
                    homo_coeff = props["HOMO_coefficients"]
                    lumo_coeff = props["LUMO_coefficients"]
                    
                    save_dir = os.path.join(output_base_dir, str(idx))
                    os.makedirs(save_dir, exist_ok=True)
                    
                    homo_path = os.path.join(save_dir, 'homo.cube')
                    lumo_path = os.path.join(save_dir, 'lumo.cube')

                    tools.cubegen.orbital(mol, homo_path, homo_coeff, nx=cube_grid, ny=cube_grid, nz=cube_grid)
                    tools.cubegen.orbital(mol, lumo_path, lumo_coeff, nx=cube_grid, ny=cube_grid, nz=cube_grid)
                    
                    try:
                         cubes_2_htmls(save_dir, 0.03)
                         cube_html_status = f"Generated in {save_dir}"
                    except:
                         pass

                to_sig4 = lambda x: float(f"{x:.4g}")

                ham_info = {
                    'Index': idx,
                    'Charge': current_mol_charge,
                    'HOMO_eV': to_sig4(homo_ev),
                    'LUMO_eV': to_sig4(lumo_ev),
                    'GAP_eV': to_sig4(gap_ev),
                    'Visuals': cube_html_status
                }
                if convert_smiles_flag:
                    ham_info.update({'SMILES': smiles})

                all_ham_info.append(ham_info)

            except Exception as e:
                print(f"Hamiltonian processing failed at idx {idx}: {e}")
                traceback.print_exc()
            finally:
                # 确保切回原目录。
                os.chdir(cwd_)
    
    end_time = time.time()
    os.chdir(cwd_)

    # Save to CSV
    output_csv_path = os.path.join(cwd_, 'ham_summary.csv')
    if all_ham_info:
        df = pd.DataFrame(all_ham_info)
        df.to_csv(output_csv_path, index=False)
    
    # Save to JSON for Agent to read easily
    output_json_path = os.path.join(cwd_, 'ham_summary.json')
    with open(output_json_path, 'w') as f:
        json.dump(all_ham_info, f, indent=2)

    second_per_item = (end_time - start_time) / max(1, len(all_ham_info))
    return f"电子结构分析完成。\nCSV报告: {output_csv_path}\nHOMO/LUMO数据与Cube文件已生成。"