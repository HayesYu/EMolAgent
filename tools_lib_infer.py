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

from typing import List, Tuple, Dict, Union
import random
from scipy.spatial import ConvexHull
from ase import Atoms
from ase.io import write
from rdkit import Chem
try:
    from emoles.build.CombineMols3D import (
        get_bond_length,
        calculate_intermolecular_repulsion,
        DEFAULT_CLASH_FACTOR,
    )
    from emoles.build.patch_picker import (
        get_patch_atoms_and_indices,
        atom_2_mol,
    )
except ImportError:
    print("Warning: emoles.build not found. Cluster building features will fail if called.")

CLASH_PENALTY_FOR_SCORING = 1e10

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

def is_xyz_path(text: str) -> bool:
    return isinstance(text, str) and text.lower().endswith(".xyz")

def make_cache_key(identifier: Union[str, Atoms, Chem.Mol]) -> str:
    if isinstance(identifier, str):
        return f"XYZ:{identifier}" if is_xyz_path(identifier) else f"SMILES:{identifier}"
    if isinstance(identifier, Atoms):
        return f"ASE@{id(identifier)}"
    if isinstance(identifier, Chem.Mol):
        return f"RDKIT@{id(identifier)}"
    raise TypeError(f"Unsupported identifier type for cache key: {type(identifier)}")

def fibonacci_sphere(samples: int, radius: float, center: np.ndarray) -> np.ndarray:
    points = np.empty((samples, 3))
    phi = np.pi * (np.sqrt(5.) - 1.)
    denom = float(max(1, samples - 1))
    for i in range(samples):
        y = 1.0 - (i / denom) * 2.0
        r_xy = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        points[i] = [np.cos(theta) * r_xy, y, np.sin(theta) * r_xy]
        if not np.all(np.isfinite(points[i])):
            points[i] = np.array([0.0, 1.0 - (i / denom) * 2.0, 0.0])
    return points * radius + center

def calculate_ligand_interactions(ion_atoms, all_ligands, current_lig_idx, clash_penalty=CLASH_PENALTY_FOR_SCORING):
    eval_lig = all_ligands[current_lig_idx]
    total_repulsion_score = 0.0
    any_clash_detected = False

    # Interaction with ion
    rep_val_ion, clash_ion = calculate_intermolecular_repulsion(ion_atoms, eval_lig)
    if clash_ion:
        any_clash_detected = True
        total_repulsion_score += clash_penalty
    else:
        total_repulsion_score += rep_val_ion

    # Interaction with other ligands
    for i, other_lig in enumerate(all_ligands):
        if i == current_lig_idx:
            continue
        rep_val_pair, clash_pair = calculate_intermolecular_repulsion(eval_lig, other_lig)
        if clash_pair:
            any_clash_detected = True
            total_repulsion_score += clash_penalty
        else:
            total_repulsion_score += rep_val_pair

    return total_repulsion_score, any_clash_detected

def check_system_clashes(ion_atoms: Atoms, all_ligands: List[Atoms]) -> bool:
    if any(calculate_intermolecular_repulsion(ion_atoms, lig)[1] for lig in all_ligands):
        return True
    for i in range(len(all_ligands)):
        for j in range(i + 1, len(all_ligands)):
            if calculate_intermolecular_repulsion(all_ligands[i], all_ligands[j])[1]:
                return True
    return False

def method_convex_hull_volume(atoms: Atoms) -> float:
    pts = atoms.get_positions()
    if len(pts) < 4:
        return 1.0
    hull = ConvexHull(pts)
    return float(hull.volume)

def get_ase_and_patch(identifier, relative_score_threshold, max_patch_atoms, verbose):
    if isinstance(identifier, str) and is_xyz_path(identifier):
        ase_obj = read(identifier)
        ase_obj = ase_obj if isinstance(ase_obj, Atoms) else ase_obj[0]
        ase_res, patch_idx = get_patch_atoms_and_indices(
            ase_obj,
            relative_score_threshold=relative_score_threshold,
            max_patch_atoms=max_patch_atoms,
            verbose=verbose,
        )
        return ase_res, patch_idx

    ase_res, patch_idx = get_patch_atoms_and_indices(
        identifier,
        relative_score_threshold=relative_score_threshold,
        max_patch_atoms=max_patch_atoms,
        verbose=verbose,
    )
    return ase_res, patch_idx

def prepare_ion(ion_identifier, relative_score_threshold, verbose):
    ion_ase, _ = get_ase_and_patch(
        ion_identifier,
        relative_score_threshold=relative_score_threshold,
        max_patch_atoms=1,
        verbose=verbose,
    )
    ion_center = ion_ase.positions[0] if len(ion_ase) > 0 else np.array([0.0, 0.0, 0.0])
    ion_symbol = ion_ase.get_chemical_symbols()[0] if len(ion_ase) > 0 else "X"
    return ion_ase, ion_center, ion_symbol

def build_ligand_templates(ligand_molecule_info, ion_symbol, relative_score_threshold, max_patch_atoms, verbose):
    cache = {}
    templates = []
    
    for key_obj, count in ligand_molecule_info:
        cache_key = make_cache_key(key_obj)
        if cache_key not in cache:
            mol_ase, patch_indices = get_ase_and_patch(
                key_obj,
                relative_score_threshold=relative_score_threshold,
                max_patch_atoms=max_patch_atoms,
                verbose=verbose,
            )
            if len(patch_indices) == 0:
                raise ValueError(f"No patch atoms found for ligand: {key_obj}")

            patch_atom_coords = mol_ase.get_positions()[patch_indices]
            patch_centroid_local = np.mean(patch_atom_coords, axis=0)

            ideal_distances = [
                get_bond_length(ion_symbol, mol_ase.get_chemical_symbols()[idx], skin=0)
                for idx in patch_indices
            ]
            avg_ideal_dist = float(np.mean(ideal_distances)) if ideal_distances else get_bond_length(
                ion_symbol, mol_ase.get_chemical_symbols()[0], skin=0
            )
            cache[cache_key] = (mol_ase, patch_indices, patch_centroid_local, avg_ideal_dist)

        mol_ase, patch_indices, centroid, avg_dist = cache[cache_key]
        for _ in range(count):
            templates.append((mol_ase.copy(), list(patch_indices), centroid.copy(), float(avg_dist), key_obj))
    return templates

def place_ligands_initial(ion_center, ligand_templates, sphere_dist_factor, orientation_mode):
    total = len(ligand_templates)
    if total == 0:
        return [], np.zeros((0, 3))

    volumes = [max(method_convex_hull_volume(tpl[0]), 1e-12) for tpl in ligand_templates]
    vol_ref = float(np.max(volumes)) if len(volumes) > 0 else 1.0

    volume_scales = []
    for v in volumes:
        raw = (vol_ref / max(v, 1e-12)) ** (1.0 / 8.0)
        volume_scales.append(float(raw))

    base_target_distances = [tpl[3] * sphere_dist_factor for tpl in ligand_templates]
    target_distances = [base_target_distances[i] * volume_scales[i] for i in range(total)]

    directions = fibonacci_sphere(total, radius=1.0, center=np.array([0.0, 0.0, 0.0]))

    ligands = []
    target_centroids = np.empty((total, 3))

    for i, (ase_mol, patch_indices, centroid_local, _, _) in enumerate(ligand_templates):
        lig = ase_mol.copy()
        target = ion_center + directions[i] * target_distances[i]
        target_centroids[i] = target

        lig.translate(target - centroid_local)

        if orientation_mode == "random" or len(lig) == 1:
            axis = np.random.rand(3) - 0.5
            norm = np.linalg.norm(axis)
            axis = axis / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])
            lig.rotate(np.random.uniform(0.0, 360.0), axis, center=target)
        elif orientation_mode == "aligned_to_ion" and patch_indices:
            primary_patch_pos = lig.positions[patch_indices[0]]
            v_centroid_to_patch = primary_patch_pos - target
            v_ion_to_centroid = target - ion_center
            if np.linalg.norm(v_centroid_to_patch) > 1e-8 and np.linalg.norm(v_ion_to_centroid) > 1e-8:
                lig.rotate(v_centroid_to_patch, -v_ion_to_centroid, center=target)
        ligands.append(lig)
    return ligands, target_centroids

def optimize_ligand_orientations(ion_atoms, ligands, rotation_centers, rotation_opt_iterations, rotation_samples_per_ligand, verbose):
    total = len(ligands)
    for rot_iter in range(rotation_opt_iterations):
        order = list(range(total))
        random.shuffle(order)
        improvements = 0

        for idx in order:
            current = ligands[idx]
            center = rotation_centers[idx]
            base_score, _ = calculate_ligand_interactions(ion_atoms, ligands, idx)

            best_local = current.copy()
            best_score = base_score

            for _ in range(rotation_samples_per_ligand):
                trial = current.copy()
                axis = np.random.rand(3) - 0.5
                norm = np.linalg.norm(axis)
                axis = axis / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])
                trial.rotate(np.random.uniform(0.0, 360.0), axis, center=center)

                tmp = list(ligands)
                tmp[idx] = trial
                trial_score, _ = calculate_ligand_interactions(ion_atoms, tmp, idx)

                if trial_score < best_score:
                    best_score = trial_score
                    best_local = trial.copy()

            if best_score < base_score - 1e-6:
                improvements += 1
            ligands[idx] = best_local

        if improvements == 0 and rot_iter > min(5, rotation_opt_iterations // 3):
            break
    return ligands

def evaluate_configuration(ion_atoms, ligands):
    has_clashes = check_system_clashes(ion_atoms, ligands)
    total_score = 0.0
    for k in range(len(ligands)):
        sc, _ = calculate_ligand_interactions(ion_atoms, ligands, k)
        total_score += sc
    return has_clashes, total_score

def build_cluster(
        ion_identifier,
        ligand_molecule_info,
        relative_score_threshold=0.7,
        max_patch_atoms=3,
        initial_sphere_skin_factor=1.0,
        sphere_skin_increment_factor=0.05,
        max_sphere_expansions=10,
        target_no_clashes=True,
        rotation_opt_iterations=30,
        rotation_samples_per_ligand=50,
        initial_ligand_orientation="random",
        verbose=True,
):
    ase_ion, ion_center, ion_symbol = prepare_ion(ion_identifier, relative_score_threshold, verbose)
    ligand_templates = build_ligand_templates(ligand_molecule_info, ion_symbol, relative_score_threshold, max_patch_atoms, verbose)
    
    if len(ligand_templates) == 0:
        return ase_ion

    best_config = []
    best_score = float("inf")
    best_has_clash = True

    sphere_factor = initial_sphere_skin_factor
    for attempt in range(max_sphere_expansions):
        placed, centers = place_ligands_initial(ion_center, ligand_templates, sphere_factor, initial_ligand_orientation)
        placed = optimize_ligand_orientations(ase_ion, placed, centers, rotation_opt_iterations, rotation_samples_per_ligand, verbose)
        has_clashes, total_score = evaluate_configuration(ase_ion, placed)

        improved = (
                best_config == []
                or (not has_clashes and best_has_clash)
                or (not has_clashes and not best_has_clash and total_score < best_score)
                or (has_clashes and best_has_clash and not target_no_clashes and total_score < best_score)
        )
        if improved:
            best_config = [lig.copy() for lig in placed]
            best_score = total_score
            best_has_clash = has_clashes

        if target_no_clashes and not best_has_clash:
            break
        sphere_factor += sphere_skin_increment_factor

    final_cluster = ase_ion.copy()
    for lig in best_config:
        final_cluster.extend(lig)

    return final_cluster

def build_cluster_db_from_smiles(
    ion_smiles: str, 
    ligands_list: List[Dict], # format: [{"smiles": "CCO", "count": 2}, ...]
    output_dir: str,
    db_filename: str = "generated_cluster.db"
) -> str:
    """
    Agent 工具入口：根据 SMILES 构建团簇并保存为 ASE DB。
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 转换 ligands_list 为 build_cluster 需要的格式 [(smiles, count), ...]
        ligand_molecule_info = []
        for item in ligands_list:
            smiles = item.get("smiles")
            count = int(item.get("count", 1))
            if smiles:
                ligand_molecule_info.append((smiles, count))
        
        # 2. 调用核心构建逻辑
        cluster_atoms = build_cluster(
            ion_identifier=ion_smiles,
            ligand_molecule_info=ligand_molecule_info,
            verbose=False # Agent 调用时保持安静
        )
        
        # 3. 保存为 ase.db
        db_path = os.path.join(output_dir, db_filename)
        # 如果文件已存在，先删除，保证是从头写入
        if os.path.exists(db_path):
            os.remove(db_path)
            
        with connect(db_path) as db:
            db.write(cluster_atoms, data={"source": "agent_build", "ion": ion_smiles})
            
        return f"团簇构建成功。\n包含原子数: {len(cluster_atoms)}\n已保存至数据库: {db_path}"

    except Exception as e:
        traceback.print_exc()
        return f"构建团簇失败: {str(e)}"

def compress_directory(dir_path: str, output_path_base: str) -> str:
    """
    压缩文件夹
    :param dir_path: 要压缩的文件夹路径
    :param output_path_base: 输出文件路径（不含后缀，例如 /tmp/res），函数会自动添加 .zip
    :return: 压缩文件的完整路径
    """
    return shutil.make_archive(output_path_base, 'zip', dir_path)