import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scipy.stats import gaussian_kde
from ase.io import read
import glob
import argparse
import os
import csv
from collections import defaultdict
import re
from ase.data import covalent_radii

def calc_bond_lengths_angles_dihedrals(mol, allowed_elements=None):
    pos = mol.get_positions()
    numbers = mol.get_atomic_numbers()
    symbols = mol.get_chemical_symbols()
    
    # 筛选原子
    selected_atoms = [
        i for i, s in enumerate(symbols)
        if (allowed_elements is None or s in allowed_elements)
    ]
    
    # 构建邻接矩阵
    adj_matrix = np.zeros((len(pos), len(pos)), dtype=bool)
    for i in selected_atoms:
        for j in selected_atoms:
            if i >= j: continue
            try:
                d = np.linalg.norm(pos[i] - pos[j])
                threshold = 1.2 * (covalent_radii[numbers[i]] + covalent_radii[numbers[j]])
                if d < threshold:
                    adj_matrix[i,j] = adj_matrix[j,i] = True
            except (KeyError, IndexError):
                # 跳过没有共价半径数据的元素
                continue
    
    bond_dict = defaultdict(list)
    angle_dict = defaultdict(list)
    dihedral_dict = defaultdict(list)
    
    # 记录键长 (只记录真正成键的原子对)
    for i in selected_atoms:
        for j in selected_atoms:
            if i >= j: continue
            if adj_matrix[i,j]:
                d = np.linalg.norm(pos[i] - pos[j])
                bond_dict[(i,j)].append(d)
    
    # 计算角度 (只计算真正的键角)
    for j in selected_atoms:  # 中心原子
        neighbors = [i for i in selected_atoms if adj_matrix[i,j]]
        for k in range(len(neighbors)):
            for l in range(k+1, len(neighbors)):
                i1, i2 = neighbors[k], neighbors[l]
                vec1 = pos[i1] - pos[j]
                vec2 = pos[i2] - pos[j]
                angle = np.degrees(np.arccos(
                    np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                ))
                angle_dict[(i1,j,i2)].append(angle)
    
    # 计算二面角 (只计算连续四个成键原子的二面角)
    for i in selected_atoms:
        for j in selected_atoms:
            if i == j or not adj_matrix[i,j]: continue
            for k in selected_atoms:
                if k == i or k == j or not adj_matrix[j,k]: continue
                for l in selected_atoms:
                    if l == i or l == j or l == k or not adj_matrix[k,l]: continue
                    # 计算二面角 i-j-k-l
                    b1 = pos[j] - pos[i]
                    b2 = pos[k] - pos[j]
                    b3 = pos[l] - pos[k]
                    
                    # 计算法向量
                    n1 = np.cross(b1, b2)
                    n2 = np.cross(b2, b3)
                    
                    # 归一化
                    n1 /= np.linalg.norm(n1)
                    n2 /= np.linalg.norm(n2)
                    b2 /= np.linalg.norm(b2)
                    
                    # 计算二面角
                    m = np.cross(n1, b2)
                    x = np.dot(n1, n2)
                    y = np.dot(m, n2)
                    dihedral = np.degrees(np.arctan2(y, x))
                    
                    dihedral_dict[(i,j,k,l)].append(dihedral)
    
    return bond_dict, angle_dict, dihedral_dict

def get_distribution(data, xmin, xmax, bins=100):
    data = np.asarray(data)
    if len(data) < 2 or np.std(data) < 1e-3:
        x = np.linspace(xmin, xmax, bins)
        p = np.zeros_like(x)
        if len(data) > 0:
            idx = np.argmin(np.abs(x - np.mean(data)))
            p[idx] = 1.0
        p += 1e-10  # 防止除以零
        p /= np.trapezoid(p, x)
        return x, p
    else:
        try:
            kde = gaussian_kde(data)
            x = np.linspace(xmin, xmax, bins)
            p = kde(x) + 1e-10
            p /= np.trapezoid(p, x)
            return x, p
        except:
            x = np.linspace(xmin, xmax, bins)
            return x, np.ones(bins)/bins

def kl_divergence(p, q, epsilon=1e-10):
    # 确保p和q都是numpy数组
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # 归一化概率分布
    p /= p.sum()
    q /= q.sum()

    # 避免0概率，做平滑
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    # 重新归一化（保证总和为1）
    p /= p.sum()
    q /= q.sum()

    return np.sum(p * np.log(p / q))

def natural_sort_key(s):
    # 用于自然排序的辅助函数
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_trajectory_files(folder, pattern, n_files):
    files = glob.glob(f"{folder}/{pattern}")
    files = sorted(files, key=natural_sort_key)[:n_files]
    trajs = [read(f, index=':') for f in files]
    return trajs, [os.path.basename(f) for f in files]

def group_and_kl(dft_data_all, lumi_data_all, kind, xmin, xmax, out_dir, out_prefix):
    kl_dict = {}
    x_vals = {}

    for key in set(dft_data_all.keys()).intersection(lumi_data_all.keys()):
        dft_raw = dft_data_all[key]
        lumi_raw = lumi_data_all[key]

        # 转换标量为列表
        if not isinstance(dft_raw[0], (list, np.ndarray)):
            dft_raw = [[v] for v in dft_raw]
        if not isinstance(lumi_raw[0], (list, np.ndarray)):
            lumi_raw = [[v] for v in lumi_raw]

        try:
            dft_vals = np.concatenate(dft_raw)
            lumi_vals = np.concatenate(lumi_raw)
        except Exception as e:
            print(f"Skipping key {key} due to error: {e}")
            continue

        x, p = get_distribution(dft_vals, xmin, xmax)
        _, q = get_distribution(lumi_vals, xmin, xmax)
        if p is not None and q is not None:
            kl = kl_divergence(p, q)
            kl_dict[key] = kl
            x_vals[key] = (x, p, q)

    if not kl_dict:
        return None

    for key in kl_dict:
        x, p, q = x_vals[key]
        kl_value = kl_dict[key]

        plt.figure(figsize=(6, 4))
        plt.plot(x, p, label='DFT', color='#1f77b4', linewidth=2)
        plt.plot(x, q, label='LaMForce', color='#ff7f0e', linewidth=2, linestyle='--')
        plt.xlabel(f"{kind} Value")
        plt.ylabel("Probability")
        plt.title(f"{kind} KL Divergence: {key} (KL = {kl_value:.4f})")
        plt.legend()
        plt.tight_layout()

        filename = f"{out_prefix}_{kind.lower().replace(' ', '_')}_{key}.png"
        plt.savefig(os.path.join(out_dir, filename), dpi=300)
        plt.close()

    # best_key = min(kl_dict, key=kl_dict.get)
    # print(f"Best {kind} match: {best_key} (KL = {kl_dict[best_key]:.4f})")
    #
    # # 绘图
    # x, p, q = x_vals[best_key]
    # plt.figure(figsize=(6, 4))
    # plt.plot(x, p, label='DFT', color='#1f77b4', linewidth=2)
    # plt.plot(x, q, label='LaMForce', color='#ff7f0e', linewidth=2, linestyle='--')
    # plt.xlabel(f"{kind} Value")
    # plt.ylabel("Probability")
    # plt.title(f"{kind} KL Divergence: {best_key} (KL = {kl_dict[best_key]:.4f})")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, f"{out_prefix}_{kind.lower().replace(' ', '_')}_best.png"), dpi=300)
    # plt.close()

    return kl_dict

def compute_avg_kl_per_molecule(dft_frames, lumi_frames, bond_range, angle_range, dihedral_range):
    bond_kls = []
    angle_kls = []
    dihedral_kls = []

    n_frames = min(len(dft_frames), len(lumi_frames))
    for f in range(n_frames):
        allowed_elements = ['C', 'N', 'O']  # 举例，只分析 C/N/O 构成的键、角和二面角

        dft_bonds, dft_angles, dft_dihedrals = calc_bond_lengths_angles_dihedrals(dft_frames[f], allowed_elements)
        lumi_bonds, lumi_angles, lumi_dihedrals = calc_bond_lengths_angles_dihedrals(lumi_frames[f], allowed_elements)

        common_bonds = set(dft_bonds.keys()).intersection(lumi_bonds.keys())
        for key in common_bonds:
            dft_vals = dft_bonds[key]
            lumi_vals = lumi_bonds[key]
            x, p = get_distribution(dft_vals, *bond_range)
            _, q = get_distribution(lumi_vals, *bond_range)
            if p is not None and q is not None:
                bond_kls.append(kl_divergence(p, q))

        common_angles = set(dft_angles.keys()).intersection(lumi_angles.keys())
        for key in common_angles:
            dft_vals = dft_angles[key]
            lumi_vals = lumi_angles[key]
            x, p = get_distribution(dft_vals, *angle_range)
            _, q = get_distribution(lumi_vals, *angle_range)
            if p is not None and q is not None:
                angle_kls.append(kl_divergence(p, q))

        common_dihedrals = set(dft_dihedrals.keys()).intersection(lumi_dihedrals.keys())
        for key in common_dihedrals:
            dft_vals = dft_dihedrals[key]
            lumi_vals = lumi_dihedrals[key]
            x, p = get_distribution(dft_vals, *dihedral_range)
            _, q = get_distribution(lumi_vals, *dihedral_range)
            if p is not None and q is not None:
                dihedral_kls.append(kl_divergence(p, q))

    avg_bond_kl = np.mean(bond_kls) if bond_kls else np.nan
    avg_angle_kl = np.mean(angle_kls) if angle_kls else np.nan
    avg_dihedral_kl = np.mean(dihedral_kls) if dihedral_kls else np.nan
    return avg_bond_kl, avg_angle_kl, avg_dihedral_kl

def plot_kl_distribution(kl_values, kind, out_dir):
    if not kl_values:
        print(f"No {kind} KL divergences to plot.")
        return

    kl_values = np.array(kl_values)
    plt.figure(figsize=(6, 4))

    n_bins = min(10, max(2, len(kl_values)//2))  
    bin_range = (0, kl_values.max() * 1.1 if kl_values.max() > 0 else 1)

    hist, bin_edges = np.histogram(kl_values, bins=n_bins, range=bin_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], alpha=0.7,
            color='tab:blue', edgecolor='black')

    plt.xlabel("KL Divergence")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {kind} KL Divergences\n(Total: {len(kl_values)})")

    plt.xlim(left=0, right=bin_range[1])
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{kind.lower().replace(' ', '_')}_kl_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved KL distribution plot: {out_path}")

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    dft_trajs, dft_names = load_trajectory_files(args.dft_folder, args.dft_pattern, args.n_files)
    lumi_trajs, lumi_names = load_trajectory_files(args.lumi_folder, args.lumi_pattern, args.n_files)
    n_pairs = min(len(dft_trajs), len(lumi_trajs))

    # 准备结果数据结构
    summary_results = []
    detailed_results = []
    
    # 用于全局统计
    all_bond_kls = []
    all_angle_kls = []
    all_dihedral_kls = []

    for i in range(n_pairs):
        dft_frames = dft_trajs[i]
        lumi_frames = lumi_trajs[i]
        name = os.path.splitext(dft_names[i])[0]

        print(f"\nProcessing molecule {i+1}/{n_pairs}: {name}")

        # 初始化数据结构
        dft_bond_all = defaultdict(list)
        dft_angle_all = defaultdict(list)
        dft_dihedral_all = defaultdict(list)
        lumi_bond_all = defaultdict(list)
        lumi_angle_all = defaultdict(list)
        lumi_dihedral_all = defaultdict(list)

        # 收集所有帧的数据
        n_frames = min(len(dft_frames), len(lumi_frames))
        for f in range(n_frames):
            dft_bonds, dft_angles, dft_dihedrals = calc_bond_lengths_angles_dihedrals(dft_frames[f])
            lumi_bonds, lumi_angles, lumi_dihedrals = calc_bond_lengths_angles_dihedrals(lumi_frames[f])
            
            for key, val in dft_bonds.items():
                dft_bond_all[key].extend(val)
            for key, val in dft_angles.items():
                dft_angle_all[key].extend(val)
            for key, val in dft_dihedrals.items():
                dft_dihedral_all[key].extend(val)
            for key, val in lumi_bonds.items():
                lumi_bond_all[key].extend(val)
            for key, val in lumi_angles.items():
                lumi_angle_all[key].extend(val)
            for key, val in lumi_dihedrals.items():
                lumi_dihedral_all[key].extend(val)

        total_data = {
            'dft_bonds': dft_bond_all,
            'dft_angles': dft_angle_all,
            'dft_dihedrals': dft_dihedral_all,
            'lumi_bonds': lumi_bond_all,
            'lumi_angles': lumi_angle_all,
            'lumi_dihedrals': lumi_dihedral_all,
        }

        np.savez('/home/gouqiaolin/zhukai/ASE_MD/Geometrical_structure/analysis/total_geometric.npz', **total_data)

        # 计算KL散度
        bond_kls = []
        angle_kls = []
        dihedral_kls = []
        
        # 键长KL计算
        common_bonds = set(dft_bond_all.keys()).intersection(lumi_bond_all.keys())
        for key in common_bonds:
            dft_vals = dft_bond_all[key]
            lumi_vals = lumi_bond_all[key]
            x, p = get_distribution(dft_vals, 0.8, 2.0)
            _, q = get_distribution(lumi_vals, 0.8, 2.0)
            if p is not None and q is not None:
                kl = kl_divergence(p, q)
                bond_kls.append(kl)
                all_bond_kls.append(kl)
                # 记录详细数据
                detailed_results.append({
                    'molecule': name,
                    'type': 'bond',
                    'atom_indices': str(key),
                    'kl_divergence': kl,
                    'n_samples_dft': len(dft_vals),
                    'n_samples_lumi': len(lumi_vals),
                    'mean_dft': np.mean(dft_vals),
                    'mean_lumi': np.mean(lumi_vals)
                })

        # 键角KL计算
        common_angles = set(dft_angle_all.keys()).intersection(lumi_angle_all.keys())
        for key in common_angles:
            dft_vals = dft_angle_all[key]
            lumi_vals = lumi_angle_all[key]
            x, p = get_distribution(dft_vals, 0, 180)
            _, q = get_distribution(lumi_vals, 0, 180)
            if p is not None and q is not None:
                kl = kl_divergence(p, q)
                angle_kls.append(kl)
                all_angle_kls.append(kl)
                detailed_results.append({
                    'molecule': name,
                    'type': 'angle',
                    'atom_indices': str(key),
                    'kl_divergence': kl,
                    'n_samples_dft': len(dft_vals),
                    'n_samples_lumi': len(lumi_vals),
                    'mean_dft': np.mean(dft_vals),
                    'mean_lumi': np.mean(lumi_vals)
                })

        # 二面角KL计算
        common_dihedrals = set(dft_dihedral_all.keys()).intersection(lumi_dihedral_all.keys())
        for key in common_dihedrals:
            dft_vals = dft_dihedral_all[key]
            lumi_vals = lumi_dihedral_all[key]
            x, p = get_distribution(dft_vals, -180, 180)
            _, q = get_distribution(lumi_vals, -180, 180)
            if p is not None and q is not None:
                kl = kl_divergence(p, q)
                dihedral_kls.append(kl)
                all_dihedral_kls.append(kl)
                detailed_results.append({
                    'molecule': name,
                    'type': 'dihedral',
                    'atom_indices': str(key),
                    'kl_divergence': kl,
                    'n_samples_dft': len(dft_vals),
                    'n_samples_lumi': len(lumi_vals),
                    'mean_dft': np.mean(dft_vals),
                    'mean_lumi': np.mean(lumi_vals)
                })

        # 保存汇总统计
        summary_results.append({
            'molecule': name,
            'avg_bond_kl': np.mean(bond_kls) if bond_kls else np.nan,
            'var_bond_kl': np.var(bond_kls) if bond_kls else np.nan,
            'min_bond_kl': np.min(bond_kls) if bond_kls else np.nan,
            'max_bond_kl': np.max(bond_kls) if bond_kls else np.nan,
            'avg_angle_kl': np.mean(angle_kls) if angle_kls else np.nan,
            'var_angle_kl': np.var(angle_kls) if angle_kls else np.nan,
            'min_angle_kl': np.min(angle_kls) if angle_kls else np.nan,
            'max_angle_kl': np.max(angle_kls) if angle_kls else np.nan,
            'avg_dihedral_kl': np.mean(dihedral_kls) if dihedral_kls else np.nan,
            'var_dihedral_kl': np.var(dihedral_kls) if dihedral_kls else np.nan,
            'min_dihedral_kl': np.min(dihedral_kls) if dihedral_kls else np.nan,
            'max_dihedral_kl': np.max(dihedral_kls) if dihedral_kls else np.nan,
            'n_bonds': len(bond_kls),
            'n_angles': len(angle_kls),
            'n_dihedrals': len(dihedral_kls)
        })

        # 为每个分子创建输出目录
        mol_out_dir = os.path.join(args.out_dir, name)
        os.makedirs(mol_out_dir, exist_ok=True)

        # 分析并绘制最佳匹配的分布
        print(f"\n=== Per-Bond Analysis for {name} ===")
        group_and_kl(
            dft_bond_all, lumi_bond_all,
            "Bond Length (Å)", 0.8, 2.0,
            mol_out_dir,
            "best"
        )

        print(f"\n=== Per-Angle Analysis for {name} ===")
        group_and_kl(
            dft_angle_all, lumi_angle_all,
            "Bond Angle (°)", 0, 180,
            mol_out_dir,
            "best"
        )

        print(f"\n=== Per-Dihedral Analysis for {name} ===")
        group_and_kl(
            dft_dihedral_all, lumi_dihedral_all,
            "Dihedral Angle (°)", -180, 180,
            mol_out_dir,
            "best"
        )

        # 绘制KL分布
        plot_kl_distribution(bond_kls, "Bond Length", mol_out_dir)
        plot_kl_distribution(angle_kls, "Bond Angle", mol_out_dir)
        plot_kl_distribution(dihedral_kls, "Dihedral Angle", mol_out_dir)

    # 保存汇总统计结果
    summary_csv = os.path.join(args.out_dir, "molecule_kl_summary.csv")
    with open(summary_csv, "w", newline="") as csvfile:
        fieldnames = [
            "molecule", 
            "avg_bond_kl", "var_bond_kl", "min_bond_kl", "max_bond_kl", "n_bonds",
            "avg_angle_kl", "var_angle_kl", "min_angle_kl", "max_angle_kl", "n_angles",
            "avg_dihedral_kl", "var_dihedral_kl", "min_dihedral_kl", "max_dihedral_kl", "n_dihedrals"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_results:
            writer.writerow(row)
    print(f"\nSaved summary statistics to {summary_csv}")

    # 保存详细KL结果
    detailed_csv = os.path.join(args.out_dir, "molecule_kl_detailed.csv")
    with open(detailed_csv, "w", newline="") as csvfile:
        fieldnames = [
            "molecule", "type", "atom_indices", "kl_divergence",
            "n_samples_dft", "n_samples_lumi", "mean_dft", "mean_lumi"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in detailed_results:
            writer.writerow(row)
    print(f"Saved detailed KL results to {detailed_csv}")

    # 打印汇总统计
    print("\nPer-Molecule KL Statistics Summary:")
    print(f"{'Molecule':<15} {'Bonds':<25} {'Angles':<25} {'Dihedrals':<25}")
    for row in summary_results:
        print(f"{row['molecule']:<15} "
              f"μ={row['avg_bond_kl']:.3f}±{np.sqrt(row['var_bond_kl']):.3f}({row['n_bonds']}) "
              f"μ={row['avg_angle_kl']:.3f}±{np.sqrt(row['var_angle_kl']):.3f}({row['n_angles']}) "
              f"μ={row['avg_dihedral_kl']:.3f}±{np.sqrt(row['var_dihedral_kl']):.3f}({row['n_dihedrals']})")

    # 绘制全局KL分布
    plot_kl_distribution(all_bond_kls, "All Molecules Bond Length", args.out_dir)
    plot_kl_distribution(all_angle_kls, "All Molecules Bond Angle", args.out_dir)
    plot_kl_distribution(all_dihedral_kls, "All Molecules Dihedral Angle", args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KL divergence per bond/angle/dihedral across molecules.")
    parser.add_argument("--dft_folder", type=str, required=True)
    parser.add_argument("--lumi_folder", type=str, required=True)
    parser.add_argument("--dft_pattern", type=str, default="*.traj")
    parser.add_argument("--lumi_pattern", type=str, default="*.traj")
    parser.add_argument("--n_files", type=int, default=15)
    parser.add_argument("--out_dir", type=str, default="kl_total_output")
    args = parser.parse_args()
    main(args)
