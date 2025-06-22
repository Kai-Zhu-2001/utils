#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.geometry import get_distances
from tqdm import tqdm
import os

def get_cache_filename(traj_file, prefix="oo_rdf", ext=".npz"):
    base = os.path.splitext(os.path.basename(traj_file))[0]
    return f"{prefix}_{base}{ext}"

def calculate_oo_rdf(traj, r_range=(0.1, 8.0), n_bins=100, pbc=True):
    bin_edges = np.linspace(r_range[0], r_range[1], n_bins + 1)
    r = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    dr = bin_edges[1] - bin_edges[0]
    
    O_indices = [i for i, atom in enumerate(traj[0]) if atom.symbol == 'O']
    if len(O_indices) < 2:
        raise ValueError("至少需要2个氧原子才能计算RDF")

    distances_all = []

    for frame in tqdm(traj, desc="计算O-O距离"):
        pos = frame.positions[O_indices]
        _, dists = get_distances(pos, pos, cell=frame.cell, pbc=pbc)
        for i in range(len(O_indices)):
            for j in range(i + 1, len(O_indices)):
                d = dists[i, j]
                if r_range[0] <= d <= r_range[1]:
                    distances_all.append(d)

    distances_all = np.array(distances_all)
    print(f"计算得到 {len(distances_all)} 个 O-O 距离样本")

    hist_OO, _ = np.histogram(distances_all, bins=bin_edges)

    volume = np.prod(traj[0].cell.lengths())  # Å^3
    rho = len(O_indices) / volume  # 原子密度
    shell_volumes = 4.0 * np.pi * r**2 * dr
    norm = rho * shell_volumes * len(O_indices) * len(traj) / 2
    g_r = hist_OO / (norm + 1e-10)

    return r, g_r

def get_or_calculate_rdf(traj_file, r_max, n_bins, pbc):
    cache_file = get_cache_filename(traj_file)
    if os.path.exists(cache_file):
        print(f"载入缓存 RDF: {cache_file}")
        data = np.load(cache_file)
        return data['r'], data['g_r']
    else:
        print(f"读取轨迹文件: {traj_file}")
        traj = read(traj_file, index=':')
        r, g_r = calculate_oo_rdf(traj, r_range=(0.1, r_max), n_bins=n_bins, pbc=pbc)
        np.savez(cache_file, r=r, g_r=g_r)
        print(f"RDF 保存到缓存文件: {cache_file}")
        return r, g_r

def plot_oo_rdf(r1, g1, r2=None, g2=None, r3=None, g3=None, output_file="oo_rdf.png", exp_file=None):
    plt.figure(figsize=(6, 5))
    plt.plot(r1, g1, label="LumiPI (H2O dataset)", color='#FF69B4', linewidth=2.5, linestyle='-', alpha=0.7)  # 粉色

    if r2 is not None and g2 is not None:
        plt.plot(r2, g2, label="LumiPI (SPICE)", color='#FFD700', linewidth=2.5, linestyle='-', alpha=0.7)  # 黄色

    if r3 is not None and g3 is not None:
        plt.plot(r3, g3, label="MACE-OFF", color='#1E90FF', linewidth=2.5, linestyle='-', alpha=0.7)  # 蓝色

    if exp_file:
        exp_data = np.genfromtxt(exp_file, comments="#")
        exp_r = exp_data[:, 3]
        exp_g = exp_data[:, 4]
        plt.plot(exp_r, exp_g, label="Exp", color='#006400', linestyle='-', linewidth=2.5, alpha=1)  # 深绿色

    plt.xlim(2, 8)
    plt.ylim(0, 3.5)
    plt.xlabel('Distance (Å)', fontsize=16)
    plt.ylabel('g(r)', fontsize=16)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"O-O RDF plot saved to {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Calculate and compare O-O RDF from ASE trajectories')
    parser.add_argument('traj_file', help='Trajectory 1: LaMForce (Cheng Bingqing)')
    parser.add_argument('--traj2', help='Trajectory 2: LaMForce (MACE fine-tuned)', default=None)
    parser.add_argument('--traj3', help='Trajectory 3: MACE-off', default=None)
    parser.add_argument('--r_max', type=float, default=8.0, help='Maximum distance for RDF (Å)')
    parser.add_argument('--bins', type=int, default=200, help='Number of bins')
    parser.add_argument('--output', default='oo_rdf.png', help='Output image file')
    parser.add_argument('--no-pbc', action='store_true', help='Disable periodic boundary conditions')
    parser.add_argument('--exp', help='Experimental RDF data file to compare (optional)', default=None)

    args = parser.parse_args()

    r1, g1 = get_or_calculate_rdf(args.traj_file, args.r_max, args.bins, pbc=not args.no_pbc)

    r2 = g2 = r3 = g3 = None
    if args.traj2:
        r2, g2 = get_or_calculate_rdf(args.traj2, args.r_max, args.bins, pbc=not args.no_pbc)
    if args.traj3:
        r3, g3 = get_or_calculate_rdf(args.traj3, args.r_max, args.bins, pbc=not args.no_pbc)

    plot_oo_rdf(r1, g1, r2, g2, r3, g3, output_file=args.output, exp_file=args.exp)


if __name__ == "__main__":
    main()
