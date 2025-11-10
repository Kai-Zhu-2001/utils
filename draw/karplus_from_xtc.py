#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from ase.io import read as ase_read
from ase.neighborlist import NeighborList, natural_cutoffs
from ase import Atoms
import collections
import os
import mdtraj as md

# ==========================================================
# 数学与基础函数
# ==========================================================
def dihedral_angle(p0, p1, p2, p3):
    """返回四点定义的二面角（单位：弧度）"""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return np.arctan2(y, x)

def block_std_err(values, weights=None, nblock=20):
    """块平均法计算标准误差"""
    n = len(values)
    if nblock > n:
        nblock = max(1, n // 2)
    m = n // nblock
    if m == 0:
        return np.nan
    vals = np.asarray(values, float)
    blocks = []
    if weights is None:
        for i in range(nblock):
            b = vals[i*m:(i+1)*m]
            if len(b) > 0:
                blocks.append(b.mean())
    else:
        w = np.asarray(weights, float)
        for i in range(nblock):
            sl = slice(i*m, (i+1)*m)
            vw, ww = vals[sl], w[sl]
            if ww.sum() > 0:
                blocks.append(np.average(vw, weights=ww))
    if len(blocks) <= 1:
        return np.nan
    return np.std(blocks, ddof=1) / np.sqrt(len(blocks))

def karplus(phi, A, B, C, form="Acos2+Bcos+C"):
    """Karplus 方程"""
    if form == "Acos2+Bcos+C":
        return A * np.cos(phi) ** 2 + B * np.cos(phi) + C
    elif form == "Acos2(2phi)+Bcos(phi)+C":
        return A * np.cos(2 * phi) + B * np.cos(phi) + C
    else:
        raise ValueError("Unknown form")

# ==========================================================
# 拓扑结构函数
# ==========================================================
def build_bond_graph(atoms, bond_mult=1.1, bond_skin=0.3):
    """利用 ASE NeighborList 构建键连接图"""
    cutoffs = natural_cutoffs(atoms, mult=bond_mult)
    nl = NeighborList(cutoffs, skin=bond_skin, self_interaction=False, bothways=True)
    nl.update(atoms)
    n = len(atoms)
    adj = [set() for _ in range(n)]
    for i in range(n):
        idxs, _ = nl.get_neighbors(i)
        for j in idxs:
            if i != j:
                adj[i].add(j)
                adj[j].add(i)
    return adj, nl

def connected_components(adj):
    """寻找所有连通分量"""
    seen = [False] * len(adj)
    comps = []
    for s in range(len(adj)):
        if seen[s]:
            continue
        stack = [s]; seen[s] = True; comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(sorted(comp))
    return comps

def _has_oxygen_neighbor(adj, atoms, idx):
    syms = atoms.get_chemical_symbols()
    return any(syms[j] == 'O' for j in adj[idx])

def find_backbone_triplets_N_CA_Cprime(adj, atoms, keep_set=None):
    """返回(N, CA, C′) 主链三元组"""
    syms = atoms.get_chemical_symbols()
    triplets = []
    for ca in range(len(atoms)):
        if syms[ca] != 'C':
            continue
        if keep_set and ca not in keep_set:
            continue
        Ns = [n for n in adj[ca] if syms[n] == 'N' and (not keep_set or n in keep_set)]
        if not Ns:
            continue
        carbons = [c for c in adj[ca] if syms[c] == 'C' and (not keep_set or c in keep_set)]
        Cprime = None
        for c in carbons:
            if _has_oxygen_neighbor(adj, atoms, c):
                Cprime = c
                break
        if Cprime is None:
            continue
        for Nidx in Ns:
            triplets.append((Nidx, ca, Cprime))
    return triplets

def build_ala_3J_quartets(adj, atoms, keep_set=None):
    """返回 ((i,j,k,m), type_tag) 四元组: HNHα/HNCβ/HNC′"""
    syms = atoms.get_chemical_symbols()
    triplets = find_backbone_triplets_N_CA_Cprime(adj, atoms, keep_set)
    out = []
    for Nidx, CA, Cprime in triplets:
        HNs = [h for h in adj[Nidx] if syms[h] == 'H' and (not keep_set or h in keep_set)]
        HAs = [h for h in adj[CA] if syms[h] == 'H' and (not keep_set or h in keep_set)]
        CBs = [c for c in adj[CA] if syms[c] == 'C' and c not in (Nidx, Cprime) and (not keep_set or c in keep_set)]
        for hn in HNs:
            for ha in HAs:
                out.append(((hn, Nidx, CA, ha), "HNHα"))
            for cb in CBs:
                out.append(((hn, Nidx, CA, cb), "HNCβ"))
            out.append(((hn, Nidx, CA, Cprime), "HNC'"))
    uniq, seen = [], set()
    for q, t in out:
        k = min(q, (q[3], q[2], q[1], q[0]))
        if k not in seen:
            seen.add(k)
            uniq.append((q, t))
    return uniq

# ==========================================================
# 轨迹加载
# ==========================================================
def load_frames_any(traj_path, top_path, max_frames, skip):
    """自动识别 ASE 或 MDTraj 格式"""
    ext = os.path.splitext(traj_path)[1].lower()
    if ext in (".trr", ".xtc"):
        if not top_path:
            raise ValueError("Need --top for .trr/.xtc trajectories")
        print(f"[INFO] Loading {traj_path} with {top_path} using MDTraj")
        traj = md.load(traj_path, top=top_path, stride=skip)
        frames = []
        symbols = [a.element.symbol for a in traj.top.atoms]
        for i, frame in enumerate(traj):
            pos = frame.xyz[0] * 10.0
            box = frame.unitcell_vectors[0] * 10.0 if frame.unitcell_vectors is not None else None
            at = Atoms(symbols=symbols, positions=pos, pbc=True)
            if box is not None:
                at.set_cell(box, scale_atoms=False)
            frames.append(at)
            if len(frames) >= max_frames:
                break
        return frames, traj.n_frames * skip
    else:
        traj = ase_read(traj_path, index=f":{max_frames}")
        frames = traj[::skip]
        return list(frames), len(traj)

# ==========================================================
# 主程序
# ==========================================================
def main():
    p = argparse.ArgumentParser(description="Compute ³J couplings for Ala peptides (exclude water)")
    p.add_argument("--traj", required=True)
    p.add_argument("--top")
    p.add_argument("--auto_ala", action="store_true")
    p.add_argument("--bond_mult", type=float, default=1.1)
    p.add_argument("--bond_skin", type=float, default=0.3)
    p.add_argument("--A", type=float, default=6.51)
    p.add_argument("--B", type=float, default=-1.76)
    p.add_argument("--C", type=float, default=1.60)
    p.add_argument("--form", default="Acos2+Bcos+C")
    p.add_argument("--theta", type=float, default=0.0)
    p.add_argument("--skip", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=20000)
    p.add_argument("--out", default=None)
    p.add_argument("--nblock", type=int, default=20)
    p.add_argument("--per_type_params", action="store_true")
    p.add_argument("--per_type_thetas", action="store_true")
    p.add_argument("--A_HNHalpha", type=float)
    p.add_argument("--B_HNHalpha", type=float)
    p.add_argument("--C_HNHalpha", type=float)
    p.add_argument("--theta_HNHalpha", type=float)
    p.add_argument("--A_HNCb", type=float)
    p.add_argument("--B_HNCb", type=float)
    p.add_argument("--C_HNCb", type=float)
    p.add_argument("--theta_HNCb", type=float)
    p.add_argument("--A_HNCp", type=float)
    p.add_argument("--B_HNCp", type=float)
    p.add_argument("--C_HNCp", type=float)
    p.add_argument("--theta_HNCp", type=float)
    args = p.parse_args()

    frames, orig = load_frames_any(args.traj, args.top, args.max_frames, args.skip)
    print(f"[INFO] Loaded {len(frames)} frames (orig {orig})")

    # 构建键图
    adj, _ = build_bond_graph(frames[0], bond_mult=args.bond_mult, bond_skin=args.bond_skin)

    # -------------------- 排除水与离子 --------------------
    syms = frames[0].get_chemical_symbols()
    comps = connected_components(adj)
    solute_comps = []
    for comp in comps:
        elems = {syms[i] for i in comp}
        if any(e in ("N", "C", "S", "P") for e in elems):
            solute_comps.extend(comp)
    keep_set = set(solute_comps)
    print(f"[INFO] Keeping {len(keep_set)} solute atoms (exclude water/ions)")

    # -------------------- 生成 Ala 四元组 --------------------
    quartets, typed = [], []
    if args.auto_ala:
        ala = build_ala_3J_quartets(adj, frames[0], keep_set=keep_set)
        quartets += [q for q, _ in ala]
        typed += ala
        print(f"[INFO] Detected {len(ala)} Ala-type quartets (HN–Hα / HN–Cβ / HN–C′)")

    def get_params(tag):
        if not args.per_type_params:
            return args.A, args.B, args.C
        if tag == "HNHα" and args.A_HNHalpha:
            return args.A_HNHalpha, args.B_HNHalpha, args.C_HNHalpha
        if tag == "HNCβ" and args.A_HNCb:
            return args.A_HNCb, args.B_HNCb, args.C_HNCb
        if tag == "HNC'" and args.A_HNCp:
            return args.A_HNCp, args.B_HNCp, args.C_HNCp
        return args.A, args.B, args.C

    def get_theta(tag):
        if not args.per_type_thetas:
            return np.deg2rad(args.theta)
        if tag == "HNHα" and args.theta_HNHalpha is not None:
            return np.deg2rad(args.theta_HNHalpha)
        if tag == "HNCβ" and args.theta_HNCb is not None:
            return np.deg2rad(args.theta_HNCb)
        if tag == "HNC'" and args.theta_HNCp is not None:
            return np.deg2rad(args.theta_HNCp)
        return np.deg2rad(args.theta)

    # -------------------- 计算 ³J --------------------
    allJ, typed_tags = [], []
    for qid, (q, tag) in enumerate(typed):
        i, j, k, m = q
        phis = np.array([dihedral_angle(f.positions[i], f.positions[j], f.positions[k], f.positions[m]) for f in frames])
        A, B, C = get_params(tag)
        th = get_theta(tag)
        J = karplus(phis + th, A, B, C, args.form)
        allJ.append(J)
        typed_tags.append(tag)
        print(f"[RESULT] {qid:03d} {i}-{j}-{k}-{m} {tag}: ⟨J⟩={J.mean():.2f} Hz")

    # -------------------- 汇总输出 --------------------
    print("\n[TABLE SUMMARY]")
    pertype = collections.defaultdict(list)
    for J, tag in zip(allJ, typed_tags):
        pertype[tag].append(J)
    for tag, arrlist in pertype.items():
        stack = np.stack(arrlist, axis=0)
        Jt = stack.mean(axis=0)
        mean = Jt.mean()
        se = block_std_err(Jt, nblock=args.nblock)
        print(f"{tag:6s}  {mean:7.2f} ± {se:5.2f} Hz  (n_res={stack.shape[0]}, n_frames={stack.shape[1]})")

if __name__ == "__main__":
    main()
