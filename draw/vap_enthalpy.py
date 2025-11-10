#!/usr/bin/env python3
import re
import numpy as np

EV_TO_KJ_PER_MOL = 96.485
R_kJmolK = 8.314462618e-3

# -------- 1) 更稳健的能量读取（兼容多种行格式） --------
# 支持的例子：
# "Energy: -2080.877 eV"
# "U(pot): -2080.877 eV"
# "Potential Energy: -2080.877 eV"
# "E(total): -2080.877 eV"（如果你想读总能，也可切换关键字）
PATTERNS = [
    r'\bEnergy:\s*([+-]?\d+(?:\.\d+)?)\s*eV',              # Energy: xxx eV
    r'\bU\(pot\):\s*([+-]?\d+(?:\.\d+)?)\s*eV',            # U(pot): xxx eV
    r'\bPotential\s+Energy:\s*([+-]?\d+(?:\.\d+)?)\s*eV',  # Potential Energy: xxx eV
]

def read_energies(filename, which='potential'):
    """
    which='potential'：读取势能；如果日志里只有 'Energy:' 就按它解析
    which='total'：若你想读总能，可换 pattern 为对应关键字
    """
    # 如果你要强制“只读势能”，把 PATTERNS 改成仅包含 Potential/U(pot) 的正则
    patterns = [re.compile(p) for p in PATTERNS]
    vals = []
    with open(filename, 'r') as f:
        for line in f:
            for pat in patterns:
                m = pat.search(line)
                if m:
                    vals.append(float(m.group(1)))
                    break
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        raise ValueError(f"[解析失败] {filename} 没找到可用的能量字段。请检查日志行格式。")
    return arr

# -------- 2) 最后 window_ps 的平均（向上取整更安全） --------
def average_last_window(energies, window_ps, timestep_fs=1.0, stride=100):
    frame_interval_ps = timestep_fs * stride * 1e-3  # fs -> ps
    if frame_interval_ps <= 0:
        raise ValueError("frame_interval_ps 非法，请检查 timestep_fs 与 stride。")
    # 用 ceil 避免 round 造成窗口不足一帧的边界问题
    nframes_window = int(np.ceil(window_ps / frame_interval_ps))
    if len(energies) < nframes_window:
        raise ValueError(f"轨迹太短，不足以取最后 {window_ps} ps（需要≥{nframes_window} 帧，实际 {len(energies)} 帧）")
    return energies[-nframes_window:].mean()

# -------- 3) 同时返回 ΔU 与 ΔH --------
def compute_vap(E_gas_avg_eV, E_liq_avg_eV, n_atoms_liq, n_atoms_gas=3, T_K=298.15):
    if n_atoms_liq % n_atoms_gas != 0:
        raise ValueError("n_atoms_liq 不是 n_atoms_gas 的整数倍，请检查。")
    n_mol_liq = n_atoms_liq // n_atoms_gas

    # eV/分子
    E_liq_per_mol_eV = E_liq_avg_eV / n_mol_liq
    E_gas_per_mol_eV = E_gas_avg_eV

    # ΔU（eV/分子）→ kJ/mol
    delta_U_kJmol = (E_gas_per_mol_eV - E_liq_per_mol_eV) * EV_TO_KJ_PER_MOL
    delta_H_kJmol = delta_U_kJmol + R_kJmolK * T_K  # + RT
    return delta_U_kJmol, delta_H_kJmol

if __name__ == "__main__":
    # --- 路径 ---
    gas_file = "1H2O.log"             # 单分子气相
    liq_file = "../h2o_298K/H2O.log"  # 液相盒子

    # --- 读取能量（确保日志里是“势能”；若你打印的是 U(pot): 更稳妥）---
    E_gas_all = read_energies(gas_file, which='potential')
    E_liq_all = read_energies(liq_file, which='potential')

    # --- 采样节奏 ---
    timestep_fs = 1.0
    stride = 100
    window_ps = 100.0

    E_gas_avg = average_last_window(E_gas_all, window_ps, timestep_fs, stride)
    E_liq_avg = average_last_window(E_liq_all, window_ps, timestep_fs, stride)

    print(f"气相势能平均 (最后{window_ps:.0f} ps): {E_gas_avg:.3f} eV/分子")
    print(f"液相势能平均 (最后{window_ps:.0f} ps): {E_liq_avg:.3f} eV/盒")

    n_atoms_liq = 1593
    T_K = 298.15
    dU, dH = compute_vap(E_gas_avg, E_liq_avg, n_atoms_liq, n_atoms_gas=3, T_K=T_K)

    print(f"ΔU_vap (不含 RT) = {dU:.2f} kJ/mol")
    print(f"RT = {R_kJmolK*T_K:.2f} kJ/mol @ {T_K:.2f} K")
    print(f"ΔH_vap (含 RT) = {dH:.2f} kJ/mol")
