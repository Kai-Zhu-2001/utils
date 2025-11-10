#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from ase.io import read
import os


def unwrap_positions_cartesian(traj, remove_com=True):
    """处理NPT轨迹的解包，支持可变晶胞"""
    n_frames = len(traj)
    n_atoms = len(traj[0])

    unwrapped_cart = np.empty((n_frames, n_atoms, 3))
    unwrapped_cart[0] = traj[0].positions.copy()

    for t in range(1, n_frames):
        cell = traj[t].cell.array  # 使用当前帧晶胞
        invT = np.linalg.inv(cell).T
        
        # 将上一帧解包坐标转换到当前晶胞的分数坐标
        frac_prev = (invT @ unwrapped_cart[t-1].T).T
        # 当前帧坐标的分数坐标
        frac_current = (invT @ traj[t].positions.T).T
        
        dfrac = frac_current - frac_prev
        dfrac -= np.round(dfrac)  # 最小镜像约定
        
        unwrapped_frac = frac_prev + dfrac
        unwrapped_cart[t] = unwrapped_frac @ cell

    if remove_com:
        # 去除整体漂移
        com = unwrapped_cart.mean(axis=1, keepdims=True)
        unwrapped_cart -= com
    
    return unwrapped_cart


def msd_from_unwrapped_robust(unwrapped_cart, max_lag=None):
    """更稳健的MSD计算，使用时间平均"""
    n_frames = len(unwrapped_cart)
    
    if max_lag is None:
        max_lag = n_frames // 4  # 默认使用1/4轨迹长度进行平均
    
    msd = np.zeros(max_lag)
    counts = np.zeros(max_lag)
    
    for tau in range(1, max_lag):
        if tau >= n_frames:
            break
            
        # 计算所有可能时间间隔的位移
        disp = unwrapped_cart[tau:] - unwrapped_cart[:-tau]
        squared_disp = np.sum(disp**2, axis=2)  # (n_frames-tau, n_atoms)
        
        # 对原子平均，然后对时间原点平均
        msd_per_atom = np.mean(squared_disp, axis=1)
        msd[tau] = np.mean(msd_per_atom)
        counts[tau] = len(msd_per_atom)
    
    return msd, counts


def msd_from_unwrapped_simple(unwrapped_cart):
    """简单的MSD计算（相对于第一帧）"""
    r0 = unwrapped_cart[0]
    disp = unwrapped_cart - r0
    msd = np.mean(np.sum(disp**2, axis=2), axis=1)
    return msd


def check_linearity(time, msd, start_idx, end_idx, threshold=0.95):
    """检查MSD线性度"""
    if end_idx - start_idx < 10:
        return False, 0.0  # 数据点太少
    
    slope, intercept, r_value, p_value, std_err = linregress(
        time[start_idx:end_idx], msd[start_idx:end_idx]
    )
    
    is_linear = (r_value**2 > threshold)
    return is_linear, r_value**2


def find_best_fit_region(time, msd, min_frac=0.1, max_frac=0.8, step=0.05):
    """自动寻找最佳拟合区域"""
    best_r2 = 0
    best_start = 0
    best_end = 0
    
    n_points = len(time)
    
    for start_frac in np.arange(min_frac, max_frac, step):
        for end_frac in np.arange(start_frac + 0.2, max_frac + step, step):
            start_idx = int(start_frac * n_points)
            end_idx = int(end_frac * n_points)
            
            if end_idx - start_idx < 10:
                continue
                
            _, _, r_value, _, _ = linregress(
                time[start_idx:end_idx], msd[start_idx:end_idx]
            )
            
            r2 = r_value**2
            if r2 > best_r2:
                best_r2 = r2
                best_start = start_idx
                best_end = end_idx
    
    return best_start, best_end, best_r2


def compute_diffusion(trajfile, outprefix="diffusion", start_frac=0.2, end_frac=0.8,
                      timestep_ps=0.5, stride=10, max_time_ps=300.0, remove_com=True,
                      robust_msd=False, auto_fit=False):

    # 读取轨迹（subsample）
    traj = read(trajfile, index=f"::{stride}")
    n_frames_total = len(traj)

    frame_interval_ps = timestep_ps * stride
    n_frames = min(n_frames_total, int(max_time_ps / frame_interval_ps))
    traj = traj[:n_frames]

    n_atoms = len(traj[0])
    print(f"[INFO] Loaded {n_frames_total} frames, using first {n_frames} frames "
          f"(~{max_time_ps} ps, stride={stride}), {n_atoms} atoms")

    # 检查晶胞是否变化（NPT判断）
    cell_variation = np.std([np.linalg.det(atoms.cell.array) for atoms in traj])
    if cell_variation / np.linalg.det(traj[0].cell.array) > 0.01:
        print("[INFO] Detected variable cell (likely NPT simulation)")
    else:
        print("[INFO] Detected fixed cell (likely NVT simulation)")

    # 解包轨迹
    unwrapped = unwrap_positions_cartesian(traj, remove_com=remove_com)
    
    # 计算MSD
    if robust_msd and n_frames > 100:
        print("[INFO] Using robust MSD calculation with time averaging")
        msd, counts = msd_from_unwrapped_robust(unwrapped)
        time = np.arange(len(msd)) * frame_interval_ps
        # 只使用有足够统计量的数据点
        valid = counts > (n_frames * 0.1)
        time = time[valid]
        msd = msd[valid]
    else:
        print("[INFO] Using simple MSD calculation")
        msd = msd_from_unwrapped_simple(unwrapped)
        time = np.arange(len(msd)) * frame_interval_ps

    # 确定拟合区域
    if auto_fit:
        start_idx, end_idx, best_r2 = find_best_fit_region(time, msd)
        start_frac_used = start_idx / len(time)
        end_frac_used = end_idx / len(time)
        print(f"[INFO] Auto-fit selected region: {start_frac_used:.2f}-{end_frac_used:.2f} (R² = {best_r2:.3f})")
    else:
        start_idx = int(start_frac * len(time))
        end_idx = int(end_frac * len(time))

    # 线性拟合
    if end_idx - start_idx < 5:
        print("[WARNING] Too few points for fitting. Adjusting fit region.")
        start_idx = int(0.1 * len(time))
        end_idx = int(0.7 * len(time))

    slope, intercept, r_value, p_value, std_err = linregress(
        time[start_idx:end_idx], msd[start_idx:end_idx]
    )

    D_A2_ps = slope / 6.0
    D_err = std_err / 6.0
    D_cm2_s = D_A2_ps * 1e-4

    # 检查线性度
    is_linear, r2 = check_linearity(time, msd, start_idx, end_idx)
    
    print(f"[RESULT] D = {D_A2_ps:.3f} ± {D_err:.3f} Å²/ps = {D_cm2_s:.3e} cm²/s")
    print(f"[INFO] MSD slope = {slope:.3f} ± {std_err:.3f} Å²/ps")
    print(f"[INFO] R² = {r2:.3f}")
    
    if not is_linear:
        print("[WARNING] MSD may not be linear (R² < 0.95)")
    if r2 < 0.9:
        print("[WARNING] Poor linear fit - consider adjusting fit region or trajectory length")

    # 绘图
    plt.figure(figsize=(10, 8))
    
    # MSD曲线
    plt.plot(time, msd, 'b-', label="MSD", alpha=0.7, linewidth=2)
    
    # 拟合区域标记
    plt.axvspan(time[start_idx], time[end_idx-1], alpha=0.2, color='red', 
                label=f"Fit region (R² = {r2:.3f})")
    
    # 拟合线
    fit_line = slope * time + intercept
    plt.plot(time, fit_line, 'r--', linewidth=2,
             label=f"Fit (D = {D_cm2_s:.2e} cm²/s)")
    
    plt.xlabel("Time (ps)", fontsize=12)
    plt.ylabel("MSD (Å²)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.title(f"Mean Squared Displacement\nD = {D_cm2_s:.2e} cm²/s", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{outprefix}_msd.png", dpi=300, bbox_inches='tight')
    
    # 保存数据
    np.savetxt(f"{outprefix}_msd.dat", np.column_stack((time, msd)),
               header="Time(ps) MSD(Å^2)")
    
    # 保存拟合结果
    with open(f"{outprefix}_results.txt", "w") as f:
        f.write(f"Diffusion Coefficient Analysis Results\n")
        f.write(f"=====================================\n")
        f.write(f"Trajectory file: {trajfile}\n")
        f.write(f"Frames used: {n_frames} (of {n_frames_total})\n")
        f.write(f"Time window: {time[-1]:.1f} ps\n")
        f.write(f"Atoms: {n_atoms}\n")
        f.write(f"Fit region: {time[start_idx]:.1f} - {time[end_idx-1]:.1f} ps\n")
        f.write(f"MSD slope: {slope:.3f} ± {std_err:.3f} Å²/ps\n")
        f.write(f"R²: {r2:.3f}\n")
        f.write(f"Diffusion coefficient: {D_A2_ps:.3f} ± {D_err:.3f} Å²/ps\n")
        f.write(f"Diffusion coefficient: {D_cm2_s:.3e} cm²/s\n")

    plt.close()
    
    return D_A2_ps, D_cm2_s, r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute diffusion coefficient with unwrapped MSD (PBC corrected)"
    )
    parser.add_argument("--traj", type=str, required=True, help="ASE trajectory file (.traj, .xyz, etc.)")
    parser.add_argument("--outprefix", type=str, default="diffusion", help="Prefix for output files")
    parser.add_argument("--start_frac", type=float, default=0.3, help="Fraction of trajectory to start fit")
    parser.add_argument("--end_frac", type=float, default=0.8, help="Fraction of trajectory to end fit")
    parser.add_argument("--timestep_ps", type=float, default=0.5, help="Time per frame in ps")
    parser.add_argument("--stride", type=int, default=10, help="Stride for reading frames")
    parser.add_argument("--max-time_ps", type=float, default=300.0, help="Maximum time window (ps) to use")
    parser.add_argument("--no-remove-com", action="store_true", help="Disable COM drift removal")
    parser.add_argument("--robust-msd", action="store_true", help="Use robust MSD calculation with time averaging")
    parser.add_argument("--auto-fit", action="store_true", help="Automatically find best fit region")

    args = parser.parse_args()

    D_A2_ps, D_cm2_s, r2 = compute_diffusion(
        args.traj,
        args.outprefix,
        args.start_frac,
        args.end_frac,
        args.timestep_ps,
        args.stride,
        max_time_ps=args.max_time_ps,
        remove_com=(not args.no_remove_com),
        robust_msd=args.robust_msd,
        auto_fit=args.auto_fit
    )
