import numpy as np
from ase.io import read
from scipy.stats import linregress
import argparse

############################################################
# 计算 MSD 的函数
############################################################
def calculate_msd(positions):
    if positions.ndim == 2:
        positions = positions[:, np.newaxis, :]  # (steps, 1, 3)

    n_steps, n_atoms, n_columns = positions.shape
    halfNstp = n_steps // 2
    msd = np.zeros((halfNstp, n_columns))

    for col in range(n_columns):
        for tau in range(halfNstp):
            if tau == 0:
                msd[tau, col] = 0.0
            else:
                time_origins = np.linspace(0, n_steps - tau - 1, halfNstp).astype(int)
                squared_displacements = [
                    (positions[t + tau, :, col] - positions[t, :, col])**2
                    for t in time_origins
                ]
                msd[tau, col] = np.mean(squared_displacements)

    return msd

############################################################
# 主程序
############################################################
def main(args):
    # 读取轨迹
    frames = read(args.trajfile, index=":")
    n_steps = len(frames)
    print(f"[INFO] Loaded {n_steps} frames from {args.trajfile}")

    # 相邻帧间隔: 500 步 × 1 fs = 500 fs = 0.5 ps
    time_step_fs = 100 * 1.0   # fs
    time_ps = (np.arange(n_steps) * time_step_fs) / 1000.0  # ps

    # 提取坐标
    positions = np.array([frame.get_positions() for frame in frames])

    # ====== 计算 MSD ======
    msd = calculate_msd(positions)   # (n_steps//2, 3)
    msd_total = msd.mean(axis=1)     # 求三方向平均

    # ====== 保存到文件 ======
    np.savetxt("msd_from_traj.dat",
               np.column_stack([np.arange(msd.shape[0]), msd, msd_total]),
               header="tau_index MSD_x MSD_y MSD_z MSD_total")

    # ====== 计算扩散系数 ======
    # 使用百分比选取拟合区间
    start_frac, end_frac = args.fit_range
    fit_start = int(start_frac * len(msd_total))
    fit_end   = int(end_frac   * len(msd_total))

    slope, intercept, r_value, p_value, std_err = linregress(
        time_ps[fit_start:fit_end], msd_total[fit_start:fit_end]
    )

    D = slope / 6.0  # 3D 系统
    print("✅ MSD 计算完成，结果保存在 msd_from_traj.dat")
    print(f"拟合区间: {start_frac*100:.0f}% ~ {end_frac*100:.0f}% (索引 {fit_start}~{fit_end})")
    print(f"扩散系数 D = {D:.4e} Å^2/ps")

    # ====== 打印部分结果 ======
    print("\n部分结果预览（单位 Å^2）：")
    print("tau_index   MSD_x     MSD_y     MSD_z     MSD_total")
    for i in range(0, len(msd_total), max(1, len(msd_total)//10)):
        print(f"{i:5d}   {msd[i,0]:8.3f}  {msd[i,1]:8.3f}  {msd[i,2]:8.3f}  {msd_total[i]:8.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MSD and diffusion coefficient from trajectory")
    parser.add_argument("trajfile", type=str, help="Trajectory file (e.g. trajectory.traj)")
    parser.add_argument("--fit_range", type=float, nargs=2, default=[0.2, 0.8],
                        help="Fractional range of trajectory for linear fit, e.g. 0.2 0.5")
    args = parser.parse_args()
    main(args)
