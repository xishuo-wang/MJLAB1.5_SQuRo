from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


# ==================================================================================================
# 文件路径配置
CSV_PATH = r"D:\MuJoCoLab_1.5\logs\rsl_rl\SQuRo_Backup\replay_videos\spine_ts1.csv"

# 控制时间配置（与仿真一致）
TIMESTEP = 0.002          # 原始仿真步长 (s)
DECIMATION = 5            # 策略控制周期倍数
DT = TIMESTEP * DECIMATION  # 实际数据采样间隔 (s)

# 分析时间配置
START_TIME = 0.0          # 分析区间起点 (s)
END_TIME = 1.5            # 分析区间终点 (s)

# 脊柱关节通道配置: (关节名, CSV列名, θ_max, 基元角色)
JOINT_CHANNELS = [
    ("F_body",   "F_body_pos",   1.57, "前肢扭转"),
    ("F_spine1", "F_spine1_pos", 0.60, "侧摆"),
    ("H_spine1", "H_spine1_pos", 0.60, "俯仰"),
    ("H_body",   "H_body_pos",   1.57, "后肢扭转"),
]
# ==================================================================================================


# 从 CSV 恢复时间轴
def load_time(df: pd.DataFrame) -> np.ndarray:
    if "step" in df.columns:
        return df["step"].to_numpy(dtype=float) * DT
    if "time" in df.columns:
        return df["time"].to_numpy(dtype=float)
    raise ValueError("CSV 中缺少 'step' 或 'time' 列，无法生成时间轴")


# 计算四个脊柱关节的基元指标
def compute_spine_primitives(df: pd.DataFrame) -> tuple[dict, float, int]:
    t = load_time(df)
    mask = (t >= START_TIME) & (t <= END_TIME)
    if mask.sum() < 2:
        raise ValueError(f"区间 [{START_TIME}, {END_TIME}] 内有效采样点不足")

    t_sel = t[mask]
    T = t_sel[-1] - t_sel[0]   # 实际区间时长，与手动指定一致

    results = {}
    S_list = []

    for joint_name, col, thmax, role in JOINT_CHANNELS:
        if col not in df.columns:
            raise KeyError(f"CSV 缺少列 '{col}' (用于 {joint_name}/{role})")

        theta = df[col].to_numpy(dtype=float)[mask]
        s = np.abs(theta)                              # 原始数据，取绝对值，不滤波
        S = float(np.sum(s) * DT)                      # 激活面积 (rad·s)
        A = S / (thmax * T)                            # 激活度
        total_s = float(np.sum(s))
        if total_s > 0:
            tau = (float(np.sum(t_sel * s)) / total_s) / T  # 归一化时序中心 [0,1]
        else:
            tau = float("nan")

        results[joint_name] = {
            "role": role,
            "theta_max": thmax,
            "S": S,
            "A": A,
            "tau": tau,
        }
        S_list.append(S)

    # 计算贡献系数
    sum_S = float(np.sum(S_list))
    for joint_name in results:
        results[joint_name]["C"] = results[joint_name]["S"] / sum_S if sum_S > 0 else float("nan")

    return results, T, mask.sum()


# 打印格式化的指标结果
def print_results(results: dict, T: float, n_points: int) -> None:
    print("\n" + "=" * 70)
    print(f"SQuRo 脊柱基元指标计算结果")
    print(f"分析区间: [{START_TIME:.3f}, {END_TIME:.3f}] s | T = {T:.3f} s | 采样点数 = {n_points} | dt = {DT:g} s")
    print("=" * 70)

    # 定义各指标块的打印逻辑
    metrics = [
        ("[1] 激活面积 S (rad·s)", "S"),
        ("[2] 脊柱基元激活度 A = S / (θ_max · T)", "A"),
        ("[3] 脊柱基元贡献系数 C = S_i / ΣS  (四通道之和 = 1)", "C"),
        ("[4] 脊柱基元时序中心 τ  (0=任务起点, 1=任务终点)", "tau"),
    ]

    for title, key in metrics:
        # 收集该指标的所有值
        values = [results[joint_name][key] for joint_name, _, _, _ in JOINT_CHANNELS]
        
        # 打印简洁数值列表（标题重复）
        value_strs = [f"{v:.4f}" for v in values]
        print(f"\n{title}")
        print(f"[{', '.join(value_strs)}]")

    print("=" * 70)


def main() -> None:
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 CSV 文件: {csv_path}")

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("CSV 为空")

    results, T, n_points = compute_spine_primitives(df)
    print_results(results, T, n_points)


if __name__ == "__main__":
    main()