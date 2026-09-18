import sys

sys.path.insert(0, "src")

from mjlab.tasks.SQuRo_Backup.SQuRo_Backup_env_cfg import SQuRo_Backup_Env_Cfg
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import _CURVES
from mjlab.tasks.SQuRo_Backup.mdp.reference import _generate_reference_table
from mjlab.tasks.SQuRo_Backup.mdp import timing as T

# 提交前验收: 权重净值 / 参考表与手调脚本一致性 / 站起确认时长统一
#
# 注意: 分段端点上参考表按浮点加法会落到下一分支, 故此处取各段"中点"做公式比对;
#       端点与全段的严格一致性由 verify_script_vs_training.py 逐点对拍负责。


def _hand_at(tn: float) -> tuple[float, float, float, float]:
    # 手调脚本分区公式 (与 slow1_target 同构), 用于独立复算
    T1, T2, T3 = T.P1_BUILD_DURATION, T.P1_RECOVER_DURATION, T.P2_DURATION
    T4 = T.STAND_TRANSITION_DURATION
    if tn < T1:
        u = tn / T1
        return (0.6 * u, -1.57 * u, 0.6 * u, 1.57 * u)
    if tn < T1 + T2:
        u = (tn - T1) / T2
        return (0.6 - 0.4 * u, -1.57, 0.6 - 0.8 * u, 1.57)
    if tn < T1 + T2 + T3:
        u = (tn - T1 - T2) / T3
        return (0.2 + 0.4 * u, -1.57 + 1.57 * u, -0.2 + 0.2 * u, 1.57 - 1.57 * u)
    if tn < T1 + T2 + T3 + T4:
        # T4: F_spine1 承接 T3 末端 0.6 线性回零, 其余脊柱保持零
        u = (tn - T1 - T2 - T3) / T4
        return (0.6 * (1.0 - u), 0.0, 0.0, 0.0)
    return (0.0, 0.0, 0.0, 0.0)


def main() -> None:
    ok = True

    print("=== 1) 奖励权重净值 (cfg.weight × _CURVES) ===")
    cfg = SQuRo_Backup_Env_Cfg()
    for name, term in cfg.rewards.items():
        # 权重键默认按 weight_<奖励项名> 推导, 只有命名不一致的少数项需要显式映射
        key = {
            "spine_target": "weight_spine_target", "height": "weight_height",
            "energy": "weight_energy",
            "action_L1": "weight_smooth_L1_leg", "action_L2": "weight_smooth_L2_leg",
        }.get(name, f"weight_{name}")
        if key not in _CURVES:
            print(f"  {name:20s} ** _CURVES 缺少 {key} **")
            ok = False
            continue
        net = term.weight * _CURVES[key][0]
        flag = "" if term.weight == 1.0 else "  ** cfg 非 1.0 **"
        if term.weight != 1.0:
            ok = False
        print(f"  {name:20s} cfg={term.weight:<4g} × {key:24s}={_CURVES[key][0]:<6g} => {net:g}{flag}")

    print("\n=== 2) 参考表 vs 手调脚本分区公式 (取各段中点) ===")
    t, ref = _generate_reference_table()
    import numpy as np
    t1, t2, t3 = T.P1_BUILD_DURATION, T.P1_RECOVER_DURATION, T.P2_DURATION
    checks = [
        ("T1 中", t1 / 2),
        ("T2 中", t1 + t2 / 2),
        ("T3 中", t1 + t2 + t3 / 2),
        ("T4 中", T.P2_END + T.STAND_TRANSITION_DURATION / 2),
    ]
    for label, tn in checks:
        i = int(np.argmin(np.abs(t - tn)))
        got = (ref[i, 0], ref[i, 1], ref[i, 8], ref[i, 9])
        expect = _hand_at(tn)
        same = all(abs(g - e) < 2e-3 for g, e in zip(got, expect))
        if not same:
            ok = False
        print(f"  {label}: 参考表=({got[0]:+.3f}, {got[1]:+.3f}, {got[2]:+.3f}, {got[3]:+.3f})  "
              f"公式=({expect[0]:+.3f}, {expect[1]:+.3f}, {expect[2]:+.3f}, {expect[3]:+.3f})  "
              f"{'OK' if same else '** 不一致 **'}")

    print("\n=== 3) 站起确认时长 ===")
    print(f"  timing.STAND_CONFIRM_DURATION = {T.STAND_CONFIRM_DURATION}")
    from mjlab.tasks.SQuRo_Backup.mdp.terminations import STAND_CONFIRM_DURATION as D
    from mjlab.scripts.SQuRo_Backup_Replay import STAND_CONFIRM_DURATION as R
    same = D == R == T.STAND_CONFIRM_DURATION
    if not same:
        ok = False
    print(f"  terminations 导入值={D}  回放脚本导入值={R}  "
          f"{'OK 两侧一致' if same else '** 不一致 **'}")

    print("\n=== 4) S1/S2 确认与达成窗参数 ===")
    c = cfg.commands["backup_cmd"]
    for k in ("window_late_s", "pose_confirm_s", "inverted_confirm_s",
              "pose_angle_tolerance_deg"):
        print(f"  {k:26s} = {getattr(c, k)}")

    print("\n=== 5) 执行器 ctrlrange 常量 vs SQuRo.xml ===")
    import xml.etree.ElementTree as ET
    from pathlib import Path
    from mjlab.tasks.SQuRo_Backup.mdp.indices import _ACTUATOR_CTRL_RANGE
    xml_path = Path("src/mjlab/asset_zoo/robots/SQuRo/xmls/SQuRo.xml")
    found: dict[str, tuple[float, float]] = {}
    if xml_path.exists():
        for act in ET.parse(xml_path).getroot().iter("position"):
            rng = act.get("ctrlrange")
            joint = act.get("joint")
            if rng and joint:
                lo, hi = (float(v) for v in rng.split())
                found[joint] = (lo, hi)
    else:
        print(f"  ** 找不到 XML: {xml_path} **")
        ok = False
    for name, rng in _ACTUATOR_CTRL_RANGE.items():
        ref = found.get(name)
        same = ref is not None and abs(ref[0] - rng[0]) < 1e-9 and abs(ref[1] - rng[1]) < 1e-9
        if not same:
            ok = False
        print(f"  {name:20s} 常量=({rng[0]:+.2f}, {rng[1]:+.2f})  XML={ref}  "
              f"{'OK' if same else '** 不一致 **'}")
    if set(_ACTUATOR_CTRL_RANGE) != set(found):
        ok = False
        print(f"  ** 常量与 XML 的关节集合不同: 常量 {len(_ACTUATOR_CTRL_RANGE)} 项, XML {len(found)} 项 **")

    print(f"\n[总判定] {'全部通过' if ok else '** 存在问题 **'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
