from __future__ import annotations
import tyro
import torch
import mjlab.tasks  # noqa: F401  触发任务注册
import numpy as np
import pandas as pd
from pathlib import Path
from math import nextafter
from typing import Any, Literal
import matplotlib.pyplot as plt
from dataclasses import dataclass
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import NativeMujocoViewer
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.wrappers import VideoRecorder
from mjlab.tasks.SQuRo_Backup.mdp.command import BackupCommandCfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import (
    _MODEL_INDICES,
    resolve_model_indices,
)
from mjlab.tasks.SQuRo_Backup.mdp.timing import STAND_CONFIRM_DURATION



FL_HOLD = (-0.28, 0.55)
HL_HOLD = (-1.50, -0.25)
LEG_INIT = [0.1, -0.3, 0.1, -0.3, -0.1, 0.3, -0.1, 0.3]

T1 = 0.65
T2 = 0.15
T3 = 0.15
T4 = 0.5
T5 = 1.05

T_SEG1_END = T1
T_SEG2_END = T_SEG1_END + T2
T_SEG3_END = T_SEG2_END + T3
T_SEG4_END = T_SEG3_END + T4
T_TOTAL = T_SEG4_END + T5
T_OFFSET = 1.0



def slow1_target(current_time: float, scale: float = 10.0) -> list[float]:
    time1 = T_OFFSET
    time2 = time1 + T1 * scale
    time3 = time2 + T2 * scale
    time4 = time3 + T3 * scale
    time5_end = time4 + T4 * scale
    r = [0.0, 0.0, 0.0, 0.0, 0.1, -0.3, 0.1, -0.3, 0.0, 0.0, -0.1, 0.3, -0.1, 0.3]
    if current_time <= time1:
        return r
    r[4], r[5] = FL_HOLD; r[6], r[7] = FL_HOLD
    r[10], r[11] = HL_HOLD; r[12], r[13] = HL_HOLD
    if current_time < time2:
        u = (current_time - time1) / (time2 - time1)
        r[0] = 0.6 * u
        r[1] = -1.57 * u
        r[8] = 0.6 * u 
        r[9] = 1.57 * u
    elif current_time < time3:
        u = (current_time - time2) / (time3 - time2)
        r[0] = 0.6 - 0.4 * u
        r[1] = -1.57
        r[8] = 0.6 - 0.8 * u
        r[9] = 1.57
    elif current_time < time4:
        u = (current_time - time3) / (time4 - time3)
        r[0] = 0.2 + 0.4 * u
        r[1] = -1.57 + 1.57 * u
        r[8] = -0.2 + 0.2 * u
        r[9] = 1.57 - 1.57 * u
    else:
        # T4: 腿从支撑角平滑回站立角；F_spine1 承接 T3 末端在过渡段内线性回零，
        # 其余脊柱保持零；之后保持站立。与 mdp/reference.py 的 T4 必须逐字一致。
        u = min(1.0, (current_time - time4) / (time5_end - time4))
        r[4] = FL_HOLD[0] + u * (LEG_INIT[0] - FL_HOLD[0])
        r[5] = FL_HOLD[1] + u * (LEG_INIT[1] - FL_HOLD[1])
        r[6], r[7] = r[4], r[5]
        r[10] = HL_HOLD[0] + u * (LEG_INIT[4] - HL_HOLD[0])
        r[11] = HL_HOLD[1] + u * (LEG_INIT[5] - HL_HOLD[1])
        r[12], r[13] = r[10], r[11]
        r[0] = 0.6 * (1.0 - u)
    return r



class StateMachinePolicy:
    def __init__(self, env: ManagerBasedRlEnv, time_scale: float, max_retry: int, buffer: float | None = None, action_scale: float | None = None, log_events: bool = True) -> None:
        self.env = env
        self.asset = env.unwrapped.scene.entities["robot"]
        resolve_model_indices(self.asset)
        self.default = self.asset.data.default_joint_pos[:, _MODEL_INDICES.joint_ids]
        self.lam = time_scale
        self.max_retry = max_retry
        command_cfg: BackupCommandCfg = env.unwrapped.cfg.commands["backup_cmd"]  # type: ignore[assignment]
        self.p1_buffer_s = float(command_cfg.p1_buffer_s if buffer is None else buffer)
        self.p2_buffer_s = float(command_cfg.p2_buffer_s if buffer is None else buffer)
        self.pose_confirm_s = float(getattr(command_cfg, "pose_confirm_s", 0.10))
        self.inverted_confirm_s = float(getattr(command_cfg, "inverted_confirm_s", 0.15))
        env_scale = float(env.unwrapped.cfg.actions["joint_pos"].scale)  # type: ignore[union-attr]
        self.action_scale = env_scale if action_scale is None else float(action_scale)
        self.log_events = log_events
        self.fb = _MODEL_INDICES.f_body_id
        self.hb = _MODEL_INDICES.h_body_id
        self.phase = "P1"
        self.t_phase = 0.0
        self.retry = {"P1": 0, "P2": 0}
        self.events: list[str] = []
        self.stand_t: float | None = None
        self._elapsed = 0.0       # 累计仿真时间 (s), 用于脊柱期望角打印
        self._step_cnt = 0        # 步计数, 每 4 步 (0.02s) 打印一次
        self._spn_ts: list[float] = []        # 每步仿真时间 (s)
        self._spn_ref: list[list[float]] = [] # 四个脊柱期望角 [F_sp1, F_body, H_sp1, H_body]
        self._spn_act: list[list[float]] = [] # 四个脊柱实际关节角
        self._s1_confirm_t = 0.0
        self._s2_confirm_t = 0.0
        self._inverted_confirm_t = 0.0
        self._last_sim_step = int(env.unwrapped.common_step_counter)
        self._last_episode_step = int(env.unwrapped.episode_length_buf[0])
        self._milestones = {"S1": False, "S2": False}


    # 诊断输出使用背腹 site 的世界高度差；阶段门控直接复用训练环境判据。
    def _uprightness(self, idx: int) -> float:
        # 返回背腹轴的世界 Z 分量 (已统一符号): >0 = 正置(腹面朝下), <0 = 倒置(腹面朝上)
        pairs = _MODEL_INDICES.segment_belly_back_ids
        assert pairs is not None, "segment_belly_back_ids 未解析"
        belly_id, back_id = pairs[idx]
        sp = self.asset.data.site_pos_w[0]
        return float(sp[back_id, 2] - sp[belly_id, 2])


    def _fz(self, body_id: int) -> float:
        return self.asset.data.body_link_pos_w[0, body_id, 2].item()


    def _state(self) -> tuple[float, float]:
        return self._uprightness(0), self._uprightness(1)


    # S1: 后段已翻正、前段未翻正, 且两段躯干都平躺贴地
    def _is_S1(self) -> bool:
        command = self.env.unwrapped.command_manager.get_term("backup_cmd")
        return bool(command._check_S1()[0]) # type: ignore


    # S2: 共用训练判据，允许贴地翻正或已经达到双段站立几何。
    def _is_S2(self) -> bool:
        command = self.env.unwrapped.command_manager.get_term("backup_cmd")
        return bool(command._check_S2()[0]) # type: ignore


    # 双段同时倒置候选；仅用于 P2/P3 回退，不把侧立或未知状态当作终止。
    def _is_both_inverted(self) -> bool:
        command = self.env.unwrapped.command_manager.get_term("backup_cmd")
        return bool(command._check_both_inverted()[0])  # type: ignore


    # 更新指定候选的连续确认时钟；姿态组合任一采样不完整即清零。
    def _update_confirmation(self, name: str, candidate: bool, dt: float) -> bool:
        current = getattr(self, name)
        current = current + dt if candidate else 0.0
        setattr(self, name, current)
        limit = self.pose_confirm_s if name in ("_s1_confirm_t", "_s2_confirm_t") else self.inverted_confirm_s
        # 与训练 float32 确认计时的边界余量一致。
        eps = torch.finfo(torch.float32).eps * max(1.0, abs(limit), abs(dt)) * 8.0
        return current >= limit - eps


    # 清除阶段切换或重试留下的候选确认时钟，不重置机器人物理状态。
    def _clear_confirmation(self) -> None:
        self._s1_confirm_t = 0.0
        self._s2_confirm_t = 0.0
        self._inverted_confirm_t = 0.0


    def _log(self, msg: str) -> None:
        if self.log_events:
            print(f"[SM] {msg}", flush=True)
        self.events.append(msg)


    # 只消费新完成的物理步；查看器重复查询策略时不能重复累计确认时间。
    def _sync_state(self) -> None:
        env = self.env.unwrapped
        step = int(env.common_step_counter)
        episode_step = int(env.episode_length_buf[0])
        steps = step - self._last_sim_step
        if episode_step < self._last_episode_step or (episode_step == 0 and steps != 0):
            self.phase = "P1"
            self.t_phase = 0.0
            self.retry = {"P1": 0, "P2": 0}
            self.stand_t = None
            self._milestones = {"S1": False, "S2": False}
            self._clear_confirmation()
            self._log("环境回合重置 -> 同步回 P1，清除连续确认与里程碑记录")
            termination = getattr(env, "termination_manager", None)
            if steps > 0 and termination is not None and bool(termination.get_term("stand")[0]):
                self.phase = "DONE"
                self._log("训练环境确认稳定站起 -> 回放结束")
        elif steps > 0 and episode_step > 0 and self.phase != "DONE":
            if steps > 1:
                # 未观测到中间姿态时，不能假设候选在漏采样期间连续成立。
                self._clear_confirmation()
                self.t_phase += (steps - 1) * self.env.step_dt
            self._advance_state(self.env.step_dt)
        self._last_sim_step = step
        self._last_episode_step = episode_step
        self._elapsed = step * self.env.step_dt


    # 在选择本次控制目标之前完成阶段检测；切换/回退当步直接使用新阶段起点。
    def _advance_state(self, dt: float) -> None:
        # 与训练阶段时钟同用 float32，避免截止时间差一个控制步。
        self.t_phase = float(torch.tensor(self.t_phase, dtype=torch.float32) + dt)
        s1_confirmed = self._update_confirmation("_s1_confirm_t", self._is_S1(), dt)
        s2_confirmed = self._update_confirmation("_s2_confirm_t", self._is_S2(), dt)
        inverted_confirmed = self._update_confirmation("_inverted_confirm_t", self._is_both_inverted(), dt)
        if self.phase == "P1":
            expected = T_SEG2_END * self.lam
            # 与训练同步的段末门控: 本段参考播完之前不推进, 否则参考会瞬移到段末。
            # 截止时成功优先于重试。
            if s1_confirmed and self.t_phase >= expected:
                t_used = self.t_phase
                self.phase = "P2"; self.t_phase = 0.0
                self._clear_confirmation()
                first = not self._milestones["S1"]
                self._milestones["S1"] = True
                self._log(f"S1 达成 (用时 {t_used:.2f}s，{'首次' if first else '再次'}，不重复计里程碑) -> 进入 P2")
            elif self.t_phase >= expected + self.p1_buffer_s:
                self.retry["P1"] += 1; self.t_phase = 0.0
                self._clear_confirmation()
                self._log(f"S1 未达 (缓冲后, 重试 {self.retry['P1']}/{self.max_retry or '不限'})")
                if self.max_retry > 0 and self.retry["P1"] >= self.max_retry:
                    self._log("P1 重试超限, 放弃"); self.phase = "DONE"
        elif self.phase == "P2":
            expected = T3 * self.lam
            deadline = expected + self.p2_buffer_s
            # 与训练一致：仅给仍有效的候选最多一个确认时长的额外等待。
            s2_grace = (self._s2_confirm_t > 0.0 and not s2_confirmed
                        and self.t_phase < deadline + self.pose_confirm_s)
            # P2 同样要等本段参考播完(名义 T3 时长)才推进, 与训练侧的段末门控一致。
            if inverted_confirmed:
                self.phase = "P1"; self.t_phase = 0.0
                self._clear_confirmation()
                self._log("P2 检测到双倒 (连续确认后) -> 回到 P1")
            elif s2_confirmed and self.t_phase >= expected:
                t_used = self.t_phase
                self.phase = "P3"; self.t_phase = 0.0
                self._clear_confirmation()
                first = not self._milestones["S2"]
                self._milestones["S2"] = True
                self._log(f"S2 达成 (用时 {t_used:.2f}s，{'首次' if first else '再次'}，不重复计里程碑) -> 进入 P3")
            elif self.t_phase >= deadline and not s2_grace:
                self.retry["P2"] += 1; self.t_phase = 0.0
                self._clear_confirmation()
                fu, hu = self._state()
                fz, hz = self._fz(self.fb), self._fz(self.hb)
                self._log(f"S2 未达 fu={fu:+.2f} hu={hu:+.2f} fz={fz:.3f} hz={hz:.3f} "
                          f"(缓冲后, 重试 {self.retry['P2']}/{self.max_retry or '不限'})")
                if self.max_retry > 0 and self.retry["P2"] >= self.max_retry:
                    self._log("P2 重试超限, 放弃"); self.phase = "DONE"
        elif self.phase == "P3":
            # P3 既接受双倒回退，也接受 S1 回到 P2；两者均须连续确认。
            if inverted_confirmed:
                self.phase = "P1"; self.t_phase = 0.0
                self._clear_confirmation()
                self._log("P3 检测到双倒 (连续确认后) -> 回到 P1")
            elif s1_confirmed:
                self.phase = "P2"; self.t_phase = 0.0
                self._clear_confirmation()
                self._log("P3 检测到 S1 (连续确认后) -> 回到 P2")
            # 成功判定与训练同源: 直接读训练环境的 stand 终止项(理由见技术细节 §7.2.1)。
            termination = getattr(self.env.unwrapped, "termination_manager", None)
            if self.phase == "P3" and termination is not None and bool(termination.get_term("stand")[0]):
                # 从确认成立的时刻回推确认时长, 即"开始站稳"的时刻。
                self.stand_t = self.t_phase - STAND_CONFIRM_DURATION
                self._log(f"训练环境确认稳定站起 (站稳 {STAND_CONFIRM_DURATION}s, P3 内 t≈{self.stand_t:.2f}s) -> 回放结束")
                self.phase = "DONE"


    def __call__(self, obs: Any) -> torch.Tensor:
        del obs
        self._sync_state()
        # 保留手调参考函数和各段动作时长；只根据已确认的新阶段查询目标。
        if self.phase == "P1":
            tn = min(self.t_phase / self.lam, T_SEG2_END)
            target = slow1_target(T_OFFSET + tn * self.lam, self.lam)
        elif self.phase == "P2":
            # 与手调函数同顺序构造边界；等待时取 T3 左端极限，不能泄漏到 T4。
            # 用户可令 T3 末端与 T4 起点不连续；此处只选择阶段，不改参考公式。
            start = T_OFFSET + T_SEG2_END * self.lam
            end = start + T3 * self.lam
            target = slow1_target(min(start + self.t_phase, nextafter(end, -float("inf"))), self.lam)
        elif self.phase == "P3":
            start = T_OFFSET + T_SEG3_END * self.lam
            target = slow1_target(start + self.t_phase, self.lam)
        else:  # DONE
            target = slow1_target(1e6, self.lam)  # 保持站立

        action = (torch.tensor(target, device=self.default.device, dtype=torch.float32) - self.default[0]) / self.action_scale
        self._step_cnt += 1
        jp = self.asset.data.joint_pos[0, _MODEL_INDICES.joint_ids]
        self._spn_ts.append(self._elapsed)
        self._spn_ref.append([target[0], target[1], target[8], target[9]])
        self._spn_act.append([jp[0].item(), jp[1].item(), jp[8].item(), jp[9].item()])
        return action.unsqueeze(0)


    def plot_spine_curves(self, save_path: str | None = None) -> None:

        if not self._spn_ts:
            print("[SPN] 无数据可绘制")
            return
        names = ["F_spine1", "F_body", "H_spine1", "H_body"]
        ts = np.asarray(self._spn_ts)
        ref = np.asarray(self._spn_ref)  # [N,4]
        act = np.asarray(self._spn_act)  # [N,4]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for i, ax in enumerate(axes.flat):
            ax.plot(ts, ref[:, i], "-", linewidth=1.4, label="期望角")
            ax.plot(ts, act[:, i], "--", linewidth=1.2, label="实际角")
            ax.set_ylabel(f"{names[i]} (rad)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            ax.set_title(names[i])
        axes.flat[-1].set_xlabel("time (s)")
        fig.suptitle(f"SQuRo Backup 脊柱期望/实际角曲线 (λ={self.lam:g})")
        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
            print(f"[SPN] 脊柱曲线已保存: {save_path}")
        else:
            plt.show()
        plt.close(fig)


    def save_spine_csv(self, save_path: str | None = None) -> None:
        if not self._spn_ts:
            print("[SPN] 无数据可保存")
            return
        # 内存顺序均为 [F_sp1, F_body, H_sp1, H_body]
        data = {
            "step": list(range(len(self._spn_ts))),
            "time": self._spn_ts,
            # 实际角 (命名 *_pos, 兼容 SQuRo_compute_spine_primitives.py)
            "F_spine1_pos": [r[0] for r in self._spn_act],
            "F_body_pos": [r[1] for r in self._spn_act],
            "H_spine1_pos": [r[2] for r in self._spn_act],
            "H_body_pos": [r[3] for r in self._spn_act],
            # 期望角
            "F_spine1_ref": [r[0] for r in self._spn_ref],
            "F_body_ref": [r[1] for r in self._spn_ref],
            "H_spine1_ref": [r[2] for r in self._spn_ref],
            "H_body_ref": [r[3] for r in self._spn_ref],
        }
        df = pd.DataFrame(data)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True) # type: ignore
        df.to_csv(save_path, index=False)
        print(f"[SPN] 脊柱CSV已保存: {save_path}")



@dataclass(frozen=True)
class VisConfig:
    time_scale: float = 1.0
    """slow1 时间缩放 (λ)。"""
    max_retry: int = 0
    """每阶段最大重试次数；0 表示不限次数，与训练一致。"""
    # 旧参数：共同覆盖 P1/P2 等待时间；None 时使用环境配置，单位为实际秒。
    buffer: float | None = None
    # 分阶段覆盖优先于 buffer；None 时沿用环境配置或共同覆盖值。
    p1_buffer_s: float | None = None
    p2_buffer_s: float | None = None
    visualize: Literal["none", "viewer", "video"] = "viewer"
    num_envs: int = 1
    device: str | None = None
    duration: float = 20.0
    """最大回放时长 (s)。"""
    video_dir: str = "logs/rsl_rl/SQuRo_Backup/replay_videos"



# 命令行等待时间同时写入内置 RL 状态机，确保两套状态机使用相同配置。
def _configure_command(command_cfg: BackupCommandCfg, args: VisConfig) -> None:
    command_cfg.fixed_time_scale = args.time_scale
    if args.buffer is not None:
        command_cfg.p1_buffer_s = args.buffer
        command_cfg.p2_buffer_s = args.buffer
    if args.p1_buffer_s is not None:
        command_cfg.p1_buffer_s = args.p1_buffer_s
    if args.p2_buffer_s is not None:
        command_cfg.p2_buffer_s = args.p2_buffer_s


def main() -> None:
    args = tyro.cli(VisConfig)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = args.num_envs
    # 注解为具体子类: env_cfg.commands 声明为 dict[str, CommandTermCfg]
    command_cfg: BackupCommandCfg = env_cfg.commands["backup_cmd"]  # type: ignore[assignment]
    _configure_command(command_cfg, args)
    render_mode = "rgb_array" if args.visualize == "video" else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if args.visualize == "video":
        video_folder = Path(args.video_dir)
        video_folder.mkdir(parents=True, exist_ok=True)
        n_frames = int(args.duration / env.step_dt) + 1
        env = VideoRecorder(
            env, video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=n_frames,
            name_prefix=f"sm_ts{args.time_scale}",
            disable_logger=False,
        )

    env.reset()
    policy = StateMachinePolicy(env, args.time_scale, args.max_retry)

    if args.visualize in ("none", "video"):
        n = int(args.duration / env.step_dt)
        for i in range(n):
            with torch.no_grad():
                action = policy(env.unwrapped.get_observations())
            obs, rew, dones, to, extras = env.step(action)
            if policy.phase == "DONE":
                break
        print(f"[RESULT] 最终 phase={policy.phase}, 重试 P1={policy.retry['P1']}, P2={policy.retry['P2']}")
        out_dir = Path(args.video_dir)
        policy.plot_spine_curves(str(out_dir / f"spine_ts{args.time_scale:g}.png"))
        policy.save_spine_csv(str(out_dir / f"spine_ts{args.time_scale:g}.csv"))
        env.close()
        return

    env.reset()
    policy = StateMachinePolicy(env, args.time_scale, args.max_retry)
    viewer = NativeMujocoViewer(env, policy, frame_rate=60)  # type: ignore[arg-type]
    viewer.run(num_steps=int(args.duration / env.step_dt))
    out_dir = Path(args.video_dir)
    policy.plot_spine_curves(str(out_dir / f"spine_ts{args.time_scale:g}.png"))
    policy.save_spine_csv(str(out_dir / f"spine_ts{args.time_scale:g}.csv"))
    env.close()



if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    main()
