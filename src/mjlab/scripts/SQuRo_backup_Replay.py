from __future__ import annotations
import tyro
import torch
import mjlab.tasks  # noqa: F401  触发任务注册
import pandas as pd
from pathlib import Path
from typing import Any, Literal
from dataclasses import dataclass
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import NativeMujocoViewer
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.wrappers import VideoRecorder
from mjlab.tasks.SQuRo_Backup.mdp.indices import (
    _MODEL_INDICES,
    resolve_model_indices,
)


FL_HOLD = (-0.28, 0.55)     # 腿支撑角 (与手调/参考表一致)
HL_HOLD = (-1.50, -0.25)
LEG_INIT = [0.1, -0.3, 0.1, -0.3, -0.1, 0.3, -0.1, 0.3]


# 完整名义参考 (MJLAB 顺序), 0-0.65 段1, 0.65-0.8 段2, 0.8-0.95 段3, 0.95-1.45 time5, 之后站立
def slow1_target(current_time: float, scale: float = 10.0) -> list[float]:
    time1 = 1.0
    time2 = time1 + 0.65 * scale
    time3 = time2 + 0.15 * scale
    time4 = time3 + 0.15 * scale
    time5_end = time4 + 0.5 * scale
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
        r[0] = 0.6 - 0.6 * u
        r[1] = -1.57
        r[8] = 0.6 - 0.6 * u
        r[9] = 1.57
    elif current_time < time4:
        u = (current_time - time3) / (time4 - time3)
        r[0] = 0.6 * u
        r[1] = -1.57 + 1.57 * u
        r[8] = 0.0
        r[9] = 1.57 - 1.57 * u
    else:
        # time5: 腿从支撑角平滑回站立角, F_sp1 0.6 -> 0; 之后保持站立
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
    def __init__(self, env: ManagerBasedRlEnv, time_scale: float, max_retry: int, buffer: float = 0.3, action_scale: float | None = None, log_events: bool = True) -> None:
        self.env = env
        self.asset = env.unwrapped.scene.entities["robot"]
        resolve_model_indices(self.asset)
        self.default = self.asset.data.default_joint_pos[:, _MODEL_INDICES.joint_ids]
        self.lam = time_scale
        self.max_retry = max_retry
        self.buffer = buffer       # 缓冲时间 (s): 超过预期时长后, 缓冲期内继续观察, 未达标才重试
        # 动作反算比例默认跟随环境配置, 保证手调动作与 RL 策略同口径; 显式传值可复现旧脚本
        env_scale = float(env.unwrapped.cfg.actions["joint_pos"].scale)  # type: ignore[union-attr]
        self.action_scale = env_scale if action_scale is None else float(action_scale)
        self.log_events = log_events
        self.fb = _MODEL_INDICES.f_body_id
        self.hb = _MODEL_INDICES.h_body_id
        self.phase = "P1"
        self.t_phase = 0.0
        self.retry = {"P1": 0, "P2": 0}
        self.events: list[str] = []
        self.stand_steps = 0
        self.stand_t: float | None = None
        self._elapsed = 0.0       # 累计仿真时间 (s), 用于脊柱期望角打印
        self._step_cnt = 0        # 步计数, 每 4 步 (0.02s) 打印一次
        self._spn_ts: list[float] = []        # 每步仿真时间 (s)
        self._spn_ref: list[list[float]] = [] # 四个脊柱期望角 [F_sp1, F_body, H_sp1, H_body]
        self._spn_act: list[list[float]] = [] # 四个脊柱实际关节角

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
        return bool(command._check_S1()[0])

    # S2: 两段躯干都已翻正并重新贴地
    def _is_S2(self) -> bool:
        command = self.env.unwrapped.command_manager.get_term("backup_cmd")
        return bool(command._check_S2()[0])

    def _log(self, msg: str) -> None:
        if self.log_events:
            print(f"[SM] {msg}", flush=True)
        self.events.append(msg)

    def __call__(self, obs: Any) -> torch.Tensor:
        del obs
        dt = self.env.step_dt
        if self.phase == "P1":
            expected = 0.8 * self.lam
            if self.t_phase < expected:
                tn = self.t_phase / self.lam
                target = slow1_target(1.0 + tn * self.lam, self.lam)
            else:
                # 缓冲期: 保持段末姿态, 持续检测 S1
                target = slow1_target(1.0 + 0.8 * self.lam, self.lam)
            self.t_phase += dt
            if self.t_phase >= expected and self._is_S1():
                t_used = self.t_phase - dt
                self.phase = "P2"; self.t_phase = 0.0
                self._log(f"S1 达成 (用时 {t_used:.2f}s) -> 进入 P2")
            elif self.t_phase >= expected + self.buffer:
                self.retry["P1"] += 1; self.t_phase = 0.0
                self._log(f"S1 未达 (缓冲后, 重试 {self.retry['P1']}/{self.max_retry})")
                if self.retry["P1"] >= self.max_retry:
                    self._log("P1 重试超限, 放弃"); self.phase = "DONE"
        elif self.phase == "P2":
            expected = 0.15 * self.lam
            if self.t_phase < expected:
                tn = 0.8 + (self.t_phase / expected) * 0.15
                target = slow1_target(1.0 + tn * self.lam, self.lam)
            else:
                # 缓冲期: 保持段3末姿态, 持续检测 S2
                target = slow1_target(1.0 + 0.95 * self.lam, self.lam)
            self.t_phase += dt
            if self.t_phase >= expected and self._is_S2():
                t_used = self.t_phase - dt
                self.phase = "P3"; self.t_phase = 0.0
                self._log(f"S2 达成 (用时 {t_used:.2f}s) -> 进入 P3")
            elif self.t_phase >= expected + self.buffer:
                self.retry["P2"] += 1; self.t_phase = 0.0
                fu, hu = self._state()
                fz, hz = self._fz(self.fb), self._fz(self.hb)
                self._log(f"S2 未达 fu={fu:+.2f} hu={hu:+.2f} fz={fz:.3f} hz={hz:.3f} "
                          f"(缓冲后, 重试 {self.retry['P2']}/{self.max_retry})")
                if self.retry["P2"] >= self.max_retry:
                    self._log("P2 重试超限, 放弃"); self.phase = "DONE"
        elif self.phase == "P3":
            tn = 0.95 + self.t_phase / self.lam
            target = slow1_target(1.0 + tn * self.lam, self.lam)
            self.t_phase += dt
            z = self.asset.data.root_link_pos_w[0, 2].item()
            up = self.asset.data.projected_gravity_b[0, 2].item()
            if z > 0.05 and up > 0.9:
                self.stand_steps += 1
            else:
                self.stand_steps = 0
            if self.stand_steps >= int(0.4 / dt):
                self.stand_t = self.t_phase - (self.stand_steps - 1) * dt
                self._log(f"稳定站起! stand_t≈{self.stand_t:.2f}s (P3 内)")
                self.phase = "DONE"
        else:  # DONE
            target = slow1_target(1e6, self.lam)  # 保持站立

        action = (torch.tensor(target, device=self.default.device, dtype=torch.float32)
                  - self.default[0]) / self.action_scale
        # 每 0.02s 打印一次四个脊柱期望角 (调试用): F_sp1, F_body, H_sp1, H_body
        self._elapsed += dt
        self._step_cnt += 1
        # 记录四个脊柱期望/实际角 (仿真结束后绘图)
        jp = self.asset.data.joint_pos[0, _MODEL_INDICES.joint_ids]
        self._spn_ts.append(self._elapsed)
        self._spn_ref.append([target[0], target[1], target[8], target[9]])
        self._spn_act.append([jp[0].item(), jp[1].item(), jp[8].item(), jp[9].item()])
        if self._step_cnt % 4 == 0:
            print(f"[SPN t={self._elapsed:.2f}s {self.phase}] "
                  f"F_sp1={target[0]:+.3f} F_body={target[1]:+.3f} "
                  f"H_sp1={target[8]:+.3f} H_body={target[9]:+.3f}", flush=True)
        return action.unsqueeze(0)

    def plot_spine_curves(self, save_path: str | None = None) -> None:
        import numpy as np
        import matplotlib.pyplot as plt
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
    max_retry: int = 5
    """每阶段最大重试次数。"""
    buffer: float = 0.3
    """缓冲时间 (s): 超过预期时长后, 缓冲期内持续检测, 未达标才重试。"""
    visualize: Literal["none", "viewer", "video"] = "viewer"
    num_envs: int = 1
    device: str | None = None
    duration: float = 20.0
    """最大回放时长 (s)。"""
    video_dir: str = "logs/rsl_rl/SQuRo_Backup/replay_videos"



def main() -> None:
    args = tyro.cli(VisConfig)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands["backup_cmd"].fixed_time_scale = args.time_scale  # type: ignore[attr-defined]

    render_mode = "rgb_array" if args.visualize == "video" else None
    print(f"[INFO] 状态机可视化: λ={args.time_scale}, max_retry={args.max_retry}, "
          f"visualize={args.visualize}")
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
    policy = StateMachinePolicy(env, args.time_scale, args.max_retry, args.buffer)

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
    policy = StateMachinePolicy(env, args.time_scale, args.max_retry, args.buffer)
    viewer = NativeMujocoViewer(env, policy, frame_rate=60)  # type: ignore[arg-type]
    viewer.run(num_steps=int(args.duration / env.step_dt))
    out_dir = Path(args.video_dir)
    policy.plot_spine_curves(str(out_dir / f"spine_ts{args.time_scale:g}.png"))
    policy.save_spine_csv(str(out_dir / f"spine_ts{args.time_scale:g}.csv"))
    env.close()



if __name__ == "__main__":
    main()
