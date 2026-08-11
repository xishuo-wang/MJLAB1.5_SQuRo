"""SQuRo Backup 开环状态机验证可视化脚本 (MJLAB / mujoco-warp)。

复刻手调 slow1 分段动作 + 状态门控重试:
  P1: 段1+2 (脊柱展开+回收)  -> 检测 S1 (F 朝上 / H 朝下 / 180°扭转 / 平躺)
  P2: 段3 (前肢扭转向下)      -> 检测 S2 (F/H 都朝下 / 趴地)
  P3: time5 过渡 + 站立保持   -> 检测稳定站立
未达标时重试当前阶段 (--max-retry 上限)。

用法:
    # 本机 GUI 交互查看
    uv run python src/mjlab/scripts/SQuRo_backup_statemachine_vis.py --time-scale 2 --visualize viewer
    # 录制视频
    uv run python src/mjlab/scripts/SQuRo_backup_statemachine_vis.py --time-scale 2 --visualize video
    # 无头打印状态事件
    uv run python src/mjlab/scripts/SQuRo_backup_statemachine_vis.py --time-scale 2 --visualize none
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tyro
import torch

import mjlab.tasks  # noqa: F401  触发任务注册
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import (
    _MODEL_INDICES,
    resolve_model_indices,
)
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer


FL_HOLD = (-0.28, 0.55)     # 腿支撑角 (与手调/参考表一致)
HL_HOLD = (-1.50, -0.25)
LEG_INIT = [0.1, -0.3, 0.1, -0.3, -0.1, 0.3, -0.1, 0.3]

# 状态检测阈值
_UP_TH = 0.5        # 背腹轴朝上/朝下判定阈值
_GROUND_TH = 0.03   # 贴地高度阈值
_GROUND_TH_S2 = 0.04  # S2(趴地) 高度阈值: 段3 末 H 后肢略翘(≈0.034), 放宽到 0.04


# 完整名义参考 (MJLAB 顺序), 0-0.65 段1, 0.65-0.8 段2, 0.8-0.95 段3, 0.95-1.45 time5, 之后站立
def slow1_target(current_time: float, scale: float = 10.0) -> list[float]:
    # 与 Loco_Backup_slow1.py 手调分段控制完全一致 (time1 等待结束 -> 段1/段2/段3),
    # 并在 time4 之后加入 time5 平滑过渡 (腿从支撑角平滑回站立角, F_sp1 0.6 -> 0)
    time1 = 1.0
    time2 = time1 + 0.65 * scale
    time3 = time2 + 0.15 * scale
    time4 = time3 + 0.15 * scale
    time5_end = time4 + 0.5 * scale
    # MJLAB actuator 顺序: [F_sp1, F_body, Neck_yaw, Neck_pitch,
    #                       FL_sh, FL_el, FR_sh, FR_el,
    #                       H_sp1, H_body, HL_hip, HL_knee, HR_hip, HR_knee]
    r = [0.0, 0.0, 0.0, 0.0, 0.1, -0.3, 0.1, -0.3, 0.0, 0.0, -0.1, 0.3, -0.1, 0.3]
    if current_time <= time1:
        return r  # 等待段 (保持站立)
    # 动作段: 腿切到支撑角
    r[4], r[5] = FL_HOLD; r[6], r[7] = FL_HOLD
    r[10], r[11] = HL_HOLD; r[12], r[13] = HL_HOLD
    if current_time < time2:
        # 段1: 脊柱展开扭转
        u = (current_time - time1) / (time2 - time1)
        r[0] = 0.6 * u
        r[1] = -1.57 * u
        r[8] = 0.6 * u
        r[9] = 1.57 * u
    elif current_time < time3:
        # 段2: 保持扭转
        u = (current_time - time2) / (time3 - time2)
        r[0] = 0.6 - 0.6 * u
        r[1] = -1.57
        r[8] = 0.6 - 0.6 * u
        r[9] = 1.57
    elif current_time < time4:
        # 段3: 前肢扭回朝下
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
    """开环状态机策略: 按 phase 输出目标动作, 内部检测 S1/S2 并推进/重试。"""

    def __init__(self, env: ManagerBasedRlEnv, time_scale: float, max_retry: int,
                 buffer: float = 0.3,
                 action_scale: float = 0.3, log_events: bool = True) -> None:
        self.env = env
        self.asset = env.unwrapped.scene.entities["robot"]
        resolve_model_indices(self.asset)
        self.default = self.asset.data.default_joint_pos[:, _MODEL_INDICES.joint_ids]
        self.lam = time_scale
        self.max_retry = max_retry
        self.buffer = buffer       # 缓冲时间 (s): 超过预期时长后, 缓冲期内继续观察, 未达标才重试
        self.action_scale = action_scale
        self.log_events = log_events
        self.fb = _MODEL_INDICES.f_body_id
        self.hb = _MODEL_INDICES.h_body_id
        self.phase = "P1"
        self.t_phase = 0.0
        self.retry = {"P1": 0, "P2": 0}
        self.events: list[str] = []
        self.stand_steps = 0
        self.stand_t: float | None = None

    def _body_up(self, body_id: int, sign: float) -> float:
        q = self.asset.data.body_link_quat_w[0, body_id]
        w, x, y, z = q[0].item(), q[1].item(), q[2].item(), q[3].item()
        return sign * 2.0 * (y*z + w*x)   # body+Y 世界 Z 分量; H 用 sign=-1 修正

    def _fz(self, body_id: int) -> float:
        return self.asset.data.body_link_pos_w[0, body_id, 2].item()

    def _state(self) -> tuple[float, float]:
        return self._body_up(self.fb, +1.0), self._body_up(self.hb, -1.0)

    def _is_S1(self) -> bool:
        fu, hu = self._state()
        return fu > _UP_TH and hu < -_UP_TH and self._fz(self.fb) < _GROUND_TH and self._fz(self.hb) < _GROUND_TH

    def _is_S2(self) -> bool:
        fu, hu = self._state()
        return fu < -_UP_TH and hu < -_UP_TH and self._fz(self.fb) < _GROUND_TH_S2 and self._fz(self.hb) < _GROUND_TH_S2

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
        return action.unsqueeze(0)


@dataclass(frozen=True)
class VisConfig:
    time_scale: float = 3.0
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
        env.close()
        return

    env.reset()
    policy = StateMachinePolicy(env, args.time_scale, args.max_retry, args.buffer)
    viewer = NativeMujocoViewer(env, policy, frame_rate=60)  # type: ignore[arg-type]
    viewer.run(num_steps=int(args.duration / env.step_dt))
    env.close()


if __name__ == "__main__":
    main()
