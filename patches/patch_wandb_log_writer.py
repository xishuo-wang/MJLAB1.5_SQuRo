"""补丁脚本：修复 rsl_rl WandbLogWriter 两个问题。

1. SameFileError: wandb.init(dir=log_dir) 将 wandb 内部文件放在 log_dir 下，
   rglob("*.mp4") 扫到已拷贝的视频副本，再次复制时触发 SameFileError。
   → 改为 dir=os.path.dirname(log_dir) 存放到父级目录。

2. Step 单调性警告: wandb 系统指标在后台线程自动递增 step 计数器，
   导致训练循环的显式 step 被判断为"倒退"而被拒绝。
   → 设置 _disable_stats=True 禁用系统指标的自动 step 递增。

用法: uv run python patches/patch_wandb_log_writer.py
"""

import os
from pathlib import Path


def patch() -> None:
    # 找到 site-packages 中的文件
    import rsl_rl.utils.wandb_log_writer as wlw_mod

    file_path = Path(wlw_mod.__file__)

    content = file_path.read_text(encoding="utf-8")

    # 补丁 1: dir=log_dir → dir=os.path.dirname(log_dir)
    old_dir = "dir=log_dir,"
    new_dir = "dir=os.path.dirname(log_dir),"
    if old_dir in content and new_dir not in content:
        content = content.replace(old_dir, new_dir, 1)
        print(f"[PATCH] dir=log_dir → dir=os.path.dirname(log_dir)")
    else:
        print("[PATCH] dir 补丁已应用或未找到目标，跳过")

    # 补丁 2: _disable_stats=True
    old_stats = "settings=wandb.Settings(start_method=\"thread\"),"
    new_stats = "settings=wandb.Settings(start_method=\"thread\", _disable_stats=True),"
    if old_stats in content and new_stats not in content:
        content = content.replace(old_stats, new_stats, 1)
        print(f"[PATCH] _disable_stats=True 已添加")
    else:
        print("[PATCH] _disable_stats 补丁已应用或未找到目标，跳过")

    # 补丁 3: 添加注释说明
    if "# dir 指向父级目录" not in content:
        comment_marker = "# 使用时间戳目录名作为 run ID，避免随机后缀"
        extra_comment = (
            "# dir 指向父级目录，避免 wandb 内部文件（媒体副本）混入 log_dir\n"
            "        # 被 rglob(\"*.mp4\") 扫到时触发 SameFileError\n        "
        )
        if comment_marker in content:
            content = content.replace(comment_marker, comment_marker + "\n        " + extra_comment)
            print("[PATCH] 注释已添加")

    file_path.write_text(content, encoding="utf-8")
    print(f"[PATCH] 完成: {file_path}")


if __name__ == "__main__":
    patch()
