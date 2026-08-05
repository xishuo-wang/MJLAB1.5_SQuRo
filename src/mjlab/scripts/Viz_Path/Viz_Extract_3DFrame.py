"""从回放视频 (mp4) 中抽取指定帧保存为 PNG — 用于 3D 实景帧 (与 2D 俯视示意图组合)
用法:
  python Viz_Extract_3DFrame.py --video xxx.mp4 --frame 200 --out frame3d.png
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="从回放视频抽取指定帧")
    parser.add_argument("--video", required=True, help="mp4 视频路径")
    parser.add_argument("--frame", type=int, default=0, help="帧号 (从 0 开始)")
    parser.add_argument("--out", default=None, help="输出 PNG 路径 (默认: 视频同名_fN.png)")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        raise SystemExit("需要 opencv-python: pip install opencv-python")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频总帧数: {total}, fps: {fps:.2f} (第 {args.frame} 帧 ≈ {args.frame / fps:.2f}s)")

    if args.frame >= total:
        cap.release()
        raise SystemExit(f"帧号 {args.frame} 超出范围 [0, {total - 1}]")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"无法读取第 {args.frame} 帧")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = Path(args.out) if args.out else Path(args.video).with_suffix(f"_f{args.frame}.png")
    from PIL import Image
    Image.fromarray(frame_rgb).save(out)
    print(f"已保存 3D 帧: {out}  ({frame_rgb.shape[1]}x{frame_rgb.shape[0]})")


if __name__ == "__main__":
    main()
