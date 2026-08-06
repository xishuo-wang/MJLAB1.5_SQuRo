"""上下拼接组合图: 上 = 3D 实景帧 (MuJoCo 渲染), 下 = XoY 俯视示意 (Viz_Slalom_Reward 输出)
用法:
  python Viz_Combine_3D2D.py --top frame3d.png --bottom corridor_slalom.png --out combined.png
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="上下拼接 3D 实景与 2D 俯视示意图")
    parser.add_argument("--top", required=True, help="3D 实景图 PNG")
    parser.add_argument("--bottom", required=True, help="2D 俯视示意图 PNG")
    parser.add_argument("--out", default="combined_3d2d.png", help="输出 PNG 路径")
    parser.add_argument("--gap", type=int, default=8, help="上下图间隙像素")
    parser.add_argument("--scale", type=float, default=1.0, help="输出缩放 (默认 1.0)")
    args = parser.parse_args()

    from PIL import Image
    top = Image.open(args.top).convert("RGB")
    bot = Image.open(args.bottom).convert("RGB")

    # 统一宽度 (以较宽者为准), 等比缩放高度
    W = max(top.width, bot.width)
    def fit(img):
        if img.width == W:
            return img
        return img.resize((W, max(1, round(img.height * W / img.width))))
    top2, bot2 = fit(top), fit(bot)

    H = top2.height + args.gap + bot2.height
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(top2, (0, 0))
    canvas.paste(bot2, (0, top2.height + args.gap))

    if args.scale != 1.0:
        canvas = canvas.resize((max(1, round(W * args.scale)), max(1, round(H * args.scale))))
    out = Path(args.out)
    canvas.save(out)
    print(f"已保存组合图: {out}  ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
