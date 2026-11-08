"""
_ex.py — 从跟踪结果视频里导出若干关键帧，方便肉眼核对 ID、框和掩膜

用途
----
track_demo.py 跑完会输出一支带标注的视频，肉眼检查时通常只关心几个时间点。
本脚本把指定帧导出成 jpg。

用法（在 yoloe/ 的 venv 里）
--------------------------
    cd "D:\\a i人物动作时序识别\\yoloe"
    .\\.venv\\Scripts\\python.exe ..\\demo\\_ex.py \
        --video "..\\demo\\out\\track\\sav_000001\\sav_000001.avi" \
        --frames 40 80 120 250 \
        --name sav_000001

所有路径都由参数指定，默认输出到 demo/out/，不再硬编码相对路径，
也不再「取最新生成的那个视频」——那样换个顺序跑就会抽错帧。
"""

import argparse
import sys
from pathlib import Path

import cv2

# 允许从任意目录以脚本方式运行（保证 import 到的是 demo/ 这份 common.py）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_OUT, DEMO_ROOT, abs_path, hr, kv, section  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="从视频里导出关键帧")
    p.add_argument("--video", required=True, help="要抽帧的视频（一般是 track_demo.py 输出的标注视频）")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT), help="图片输出目录")
    p.add_argument("--frames", nargs="+", type=int, default=[40, 80, 120, 250], help="要导出的帧号")
    p.add_argument("--name", default="", help="图片名前缀（默认取视频文件名）")
    return p.parse_args()


def list_candidates():
    """视频路径写错时，把可用的候选视频列出来，省得自己翻目录。"""
    track_root = DEMO_ROOT / "out" / "track"
    if not track_root.is_dir():
        return []
    vids = sorted(p for p in track_root.glob("**/*") if p.suffix.lower() in (".mp4", ".avi"))
    return vids


def main():
    args = parse_args()

    video = abs_path(args.video)
    out_dir = abs_path(args.out_dir)
    name = args.name or video.stem

    if not video.is_file():
        cands = list_candidates()
        msg = [f"视频不存在: {video}"]
        if cands:
            msg.append("可用的候选视频:")
            msg += [f"    {c}" for c in cands]
        else:
            msg.append(f"（{DEMO_ROOT / 'out' / 'track'} 下还没有任何视频，先跑 track_demo.py）")
        raise SystemExit("\n".join(msg))

    out_dir.mkdir(parents=True, exist_ok=True)

    section("导出关键帧")
    kv("视频", video)
    kv("输出目录", out_dir)
    kv("图片名前缀", name)
    hr()

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kv("总帧数", total)

    exported = 0
    for idx in args.frames:
        if total and idx >= total:
            print(f"跳过帧 {idx}: 超出视频长度 {total}")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"跳过帧 {idx}: 解码失败")
            continue
        dst = out_dir / f"{name}_f{idx:03d}.jpg"
        cv2.imwrite(str(dst), frame)
        exported += 1
        print(f"导出帧 {idx:<6} -> {dst}")
    cap.release()

    hr()
    kv("导出张数", exported)


if __name__ == "__main__":
    main()
