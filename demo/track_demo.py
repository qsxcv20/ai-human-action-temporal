"""
track_demo.py — YOLOE 开放词汇检测 + 多目标跟踪（ByteTrack / BoT-SORT）

用途
----
把「用文字描述你要找的东西」（YOLOE 的开放词汇能力）和「跨帧保持同一个 ID」
（ByteTrack / BoT-SORT）串在一起，这是人物动作时序识别流水线的前半段：
    文字提示定位人物 -> 跟踪得到每个人的 ID -> 按 ID 裁剪人体区域 -> 动作分类

运行（在仓库的 venv 里，任意目录都可以跑）
------------------------------------------
    cd D:\\a i人物动作时序识别\\yoloe
    .\\.venv\\Scripts\\python.exe ..\\demo\\track_demo.py --source <视频路径>

注意：脚本在 demo/ 目录，但 ultralytics 必须用仓库那份（editable 安装指向 yoloe/ultralytics）；
cwd 由 common.ensure_repo_cwd() 自动切到仓库根，开放词汇的 mobileclip_blt.pt 才能加载。
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLOE

# 允许从任意目录以脚本方式运行（保证 import 到的是 demo/ 这份 common.py）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DEFAULT_CKPT_NAME,
    DEFAULT_OUT,
    REPO_ROOT,
    abs_path,
    ensure_repo_cwd,
    hr,
    kv,
    pad,
    resolve_checkpoint,
    resolve_tracker,
    restore_cwd,
    section,
)

DEFAULT_SOURCE = REPO_ROOT / "third_party" / "sam2" / "sav_dataset" / "example" / "sav_000001.mp4"


def parse_args():
    p = argparse.ArgumentParser(description="YOLOE 开放词汇检测 + 多目标跟踪")
    p.add_argument("--source", default=str(DEFAULT_SOURCE), help="输入视频/图片路径")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT_NAME, help="YOLOE 权重（文件名或路径）")
    p.add_argument(
        "--names",
        nargs="+",
        default=["person"],
        help="开放词汇提示词，可以写任意自然语言，如 'person' 'person wearing red'",
    )
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml | botsort.yaml | 本地 yaml")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--device", default="cuda:0", help="cuda:0 / cpu")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT), help="输出根目录")
    p.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0=全部")
    p.add_argument("--no-save", action="store_true", help="不保存视频，只要统计结果")
    return p.parse_args()


def main():
    args = parse_args()

    # 相对路径要在切 cwd 之前解析，否则切到仓库根后按原目录找文件会失败
    src = abs_path(args.source)
    out_root = abs_path(args.out_dir)
    try:
        ckpt = resolve_checkpoint(args.checkpoint)
        tracker = resolve_tracker(args.tracker)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    if not src.exists():
        raise SystemExit(f"输入不存在: {src}")

    # 每个源视频一个独立子目录，避免第二个视频把第一个的统计覆盖掉
    run_dir = out_root / "track" / src.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    section("YOLOE 开放词汇检测 + 多目标跟踪")
    kv("权重", ckpt)
    kv("输入", src)
    kv("提示词", " ".join(args.names))
    kv("跟踪器", tracker)
    kv("置信度", args.conf)
    kv("设备", args.device)
    kv("输出目录", run_dir)
    hr()

    old_cwd = ensure_repo_cwd()
    try:
        # 1) 加载 YOLOE 并把文字提示变成类别嵌入（开放词汇的关键一步）
        model = YOLOE(str(ckpt))
        model.set_classes(args.names, model.get_text_pe(args.names))

        # 2) 跑跟踪：persist=True 让跟踪器状态跨帧保持
        results = model.track(
            source=str(src),
            tracker=tracker,
            persist=True,
            conf=args.conf,
            device=args.device,
            stream=True,
            save=not args.no_save,
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
            verbose=False,
        )

        # 3) 统计：每个 ID 出现多少帧、平均置信度、面积中位数、帧区间
        id_frames = defaultdict(int)
        id_conf = defaultdict(list)
        id_area = defaultdict(list)
        id_first = {}
        id_last = {}
        total_frames = 0
        box_count = 0

        for r in results:
            total_frames += 1
            boxes = r.boxes
            if boxes is not None and len(boxes):
                ids = boxes.id
                confs = boxes.conf
                xyxy = boxes.xyxy.cpu().tolist()
                box_count += len(boxes)
                if ids is not None:
                    for tid, c, (x1, y1, x2, y2) in zip(ids.tolist(), confs.tolist(), xyxy):
                        tid = int(tid)
                        id_frames[tid] += 1
                        id_conf[tid].append(float(c))
                        id_area[tid].append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
                        id_first.setdefault(tid, total_frames)
                        id_last[tid] = total_frames
            if args.max_frames and total_frames >= args.max_frames:
                break
    finally:
        restore_cwd(old_cwd)

    hr()
    kv("处理帧数", total_frames)
    kv("检测框总数", box_count)
    kv("独立轨迹(人数)", len(id_frames))
    if id_frames:
        hr()
        headers = ["轨迹ID", "出现帧数", "平均置信度", "面积中位数", "帧区间"]
        widths = [10, 12, 12, 14, 12]
        print("  ".join(pad(h, widths[i]) for i, h in enumerate(headers)))
        for tid in sorted(id_frames, key=lambda t: -id_frames[t]):
            avg = sum(id_conf[tid]) / len(id_conf[tid])
            area = statistics.median(id_area[tid])
            cells = [
                str(tid),
                str(id_frames[tid]),
                f"{avg:.3f}",
                f"{area:.0f}px²",
                f"{id_first[tid]}-{id_last[tid]}",
            ]
            print("  ".join(pad(c, widths[i]) for i, c in enumerate(cells)))

    # 4) 落一份 JSON，方便后续接动作识别模块
    summary = {
        "source": str(src),
        "checkpoint": str(ckpt),
        "names": args.names,
        "tracker": tracker,
        "conf": args.conf,
        "device": args.device,
        "frames": total_frames,
        "boxes": box_count,
        "tracks": len(id_frames),
        "per_track": {
            str(t): {
                "frames": id_frames[t],
                "avg_conf": round(sum(id_conf[t]) / len(id_conf[t]), 4),
                "area_median": round(statistics.median(id_area[t]), 2),
                "first_frame": id_first[t],
                "last_frame": id_last[t],
            }
            for t in sorted(id_frames)
        },
    }
    js = run_dir / "track_summary.json"
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    hr()
    kv("统计结果", js)
    kv("标注视频", run_dir)
    print("（标注视频文件名与源视频同名；--no-save 时不产出视频）")


if __name__ == "__main__":
    main()
