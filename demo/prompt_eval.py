"""
prompt_eval.py — 开放词汇提示词「体检」工具

用途
----
同一段视频，换一个提示词，结果可能天差地别：
  "sports ball"        -> 大把误检（草丛里的纸屑、落叶全被认成球）
  "a big inflatable soccer ball" -> 干净命中，画面里那一个大号黑白足球

本脚本批量跑几个候选提示词，对比每条轨迹的：
  帧数        —— 覆盖了多少帧
  平均置信度  —— 模型有多确定
  面积中位数  —— **最关键的判据**：球是 ~4000px² 的大物，垃圾是 ~130px² 的小点
  帧区间      —— 目标在画面里的时间跨度

并按面积中位数给出「哪条才是真目标」的结论；结论和原始统计都落盘成 JSON，
方便把这次体检的结论直接喂给后面的动作识别流程。

用法（在 yoloe/ 的 venv 里）
--------------------------
    cd "D:\\a i人物动作时序识别\\yoloe"
    .\\.venv\\Scripts\\python.exe ..\\demo\\prompt_eval.py \
        --source ..\\demo\\ball.mp4 \
        --prompts "sports ball" "giant soccer ball" "a big inflatable soccer ball"

结果落盘: <out>/prompt_eval/<源视频名>_prompt_eval.json （可用 --json-out 覆盖）

注意：每换一个提示词必须新建 YOLOE 对象，否则提示词会被忽略（模型只认第一次的）；
每轮结束都会 del 模型并 torch.cuda.empty_cache() 释放显存，避免多提示词连跑时 OOM。
"""

import argparse
import gc
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from ultralytics import YOLOE

# 允许从任意目录以脚本方式运行（保证 import 到的是 demo/ 这份 common.py）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DEFAULT_CKPT_NAME,
    DEFAULT_OUT,
    DEMO_ROOT,
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

DEFAULT_SRC = DEMO_ROOT / "ball.mp4"
DEFAULT_TRACKER = DEMO_ROOT / "bytetrack_ball.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="开放词汇提示词批量体检")
    p.add_argument("--source", default=str(DEFAULT_SRC), help="输入视频")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT_NAME, help="YOLOE 权重（文件名或路径）")
    p.add_argument("--tracker", default=str(DEFAULT_TRACKER), help="跟踪器 yaml（内置名或本地路径）")
    p.add_argument("--prompts", nargs="+", required=True, help="候选提示词列表")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--device", default="cuda:0", help="cuda:0 / cpu")
    p.add_argument("--top", type=int, default=6, help="每个提示词打印前几条轨迹")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT), help="输出根目录")
    p.add_argument("--json-out", default="", help="结果 JSON 落盘路径（默认 <out>/prompt_eval/<源名>_prompt_eval.json）")
    return p.parse_args()


def run_prompt(prompt, args, ckpt, tracker):
    """跑完一条提示词，返回该提示词的统计字典。"""
    model = YOLOE(str(ckpt))
    try:
        model.set_classes([prompt], model.get_text_pe([prompt]))
        results = model.track(
            source=str(args.source),
            tracker=tracker,
            persist=True,
            conf=args.conf,
            device=args.device,
            stream=True,
            save=False,
            verbose=False,
        )
        stats = defaultdict(lambda: {"n": 0, "conf": [], "area": [], "first": None, "last": None})
        nbox = 0
        for i, r in enumerate(results):
            idx = i + 1
            b = r.boxes
            if b is None or not len(b):
                continue
            xyxy = b.xyxy.cpu().tolist()
            ids = b.id.cpu().tolist() if b.id is not None else [None] * len(b)
            confs = b.conf.cpu().tolist()
            for (x1, y1, x2, y2), tid, cf in zip(xyxy, ids, confs):
                nbox += 1
                if tid is None:
                    continue
                tid = int(tid)
                s = stats[tid]
                s["n"] += 1
                s["conf"].append(float(cf))
                s["area"].append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
                if s["first"] is None:
                    s["first"] = idx
                s["last"] = idx
    finally:
        # 多提示词连跑时显存会累积，每轮结束必须把模型和缓存都放掉
        del model
        gc.collect()
        torch.cuda.empty_cache()

    per_track = [
        {
            "track_id": tid,
            "frames": s["n"],
            "avg_conf": round(sum(s["conf"]) / len(s["conf"]), 4),
            "area_median": round(statistics.median(s["area"]), 2),
            "first_frame": s["first"],
            "last_frame": s["last"],
            "frame_range": [s["first"], s["last"]],
        }
        for tid, s in sorted(stats.items(), key=lambda kv_: -kv_[1]["n"])
    ]
    return {
        "prompt": prompt,
        "total_boxes": nbox,
        "track_count": len(per_track),
        "per_track": per_track,
        "conclusion": pick_target(per_track),
    }


def pick_target(per_track):
    """按面积中位数判断「哪条才是真目标」。

    面积中位数最大的那条才可能是目标本体；只有一两百像素的轨迹是误检的杂物。
    """
    if not per_track:
        return {
            "target_track_id": None,
            "target_area_median": None,
            "reason": "没有任何检测框，说明这个提示词在当前阈值下完全落空",
        }
    best = max(per_track, key=lambda t: t["area_median"])
    others = [t for t in per_track if t["track_id"] != best["track_id"]]
    if others:
        second = max(others, key=lambda t: t["area_median"])
        ratio = best["area_median"] / second["area_median"] if second["area_median"] > 0 else float("inf")
        reason = (
            f"轨迹 {best['track_id']} 的面积中位数 {best['area_median']:.0f}px² 最大，"
            f"是次大轨迹 {second['track_id']}（{second['area_median']:.0f}px²）的 {ratio:.1f} 倍，"
            f"这条最可能是真目标；其余轨迹面积只有一两百像素，是被误当成目标的杂物"
        )
    else:
        reason = (
            f"只有一条轨迹（id {best['track_id']}），面积中位数 {best['area_median']:.0f}px²，"
            f"没有别的候选可比对"
        )
    return {
        "target_track_id": best["track_id"],
        "target_area_median": best["area_median"],
        "target_frames": best["frames"],
        "target_frame_range": best["frame_range"],
        "reason": reason,
    }


def main():
    args = parse_args()

    # 相对路径要在切 cwd 之前解析成绝对路径
    args.source = abs_path(args.source)
    out_root = abs_path(args.out_dir)
    try:
        ckpt = resolve_checkpoint(args.checkpoint)
        tracker = resolve_tracker(args.tracker)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    if not Path(args.source).exists():
        raise SystemExit(f"输入不存在: {args.source}")

    json_out = (
        abs_path(args.json_out)
        if args.json_out
        else out_root / "prompt_eval" / f"{Path(args.source).stem}_prompt_eval.json"
    )

    section("开放词汇提示词体检")
    kv("视频", args.source)
    kv("权重", ckpt)
    kv("跟踪器", tracker)
    kv("置信度", args.conf)
    kv("设备", args.device)
    kv("结果 JSON", json_out)
    hr()

    results = []
    old_cwd = ensure_repo_cwd()
    try:
        for prompt in args.prompts:
            res = run_prompt(prompt, args, ckpt, tracker)
            results.append(res)

            print(f"\n提示词: \"{prompt}\"   总检测框 {res['total_boxes']}   轨迹数 {res['track_count']}")
            headers = ["轨迹ID", "帧数", "均conf", "面积中位数", "帧区间"]
            widths = [10, 10, 10, 14, 12]
            print("  ".join(pad(h, widths[i]) for i, h in enumerate(headers)))
            for t in res["per_track"][: args.top]:
                cells = [
                    str(t["track_id"]),
                    str(t["frames"]),
                    f"{t['avg_conf']:.3f}",
                    f"{t['area_median']:.0f}px²",
                    f"{t['first_frame']}-{t['last_frame']}",
                ]
                print("  ".join(pad(c, widths[i]) for i, c in enumerate(cells)))
            if res["track_count"] > args.top:
                print(f"  …… 其余 {res['track_count'] - args.top} 条轨迹见 JSON")
            print(f"  结论: {res['conclusion']['reason']}", flush=True)
    finally:
        restore_cwd(old_cwd)

    payload = {
        "source": str(args.source),
        "checkpoint": str(ckpt),
        "tracker": tracker,
        "conf": args.conf,
        "device": args.device,
        "prompts": results,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    hr()
    kv("结果 JSON", json_out)


if __name__ == "__main__":
    main()
