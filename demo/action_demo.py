"""
action_demo.py — YOLOE 开放词汇检测 + ByteTrack 跟踪 + 零样本动作分类

这是「AI 人物动作时序识别」流水线的完整闭环：

    文字提示定位人物  →  跟踪保持每人一个 ID  →  按 ID 裁剪人体  →  逐段零样本判动作

三段能力各由一个模型承担：
    YOLOE        开放词汇检测（要追的目标由自然语言指定）+ 实例分割掩膜
    ByteTrack    跨帧 ID 关联（人走了再回来还是同一个 ID）
    X-CLIP       零样本视频分类（动作标签由自然语言指定，不用训练）

运行（在 yoloe 仓库的 venv 里）
------------------------------
    cd D:\\a i人物动作时序识别\\yoloe
    .\\.venv\\Scripts\\python.exe ..\\demo\\action_demo.py

注意：脚本在 demo/ 目录，但 ultralytics 必须用仓库那份（editable 安装指向 yoloe/ultralytics）。
"""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLOE

# ---------------------------------------------------------------- 路径推导
REPO = Path(__file__).resolve().parent.parent / "yoloe"
DEMO = Path(__file__).resolve().parent

DEFAULT_CKPT = REPO / "pretrain" / "yoloe-11s-seg.pt"
DEFAULT_SOURCE = REPO / "third_party" / "sam2" / "sav_dataset" / "example" / "sav_000001.mp4"

# 默认动作候选：零样本分类就是拿视频片段和这组文本比相似度
DEFAULT_ACTIONS = [
    "doing a handstand",
    "standing still",
    "walking",
    "running",
    "jumping",
    "squatting",
]


def parse_args():
    p = argparse.ArgumentParser(description="YOLOE 检测 + 跟踪 + 零样本动作分类")
    p.add_argument("--source", default=str(DEFAULT_SOURCE), help="输入视频路径")
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT), help="YOLOE 权重路径")
    p.add_argument("--names", nargs="+", default=["person"], help="检测提示词（要追的目标）")
    p.add_argument("--labels", nargs="+", default=DEFAULT_ACTIONS, help="动作候选标签（可任意自然语言）")
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml | botsort.yaml")
    p.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    p.add_argument("--device", default="cuda:0", help="cuda:0 / cpu")
    p.add_argument("--out-dir", default=str(DEMO / "out"), help="输出根目录")
    p.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0=全部")
    p.add_argument("--stride", type=int, default=4, help="每隔多少帧做一次动作分类")
    p.add_argument("--margin", type=float, default=0.15, help="人体裁剪外扩比例")
    p.add_argument("--crop-mode", choices=["mask_bbox", "box"], default="mask_bbox",
                   help="mask_bbox=用实例分割掩膜定位人体轮廓（YOLOE-seg 优势）；box=直接用检测框")
    p.add_argument("--xclip", default="microsoft/xclip-base-patch32", help="视频分类模型")
    return p.parse_args()


def crop_person(img, xyxy, masks, i, margin, crop_mode):
    """裁出第 i 个人体区域，返回 RGB 小图。掩膜用来精确定位人体轮廓，比检测框更贴合。"""
    H, W = img.shape[:2]
    if crop_mode == "mask_bbox" and masks is not None and len(masks) == len(xyxy):
        m = masks[i]
        ys, xs = np.where(m > 0.5)
        if len(xs) > 0:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        else:
            x1, y1, x2, y2 = xyxy[i]
    else:
        x1, y1, x2, y2 = xyxy[i]

    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    w, h = x2 - x1, y2 - y1
    x1 = max(0.0, x1 - w * margin)
    y1 = max(0.0, y1 - h * margin)
    x2 = min(float(W - 1), x2 + w * margin)
    y2 = min(float(H - 1), y2 + h * margin)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[int(y1):int(y2) + 1, int(x1):int(x2) + 1]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def main():
    args = parse_args()

    ckpt = Path(args.checkpoint)
    src = Path(args.source)
    if not ckpt.exists():
        raise SystemExit(f"权重不存在: {ckpt}")
    if not src.exists():
        raise SystemExit(f"输入不存在: {src}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 64)
    print("YOLOE 检测 + ByteTrack 跟踪 + X-CLIP 零样本动作分类")
    print("=" * 64)
    print(f"权重    : {ckpt}")
    print(f"输入    : {src}")
    print(f"检测提示: {args.names}")
    print(f"动作标签: {args.labels}")
    print(f"跟踪器  : {args.tracker}   设备: {device}")
    print("-" * 64)

    # ------------------------------------------------------------ 1) 检测器
    model = YOLOE(str(ckpt))
    model.set_classes(args.names, model.get_text_pe(args.names))

    # ------------------------------------------------------------ 2) 分类器
    from transformers import AutoProcessor, XCLIPModel

    processor = AutoProcessor.from_pretrained(args.xclip)
    clip_model = XCLIPModel.from_pretrained(args.xclip).to(device).eval()
    video_proc = getattr(processor, "video_processor", processor)
    num_frames = int(getattr(video_proc, "num_frames", 8))
    print(f"动作模型: {args.xclip}  (每段 {num_frames} 帧)")

    @torch.no_grad()
    def classify(frames):
        """frames: 长度=num_frames 的 RGB 小图列表 → 返回 (标签, 概率)"""
        inputs = processor(text=args.labels, videos=[frames], return_tensors="pt", padding=True)
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        probs = clip_model(**inputs).logits_per_video.softmax(dim=1)[0]
        idx = int(probs.argmax())
        return args.labels[idx], float(probs[idx])

    # ------------------------------------------------------------ 3) 跟踪 + 分类
    results = model.track(
        source=str(src),
        tracker=args.tracker,
        persist=True,
        conf=args.conf,
        device=args.device,
        stream=True,
        verbose=False,
    )

    buffers = defaultdict(lambda: deque(maxlen=64))   # id -> [(帧号, 裁剪图)]
    last_cls = defaultdict(int)                      # id -> 上次分类的帧号
    timeline = defaultdict(list)                     # id -> [{frame, label, score}]
    latest = {}                                      # id -> (label, score)

    out_dir = Path(args.out_dir) / "action"
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_out = out_dir / (src.stem + "_action.mp4")
    writer = None
    frame_no = 0

    for r in results:
        frame_no += 1
        img = r.orig_img
        boxes = r.boxes
        xyxy = ids = masks = None
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy))
            if r.masks is not None and len(r.masks.data):
                masks = r.masks.data.cpu().numpy()

            for i, tid in enumerate(ids):
                tid = int(tid)
                crop = crop_person(img, xyxy, masks, i, args.margin, args.crop_mode)
                if crop is None:
                    continue
                buffers[tid].append((frame_no, crop))

                # 攒够一个片段且到了 stride 间隔，就做一次动作分类
                if len(buffers[tid]) >= num_frames and frame_no - last_cls[tid] >= args.stride:
                    frames = [c for _, c in list(buffers[tid])[-num_frames:]]
                    try:
                        label, score = classify(frames)
                    except Exception as e:                      # 分类失败不要拖垮整条流水线
                        print(f"  [warn] id={tid} 分类失败: {e}")
                        continue
                    timeline[tid].append({"frame": frame_no, "label": label, "score": round(score, 4)})
                    latest[tid] = (label, score)
                    last_cls[tid] = frame_no

        # ------------------------------------------------------ 4) 画标注视频
        canvas = img.copy()
        if boxes is not None and len(boxes):
            for i, tid in enumerate(ids):
                tid = int(tid)
                x1, y1, x2, y2 = xyxy[i].astype(int)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                tag = f"id:{tid}"
                if tid in latest:
                    lab, sc = latest[tid]
                    tag += f" {lab} {sc:.2f}"
                cv2.putText(canvas, tag, (x1, max(16, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        if writer is None:
            h, w = canvas.shape[:2]
            fps = 24.0
            writer = cv2.VideoWriter(str(vid_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(canvas)

        if args.max_frames and frame_no >= args.max_frames:
            break

    if writer is not None:
        writer.release()

    # ------------------------------------------------------------ 5) 汇总输出
    print("-" * 64)
    print(f"处理帧数   : {frame_no}")
    print(f"独立轨迹数 : {len(buffers)}")

    summary = {
        "source": str(src),
        "checkpoint": str(ckpt),
        "names": args.names,
        "labels": args.labels,
        "tracker": args.tracker,
        "frames": frame_no,
        "tracks": len(buffers),
        "per_track": {},
    }

    for tid in sorted(buffers, key=lambda t: -len(buffers[t])):
        segs = timeline.get(tid, [])
        # 合并连续同标签的片段，得到「动作时序」
        merged = []
        for s in segs:
            if merged and merged[-1]["label"] == s["label"]:
                merged[-1]["end_frame"] = s["frame"]
                merged[-1]["score"] = round((merged[-1]["score"] + s["score"]) / 2, 4)
                merged[-1]["hits"] += 1
            else:
                merged.append({"start_frame": s["frame"], "end_frame": s["frame"],
                               "label": s["label"], "score": s["score"], "hits": 1})
        # 只保留置信度最高的主动作
        best = max(merged, key=lambda m: m["score"]) if merged else None
        summary["per_track"][str(tid)] = {
            "detected_frames": len(buffers[tid]),
            "timeline": merged,
            "dominant_action": best["label"] if best else None,
            "dominant_score": best["score"] if best else None,
        }
        print(f"  id:{tid:<4} 出现 {len(buffers[tid]):>4} 帧  主动作: {best['label'] if best else '-'}"
              f" ({best['score']:.2f})" if best else f"  id:{tid:<4} 出现 {len(buffers[tid]):>4} 帧")

    js = out_dir / "action_summary.json"
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-" * 64)
    print(f"统计结果: {js}")
    print(f"标注视频: {vid_out}")


if __name__ == "__main__":
    main()
