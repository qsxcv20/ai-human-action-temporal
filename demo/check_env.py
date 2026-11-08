"""uv 环境部署后的验收检查：逐项确认关键包、CUDA 与本地可编辑安装是否正常。"""

import importlib
import sys

CHECKS = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("opencv-python", "cv2"),
    ("ultralytics", "ultralytics"),
    ("supervision", "supervision"),
    ("lap", "lap"),
    ("lvis", "lvis"),
    ("mobileclip", "mobileclip"),
    ("clip", "clip"),
    ("timm", "timm"),
    ("gradio", "gradio"),
]

print(f"Python  : {sys.version.split()[0]}  ({sys.executable})")
print("-" * 62)

ok = True
for label, mod in CHECKS:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "-")
        loc = getattr(m, "__file__", "-")
        print(f"[OK]   {label:<16} {ver:<14} {loc}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {label:<16} {type(e).__name__}: {e}")

print("-" * 62)

try:
    import torch

    print(f"torch           : {torch.__version__}")
    print(f"cuda built      : {torch.version.cuda}")
    print(f"cuda available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu             : {torch.cuda.get_device_name(0)}")
        print(f"capability      : {torch.cuda.get_device_capability(0)}")
        t = torch.zeros(8, device="cuda")
        print(f"gpu tensor test : {t.sum().item()} (期望 0.0)")
except Exception as e:  # noqa: BLE001
    ok = False
    print(f"[FAIL] torch/cuda 检查异常: {e}")

print("-" * 62)
print("ultralytics 来源 :", importlib.import_module("ultralytics").__file__)
print("结果             :", "全部通过" if ok else "存在失败项")
sys.exit(0 if ok else 1)
